"""
Coordinated PGD: a Madry-style projected-gradient adversary whose perturbation is
optimised JOINTLY across every compromised agent against a GLOBAL objective.

Why this exists
---------------
The FGSM study was a negative result: a myopic per-agent gradient flips ~26% of
routing decisions but extracts almost none of the ~21pp damage ceiling, because the
K-path redundancy absorbs the flips. The learned (SA-MDP / DDPG) adversary was meant
to test whether that robustness is real, but it does not train: the per-step packet
loss reward has a signal-to-noise ratio of ~0.008 w.r.t. the adversary's action, so
the critic never becomes action-dependent and the actor collapses to a single
saturated direction (see tools/diagnose_adv.py).

Madry et al. (2017) argue the inner maximisation  max_{delta in S} L(x+delta)  is
best solved by direct multi-step first-order optimisation, NOT by amortising it into
a learned policy. That side-steps the reward-noise problem entirely: PGD uses
analytic gradients of a differentiable objective, so there is no credit assignment
and no reward signal to drown out.

What makes it COORDINATED
-------------------------
`FGSMAttackFramework.generate_adversarial_state_critic` already grounds the attack in
the victim's own centralised critic, but it perturbs ONE agent at a time and holds the
other agents' actions fixed at their clean block-onehot values. The gradient it
computes therefore answers "how do I make agent i's action look worst, given everyone
else stays put" — it structurally cannot express "push agent i AND agent j onto the
same surviving link".

Here every compromised agent's observation is perturbed simultaneously and ALL of
their actions vary through a single shared objective, so one gradient step trades the
agents off against each other. That is the mechanism the per-agent gradient cannot
express, and the one that should actually reach the damage ceiling if it is reachable
at all.

Fidelity to Madry et al.
------------------------
  * multi-step ascent with L-inf projection back onto the epsilon-ball each step
  * RANDOM START: delta is initialised uniformly in the ball (the paper's PGD; the
    existing `improved_fgsm_attack` path starts from the clean point, i.e. BIM)
  * RANDOM RESTARTS: several independent starts, keeping the best final objective
    (their Table 1: 40 steps/1 restart 93.2% -> 100 steps/20 restarts 89.3%)
  * step size defaults to alpha = 2.5 * epsilon / n_steps, the rule from the paper
    that guarantees the boundary is reachable from any interior start

Eval: exposes `generate_adversarial_state(...)` with the SAME signature as
FGSMAttackFramework, so it drops straight into `_attack_episodes` and is scored by the
existing paired clean/random/learned harness.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CoordinatedPGDAdversary:
    """Joint multi-agent PGD in the observation space.

    Parameters
    ----------
    maddpg, env
        The FROZEN victim and its environment. Only forward passes are used; no
        victim parameter is ever updated.
    trainable_indices
        Topology indices carrying a learning actor. Position i in this list is the
        `agent_index` the runner passes for that host, so the joint vector's block
        order matches the runner's call site exactly.
    objective
        'critic'     — ascend -Q on the victim's own centralised critics (mean over
                       the compromised agents). Grounded in the victim's value
                       model, so it cannot accidentally *improve* routing.
        'congestion' — maximise the true bottleneck utilisation of the paths the
                       victim is steered onto, summed over agents. Directly encodes
                       the shared-bottleneck hypothesis. Utilisation coefficients
                       come from the CLEAN observation, never the perturbed one
                       (otherwise the attacker would be optimising against its own
                       forged view rather than the real network).
    """

    def __init__(self, maddpg, env, trainable_indices: Sequence[int],
                 epsilon: float = 0.30, n_steps: int = 10,
                 alpha: Optional[float] = None, random_start: bool = True,
                 n_restarts: int = 1, objective: str = "critic",
                 bandwidth_indices: Optional[Sequence[int]] = None,
                 seed: int = 20240601):
        if getattr(maddpg, "critic_type", None) != "central_critic" and objective == "critic":
            raise ValueError(
                "objective='critic' needs a central_critic variant (the joint action "
                "has to enter one shared value function); use objective='congestion' "
                f"for critic_type={getattr(maddpg, 'critic_type', None)!r}")
        if getattr(maddpg, "gnn_processor", None) is not None:
            raise ValueError(
                "GNN variants encode all agents' observations jointly; the "
                "differentiable per-agent path this attack needs is not wired for "
                "them (same restriction as the critic-grounded FGSM attack)")

        self.maddpg = maddpg
        self.env = env
        self.trainable_indices = list(trainable_indices)
        self.epsilon = float(epsilon)
        self.n_steps = max(1, int(n_steps))
        # Madry's rule: large enough to cross the ball from any interior start.
        self.alpha = float(alpha) if alpha else 2.5 * self.epsilon / self.n_steps
        self.random_start = bool(random_start)
        self.n_restarts = max(1, int(n_restarts))
        self.objective = objective
        self.attack_type = "coordinated_pgd"   # API parity with FGSMAttackFramework

        engine = env.engine
        hosts = engine.get_all_hosts()
        self.group_hosts = [hosts[i] for i in self.trainable_indices]
        self.n_agents = len(self.trainable_indices)
        self.n_actions = maddpg.n_actions
        self.n_dest = getattr(engine, "n_destinations", 1)
        self.block_size = max(1, self.n_actions // max(1, self.n_dest))
        self.obs_dim = len(engine.get_state(self.group_hosts[0]))
        self.device = maddpg.agents[0].actor.device
        self.bandwidth_indices = (np.asarray(list(bandwidth_indices), dtype=np.int64)
                                  if bandwidth_indices else None)
        self.rng = np.random.default_rng(seed)

        # Where the action-aligned K-path bottleneck utilisations live in the
        # observation. Anchored on the START of the block, deliberately: get_state
        # builds mn + 3 + n_dest + 3 + n_dest*K + 1 slots but then truncates to
        # state_dims = mn + n_dest + 7 - K + K*n_dest, which is K SHORTER. The
        # trailing mean-hops slot and the last K-1 util slots are therefore cut off,
        # so counting back from the end lands in the wrong place.
        #   [0:mn] neighbour bandwidth | [mn:mn+3] queue, diversity, time
        #   [mn+3:mn+3+n_dest] destination one-hot | next 3: mean/max/var adj util
        #   then the per-destination K-path bottleneck utils (possibly truncated)
        self._util_lo = engine.max_neighbors + self.n_dest + 6
        self._util_hi = min(self._util_lo + self.n_dest * self.block_size, self.obs_dim)
        self._n_util = max(0, self._util_hi - self._util_lo)
        if self._n_util < self.n_dest * self.block_size:
            logger.info(
                "[coord-pgd] observation truncates the K-path util block to %d of %d "
                "slots; the congestion objective scores the first %d action slots.",
                self._n_util, self.n_dest * self.block_size, self._n_util)
        if self.objective == "congestion" and self._n_util == 0:
            raise ValueError("congestion objective needs the K-path util block, but "
                              "the observation layout exposes none of it")

        # one joint optimisation per timestep, shared by every agent that calls in
        self._cache_step = None
        self._cache_blocks: Optional[List[np.ndarray]] = None

        self.attack_stats: Dict = {"total_attacks": 0, "attack_success_count": 0}

    # ── projection ────────────────────────────────────────────────────────────
    def _project(self, orig: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
        """L-inf ball around orig, then the domain clamp. Shapes are [n_agents, d]."""
        adv = orig + torch.clamp(adv - orig, -self.epsilon, self.epsilon)
        if self.bandwidth_indices is not None:
            idx = torch.as_tensor(self.bandwidth_indices, device=adv.device)
            adv[:, idx] = adv[:, idx].clamp(0.0, 1.0)
        else:
            adv = adv.clamp(0.0, 1.0)
        return adv

    @staticmethod
    def _straight_through_block_onehot(a_soft: torch.Tensor, block_size: int) -> torch.Tensor:
        """Per-block argmax one-hot, straight-through gradient.

        The environment decodes routing by taking one path per K_PATHS block, and the
        central critic was trained on exactly those block-projected actions. Forward
        is therefore the hard one-hot; backward passes the soft actor gradient so the
        argmax does not kill the signal (the routing decision is piecewise constant
        in the observation, so without this the gradient would be zero a.e.).
        """
        n = a_soft.shape[-1]
        hard = torch.zeros_like(a_soft)
        rows = torch.arange(a_soft.shape[0], device=a_soft.device)
        for bs in range(0, n, block_size):
            be = min(bs + block_size, n)
            idx = a_soft[:, bs:be].argmax(dim=1)
            hard[rows, bs + idx] = 1.0
        return (hard - a_soft).detach() + a_soft

    # ── the joint objective ───────────────────────────────────────────────────
    def _joint_actions(self, x: torch.Tensor) -> torch.Tensor:
        """[n_agents, obs_dim] -> [n_agents, n_actions], differentiable in x.

        Every compromised agent's actor is evaluated on its OWN perturbed row, and
        all rows stay in one graph, so a single backward pass gives the gradient of
        the shared objective w.r.t. EVERY agent's perturbation at once. This is
        where the coordination happens.
        """
        outs = []
        for pos, topo_idx in enumerate(self.trainable_indices):
            agent = self.maddpg.agents[pos]
            a_soft = agent.actor(x[pos].unsqueeze(0))            # [1, n_actions]
            outs.append(self._straight_through_block_onehot(a_soft, self.block_size))
        return torch.cat(outs, dim=0)                            # [n_agents, n_actions]

    def _objective_value(self, x: torch.Tensor, central_state: torch.Tensor,
                         clean_util: torch.Tensor) -> torch.Tensor:
        """Scalar the attacker MAXIMISES (higher = worse for the victim)."""
        a_joint = self._joint_actions(x)

        if self.objective == "critic":
            joint_flat = a_joint.view(1, -1)                     # [1, n_agents*n_actions]
            # Mean of the compromised agents' own value estimates. Ascending -Q
            # steers the joint action to what the victim's value model itself rates
            # as worst, so it cannot accidentally improve routing.
            qs = [self.maddpg.agents[pos].critic(central_state, joint_flat)
                  for pos in range(self.n_agents)]
            return -torch.stack(qs).mean()

        if self.objective == "congestion":
            # clean_util is [n_agents, n_util], action-aligned: slot k of destination
            # d is the TRUE bottleneck utilisation of path k (taken from the clean
            # observation, never the perturbed one — otherwise the attacker would be
            # optimising against its own forged view). Weighting it by the
            # straight-through chosen action gives the utilisation of the path the
            # victim is steered onto; summing over agents rewards piling SEVERAL
            # agents onto the SAME congested link, the shared-bottleneck mechanism
            # under test. Action slots are aligned 1:1 with util slots, so the
            # truncated tail is simply not scored.
            chosen = a_joint[:, :clean_util.shape[1]]
            return (chosen * clean_util).sum()

        raise ValueError(f"unknown objective {self.objective!r}")

    # ── the attack ────────────────────────────────────────────────────────────
    def _optimise_joint(self, clean_states: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Run PGD on the concatenated perturbation; return per-agent adv observations."""
        engine = self.env.engine
        O = torch.tensor(np.asarray(clean_states, dtype=np.float32), device=self.device)

        # True global state, held FIXED during the attack: the adversary perturbs
        # what the agents SEE, never the network itself.
        cs = torch.tensor(
            np.asarray(engine.get_central_state(self.group_hosts), dtype=np.float32)[None, :],
            device=self.device)
        clean_util = O[:, self._util_lo:self._util_hi].detach()

        actors_training = [a.actor.training for a in self.maddpg.agents]
        critics_training = [a.critic.training for a in self.maddpg.agents]
        for a in self.maddpg.agents:
            a.actor.eval()
            a.critic.eval()

        best_x, best_val = None, -float("inf")
        try:
            with torch.enable_grad():
                for _restart in range(self.n_restarts):
                    if self.random_start:
                        # uniform in the ball — Madry's PGD, not BIM
                        d0 = self.rng.uniform(-self.epsilon, self.epsilon, size=O.shape)
                        x = self._project(O, O + torch.tensor(d0, dtype=torch.float32,
                                                              device=self.device))
                    else:
                        x = O.clone()
                    x = x.detach().requires_grad_(True)

                    for _step in range(self.n_steps):
                        if x.grad is not None:
                            x.grad.zero_()
                        val = self._objective_value(x, cs, clean_util)
                        val.backward()
                        if x.grad is None:
                            raise RuntimeError("coordinated PGD produced no gradient")
                        with torch.no_grad():
                            x_new = x + self.alpha * torch.sign(x.grad)
                            x_new = self._project(O, x_new)
                        x = x_new.detach().requires_grad_(True)

                    with torch.no_grad():
                        final = float(self._objective_value(x, cs, clean_util))
                    if final > best_val:      # keep the best restart, as in the paper
                        best_val, best_x = final, x.detach()
        except Exception as e:                # never let an attack failure kill a run
            logger.error(f"coordinated PGD failed: {e}")
            return [np.asarray(s, dtype=np.float32) for s in clean_states]
        finally:
            for a, was in zip(self.maddpg.agents, actors_training):
                if was:
                    a.actor.train()
            for a, was in zip(self.maddpg.agents, critics_training):
                if was:
                    a.critic.train()

        self.attack_stats["total_attacks"] += 1
        return [row.cpu().numpy() for row in best_x]

    # ── FGSM-compatible entry point ───────────────────────────────────────────
    def generate_adversarial_state(self, state, agent_network=None,
                                   network_engine=None, agent_index: int = 0,
                                   bandwidth_indices=None) -> np.ndarray:
        """Called once per compromised agent per step by `_attack_episodes`.

        The joint optimisation is run once per timestep (keyed on the engine's own
        step counter) and each caller receives its own block, so N agents cost ONE
        PGD solve rather than N — and, more importantly, they all act on the SAME
        jointly-optimised perturbation instead of N independent ones.
        """
        engine = network_engine if network_engine is not None else self.env.engine
        step = getattr(engine, "time_step", None)
        if step is None or step != self._cache_step:
            clean = [engine.get_state(h) for h in self.group_hosts]
            self._cache_blocks = self._optimise_joint(clean)
            self._cache_step = step
        return self._cache_blocks[agent_index]
