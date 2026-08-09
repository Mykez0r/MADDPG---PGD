"""
Learned worst-case observation adversary (SA-MDP / SA-DDPG style) for the MADDPG
routing victim.  This is the *scaffold* for the follow-up MSc project: it turns the
myopic single-step FGSM attack into a trained adversary whose objective is the
victim's packet loss over whole trajectories, so we can test whether the robustness
measured under FGSM is fundamental or just an artefact of a weak (myopic, per-agent)
attacker.

Design (mirrors the paper's threat model exactly, so results stay comparable):
  * Threat model: observation-space, L-inf bounded by epsilon, bandwidth/utilisation
    features re-projected to [0,1]. Identical ball to FGSMAttackFramework.
  * Adversary: a deterministic policy  pi_adv(obs) -> delta  in the epsilon-ball,
    trained by DDPG with the *victim's per-step packet loss* as reward (SA-MDP: the
    optimal observation adversary is itself an RL agent, Zhang et al. NeurIPS 2020).
  * Victim: a FROZEN trained MADDPG variant. The adversary never sees the victim's
    weights except through forward passes at attack time (grey-box); set
    `white_box=True` to also let the DDPG critic condition on victim internals.

Two extension points make the attack *specific to this routing scenario* and are
the intended research contributions:
  (A) COORDINATED multi-agent perturbation  (steer flows onto a SHARED bottleneck).
      IMPLEMENTED — set AdversaryConfig.coordinate=True (`--coordinate`). A single
      actor consumes the concatenation of every compromised agent's observation
      and emits one joint delta; the critic additionally conditions on the GLOBAL
      link state (NetworkEngine.get_central_state). See
      AdversaryTrainer._attack_states_joint and
      LearnedObservationAdversary.perturb_joint / generate_adversarial_state.
  (B) STRATEGICALLY-TIMED / critical-state attacks  (spend an L0 budget only at the
      high-leverage moments — congestion onset, immediately post-failure).
      IMPLEMENTED — set AdversaryConfig.timing_budget=<fraction> (`--timing-budget`).
      A TimingGate scores every step by the network's own current max link
      utilisation (this is exactly where congestion onset and post-failure
      pressure show up — see the FGSM fail4/fail6 sweep, where robustness
      collapses once redundancy is thin and surviving links saturate) and only
      attacks the highest-scoring `timing_budget` fraction of steps, calibrated
      online against a rolling window of recent scores so the realised attack
      rate tracks the budget without needing foresight into the rest of the
      episode. Skipped steps pass the clean state through untouched and push no
      replay transition. See TimingGate, AdversaryTrainer._attack_states, and
      LearnedObservationAdversary.generate_adversarial_state.

Eval: `LearnedObservationAdversary` exposes `generate_adversarial_state(...)` with
the SAME signature as FGSMAttackFramework, so a trained adversary drops straight
into `standalone_experiment_runner._attack_episodes` via attack_type='learned'
and is scored by the existing damage-ceiling / random-control / action-flip metrics.

Run the trainer with  tools/train_adversary.py  (wires up env + frozen victim).
"""
from __future__ import annotations

import logging
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _log_timing_summary(tag: str, counts: "Counter", budget: float, window: int) -> None:
    """(B) Flush one aggregated line for a window of episodes' worth of timing-gate
    decisions instead of logging every fire — tallies WHEN (fired/skipped out of
    total steps) and WHY (congestion severity band of the fired steps), then
    resets `counts` for the next window."""
    fired = counts.get("fired", 0)
    total = fired + counts.get("skipped", 0)
    if total == 0:
        return
    logger.info(
        "%s last %d eps: attacked %d/%d steps (%.1f%%, budget %.0f%%)  "
        "reasons: near-saturation=%d  high-congestion=%d  at-threshold=%d",
        tag, window, fired, total, 100.0 * fired / total, budget * 100,
        counts.get("near-saturation", 0), counts.get("high-congestion", 0),
        counts.get("at-threshold", 0))
    counts.clear()


# ─────────────────────────── networks ────────────────────────────────────────
class AdversaryActor(nn.Module):
    """obs -> perturbation direction in [-1,1]^d (scaled by epsilon at apply time)."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class AdversaryCritic(nn.Module):
    """Q(obs, delta[, link_state]) for the DDPG update of the adversary.

    link_state_dim > 0 conditions the critic on the GLOBAL link state (see
    NetworkEngine.get_central_state: per-edge available bandwidth + per-agent
    modal destination) in addition to the observation/delta it is scoring.
    This is what lets a coordinated adversary (cfg.coordinate=True) learn to
    push several flows onto ONE shared link that is about to bottleneck,
    rather than judging each agent's perturbation against its own local view.
    """

    def __init__(self, obs_dim: int, hidden: int = 256, link_state_dim: int = 0):
        super().__init__()
        self.link_state_dim = link_state_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2 + link_state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, delta: torch.Tensor,
                link_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        parts = [obs, delta]
        if self.link_state_dim:
            if link_state is None:
                raise ValueError("critic was built with link_state_dim > 0 but "
                                  "forward() got no link_state")
            parts.append(link_state)
        return self.net(torch.cat(parts, dim=-1))


# ─────────────────────────── replay ──────────────────────────────────────────
class ReplayBuffer:
    """Transitions optionally carry the GLOBAL link state (coordinated mode
    only; None in independent per-agent mode) alongside the usual SARS'd
    tuple, so the critic can condition on it during `_update`."""

    def __init__(self, capacity: int = 200_000):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, obs, delta, reward, next_obs, done, link_state=None, next_link_state=None):
        self.buf.append((obs, delta, reward, next_obs, done, link_state, next_link_state))

    def sample(self, batch: int, device):
        idx = random.sample(range(len(self.buf)), batch)
        obs, delta, r, nobs, done, ls, nls = zip(*(self.buf[i] for i in idx))
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
        link_state = t(ls) if ls[0] is not None else None
        next_link_state = t(nls) if nls[0] is not None else None
        return (t(obs), t(delta), t(r).unsqueeze(-1), t(nobs), t(done).unsqueeze(-1),
                link_state, next_link_state)

    def __len__(self):
        return len(self.buf)


# ─────────────────────────── timing gate ──────────────────────────────────────
class TimingGate:
    """(B) L0-budget timing gate: attack only the `budget` fraction of steps with
    the highest critical-state score, so a limited attack budget is spent on
    high-leverage moments (congestion onset, immediately post-failure) instead of
    spread thin over every step.

    Score = current max link utilisation across the whole topology — congestion
    and thinned redundancy (post-failure) both surface there directly. The
    trigger threshold is the (1 - budget) quantile of a rolling window of scores,
    recalibrated online after every step, so the realised attack rate tracks
    `budget` over time without needing to see the rest of the episode in advance.
    The first `min_history` steps always fire, to seed the window before the
    quantile means anything.
    """

    def __init__(self, budget: float, window: int = 1000, min_history: int = 100):
        self.budget = float(budget)
        self.min_history = min_history
        self.history: deque = deque(maxlen=window)

    # score bands for the "why" breakdown in the periodic summary — coarse
    # severity of the congestion the gate reacted to, not a precise cause
    _SEVERITY_BANDS = (("near-saturation", 0.90), ("high-congestion", 0.75))

    @staticmethod
    def score(network_engine) -> float:
        topo = network_engine.topology
        edges = topo.graph.edges()
        return max((topo.get_util(u, v) for u, v in edges), default=0.0)

    @classmethod
    def severity(cls, score: float) -> str:
        for label, cutoff in cls._SEVERITY_BANDS:
            if score >= cutoff:
                return label
        return "at-threshold"

    def should_attack(self, network_engine) -> Tuple[bool, float, float]:
        """Returns (fire, score, threshold). `threshold` is computed from history
        BEFORE this step's score is appended, so a step never influences its own
        cutoff."""
        s = self.score(network_engine)
        if len(self.history) < self.min_history:
            threshold = 0.0
            fire = True
        else:
            threshold = float(np.quantile(self.history, 1.0 - self.budget))
            fire = s >= threshold
        self.history.append(s)
        return fire, s, threshold


# ─────────────────────────── config ──────────────────────────────────────────
@dataclass
class AdversaryConfig:
    epsilon: float = 0.30            # L-inf budget (match the FGSM sweep)
    hidden: int = 256
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.95
    tau: float = 5e-3                # target soft-update
    batch_size: int = 256
    warmup_steps: int = 2_000
    updates_per_step: int = 1
    explore_noise: float = 0.10      # exploration noise on the perturbation direction
    # --- extension flags (see TODO(student) blocks) ---
    coordinate: bool = False         # (A) joint multi-agent perturbation
    timing_budget: Optional[float] = None  # (B) fraction of steps the attacker may act
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────── eval-time interface (FGSM-compatible) ───────────────────
class LearnedObservationAdversary:
    """
    Wraps a trained AdversaryActor behind the SAME interface as
    FGSMAttackFramework, so it drops into `_attack_episodes` unchanged. Set the
    runner's `attack_framework` to an instance of this to evaluate a learned
    adversary with the existing damage-ceiling / random-control / flip metrics.

    Coordinated mode (cfg.coordinate=True, see AdversaryConfig): the runner's
    call site is still per-agent (`generate_adversarial_state` is called once
    per compromised host per step, in `_attack_episodes`'s per-agent loop), so
    this class cannot receive the whole group's observations in one call. It
    reconstructs them itself via `network_engine.get_state(host)` for every
    host in `group_hosts`, runs the SINGLE joint actor once per timestep, and
    hands each caller its own slice of the resulting joint delta. Every
    compromised agent in a given step therefore acts on the SAME joint
    decision — a coordinated attack — without the eval call site changing.
    `group_hosts` must be ordered so position i is the MADDPG agent_index the
    runner will pass for host i (i.e. the same order as trainable_indices).
    """

    def __init__(self, obs_dim: int, cfg: AdversaryConfig,
                 bandwidth_indices: Optional[Sequence[int]] = None,
                 group_hosts: Optional[Sequence[str]] = None):
        self.cfg = cfg
        self.epsilon = cfg.epsilon           # runner sets this per case; kept in sync
        self.attack_type = "learned"
        self.device = torch.device(cfg.device)
        self.coordinate = bool(cfg.coordinate)
        self.per_agent_obs_dim = obs_dim

        if self.coordinate:
            if not group_hosts:
                raise ValueError("cfg.coordinate=True requires group_hosts (the "
                                  "ordered list of compromised agents' hosts) so the "
                                  "adversary knows what joint observation to "
                                  "reconstruct at eval time")
            self.group_hosts = list(group_hosts)
        else:
            self.group_hosts = None
        self.group_size = len(self.group_hosts) if self.coordinate else 1
        self.actor_obs_dim = obs_dim * self.group_size
        self.actor = AdversaryActor(self.actor_obs_dim, cfg.hidden).to(self.device)
        self.actor.eval()

        local_bw = list(bandwidth_indices) if bandwidth_indices else None
        if local_bw is not None and self.group_size > 1:
            # tile the per-agent bandwidth slots across every block of the joint vector
            self.bandwidth_indices = np.concatenate([
                np.asarray(local_bw, dtype=np.int64) + k * obs_dim
                for k in range(self.group_size)
            ])
        elif local_bw is not None:
            self.bandwidth_indices = np.asarray(local_bw, dtype=np.int64)
        else:
            self.bandwidth_indices = None

        # per-timestep cache so N compromised agents in the same group cost ONE
        # joint actor pass per step, not N (keyed on the engine's own step counter)
        self._cache_step = None
        self._cache_blocks: Optional[List[np.ndarray]] = None

        # (B) timing gate: one fire/skip decision per timestep, shared by every
        # compromised agent that calls in during that step (independent mode
        # included — see the cache in generate_adversarial_state). Counts are
        # summarised every 10 (eval-)episodes instead of logged per fire —
        # episode boundaries are inferred from network_engine.time_step wrapping.
        self.timing_gate = TimingGate(cfg.timing_budget) if cfg.timing_budget else None
        self._gate_step = None
        self._gate_fire = True
        self._timing_counts: Counter = Counter()
        self._timing_episode = 0
        self._timing_last_step = None

        # stats block kept for API parity with FGSMAttackFramework
        self.attack_stats: Dict = {"total_attacks": 0, "attack_success_count": 0}

    # -- projection into the admissible perturbation set --------------------
    def _project(self, orig: np.ndarray, adv: np.ndarray) -> np.ndarray:
        """L-inf ball around orig, then domain clamp (obs features live in [0,1])."""
        adv = np.clip(adv, orig - self.epsilon, orig + self.epsilon)
        if self.bandwidth_indices is not None:
            adv[self.bandwidth_indices] = np.clip(adv[self.bandwidth_indices], 0.0, 1.0)
        else:
            adv = np.clip(adv, 0.0, 1.0)      # env observations are normalised
        return adv.astype(np.float32)

    @torch.no_grad()
    def _actor_delta(self, joint_obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(joint_obs, device=self.device).unsqueeze(0)
        return self.actor(o).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def perturb(self, state: np.ndarray) -> np.ndarray:
        if self.coordinate:
            raise RuntimeError("coordinated adversary: use generate_adversarial_state "
                                "(network_engine + agent_index) or perturb_joint, not "
                                "perturb() on a single agent's state")
        orig = np.asarray(state, dtype=np.float32)
        delta = self._actor_delta(orig) * self.epsilon
        return self._project(orig, orig + delta)

    @torch.no_grad()
    def perturb_joint(self, joint_states: Sequence[np.ndarray]) -> List[np.ndarray]:
        """(A) Coordinated perturbation. `joint_states` are the CLEAN observations
        of every agent in the group, in group order. All are perturbed from ONE
        joint delta emitted by a single actor pass, so the adversary can trade off
        pushing several agents' flows onto one shared surviving link rather than
        optimising each agent's observation in isolation."""
        blocks = [np.asarray(s, dtype=np.float32) for s in joint_states]
        joint_orig = np.concatenate(blocks)
        delta = self._actor_delta(joint_orig) * self.epsilon
        applied = self._project(joint_orig, joint_orig + delta)
        d = self.per_agent_obs_dim
        return [applied[i * d:(i + 1) * d] for i in range(self.group_size)]

    def generate_adversarial_state(self, state, agent_network=None,
                                   network_engine=None, agent_index: int = 0,
                                   bandwidth_indices=None) -> np.ndarray:
        """FGSM-compatible entry point: called once per compromised agent per
        step. In independent mode this acts on `state` alone. In coordinated
        mode `network_engine` (the runner passes `env.engine`) is required —
        it is used to reconstruct the other compromised agents' observations
        for this same step so a genuinely joint delta can be computed.

        (B) If cfg.timing_budget is set, a TimingGate decides once per timestep
        (cached across every compromised agent's call this step, coordinate or
        not) whether this is a high-leverage enough moment to spend the attack
        budget on; skipped steps return the CLEAN state untouched. Fire/skip
        counts (and why — the congestion severity band) are tallied and flushed
        as one summary line every 10 eval episodes, not logged per step."""
        if self.timing_gate is not None:
            if network_engine is None:
                raise ValueError("cfg.timing_budget requires network_engine to "
                                  "score the critical-state signal")
            step = getattr(network_engine, "time_step", None)
            if step is None or step != self._gate_step:
                if self._timing_last_step is not None and step is not None \
                        and step < self._timing_last_step:
                    self._timing_episode += 1
                    if self._timing_episode % 10 == 0:
                        _log_timing_summary("[adv-timing][eval]", self._timing_counts,
                                            self.timing_gate.budget, 10)
                self._timing_last_step = step
                fire, score, threshold = self.timing_gate.should_attack(network_engine)
                self._gate_step = step
                self._gate_fire = fire
                self._timing_counts["fired" if fire else "skipped"] += 1
                if fire:
                    self._timing_counts[self.timing_gate.severity(score)] += 1
            if not self._gate_fire:
                return np.asarray(state, dtype=np.float32)
        if not self.coordinate:
            return self.perturb(state)
        if network_engine is None:
            raise ValueError("coordinated adversary needs network_engine to "
                              "reconstruct the other compromised agents' "
                              "observations for this step")
        step = getattr(network_engine, "time_step", None)
        if step is None or step != self._cache_step:
            joint_states = [network_engine.get_state(h) for h in self.group_hosts]
            self._cache_blocks = self.perturb_joint(joint_states)
            self._cache_step = step
        return self._cache_blocks[agent_index]

    # -- persistence -------------------------------------------------------
    def save(self, path: str):
        torch.save({"actor": self.actor.state_dict(),
                    "epsilon": self.cfg.epsilon,
                    "coordinate": self.coordinate,
                    "group_size": self.group_size,
                    "per_agent_obs_dim": self.per_agent_obs_dim}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if ckpt.get("group_size", self.group_size) != self.group_size:
            raise ValueError(f"checkpoint group_size={ckpt.get('group_size')} does not "
                              f"match this adversary's group_size={self.group_size}")
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()
        return self


# ──────────────────── random control (same ball, no learning) ─────────────────
class RandomControlAdversary:
    """Random-perturbation CONTROL for the learned adversary.

    Identical threat model to LearnedObservationAdversary — same L-inf epsilon
    ball, same domain re-projection, same optional timing gate — but the
    direction is drawn uniformly at random instead of learned. This is the arm
    the learned attack has to beat.

    Why it matters: under link failures the RAW PDR drop is dominated by the
    failures themselves, not by the attack, so `clean - learned` mostly measures
    the topology falling over. Only the `random - learned` gap isolates what the
    *adversary* contributed. Report that gap, not the raw drop.

    RNG discipline: draws come from a DEDICATED Generator, never the global
    numpy/random streams that `_attack_episodes` seeds from `traffic_seed`. If
    this arm consumed entropy from those streams it would desync the injected
    traffic relative to the clean and learned runs and silently break the
    pairing (and with it the paired CIs). Mirrors the runner's own
    `_attack_rng` / `_rule_rng` discipline.
    """

    def __init__(self, obs_dim: int, cfg: AdversaryConfig,
                 bandwidth_indices: Optional[Sequence[int]] = None,
                 seed: int = 0):
        self.cfg = cfg
        self.epsilon = cfg.epsilon        # runner overwrites this per case
        self.attack_type = "random"
        self.per_agent_obs_dim = obs_dim
        self.bandwidth_indices = (np.asarray(list(bandwidth_indices), dtype=np.int64)
                                  if bandwidth_indices else None)
        self.rng = np.random.default_rng(seed)

        # Same gate as the learned arm so the comparison holds TIMING fixed and
        # varies only the perturbation DIRECTION (learned vs random). Each arm
        # gates on its own trajectory, which is correct: the trajectories
        # genuinely diverge once the attacks differ.
        self.timing_gate = TimingGate(cfg.timing_budget) if cfg.timing_budget else None
        self._gate_step = None
        self._gate_fire = True

        self.attack_stats: Dict = {"total_attacks": 0, "attack_success_count": 0}

    def _project(self, orig: np.ndarray, adv: np.ndarray) -> np.ndarray:
        adv = np.clip(adv, orig - self.epsilon, orig + self.epsilon)
        if self.bandwidth_indices is not None:
            adv[self.bandwidth_indices] = np.clip(adv[self.bandwidth_indices], 0.0, 1.0)
        else:
            adv = np.clip(adv, 0.0, 1.0)
        return adv.astype(np.float32)

    def perturb(self, state: np.ndarray) -> np.ndarray:
        orig = np.asarray(state, dtype=np.float32)
        d = self.rng.uniform(-1.0, 1.0, size=orig.shape)
        return self._project(orig, orig + d * self.epsilon)

    def generate_adversarial_state(self, state, agent_network=None,
                                   network_engine=None, agent_index: int = 0,
                                   bandwidth_indices=None) -> np.ndarray:
        """FGSM-compatible entry point, same signature as the learned arm."""
        if self.timing_gate is not None:
            if network_engine is None:
                raise ValueError("cfg.timing_budget requires network_engine to "
                                  "score the critical-state signal")
            step = getattr(network_engine, "time_step", None)
            if step is None or step != self._gate_step:
                fire, _score, _thr = self.timing_gate.should_attack(network_engine)
                self._gate_step = step
                self._gate_fire = fire
            if not self._gate_fire:
                return np.asarray(state, dtype=np.float32)
        return self.perturb(state)


# ─────────────────────────── trainer ─────────────────────────────────────────
class AdversaryTrainer:
    """
    SA-MDP DDPG trainer. Drives a FROZEN victim through the environment; at each
    step the adversary perturbs the compromised agents' observations; the reward
    is the victim's per-step packet loss (so maximising return = minimising the
    victim's delivery). Victim weights are never updated.

    Required collaborators (wire them in tools/train_adversary.py):
      victim:            object with .choose_action(list_of_states) and .agents
      env:               NetworkEnv (with .engine.get_state / .step / reset)
      trainable_indices: topology indices that carry a learning actor
      hosts:             env.engine.get_all_hosts()
    """

    def __init__(self, victim, env, trainable_indices: Sequence[int],
                 obs_dim: int, cfg: AdversaryConfig,
                 build_full_actions: Callable,
                 bandwidth_indices: Optional[Sequence[int]] = None):
        self.victim = victim
        self.env = env
        self.trainable_indices = list(trainable_indices)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.build_full_actions = build_full_actions      # runner._build_full_actions
        self.hosts = env.engine.get_all_hosts()
        self.n_total_hosts = getattr(env.engine, "n_total_hosts", len(self.hosts))
        self.n_actions = victim.n_actions

        # (A) coordinated group: every trainable index is jointly attacked, in
        # the same order the runner will later pass as agent_index at eval time
        self.group_hosts = [self.hosts[i] for i in self.trainable_indices]

        self.adv = LearnedObservationAdversary(
            obs_dim, cfg, bandwidth_indices,
            group_hosts=self.group_hosts if cfg.coordinate else None,
        )
        self.actor = self.adv.actor
        actor_obs_dim = self.adv.actor_obs_dim
        self.actor_t = AdversaryActor(actor_obs_dim, cfg.hidden).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.link_state_dim = (int(env.engine.get_central_state(self.group_hosts).shape[0])
                                if cfg.coordinate else 0)
        self.critic = AdversaryCritic(actor_obs_dim, cfg.hidden, self.link_state_dim).to(self.device)
        self.critic_t = AdversaryCritic(actor_obs_dim, cfg.hidden, self.link_state_dim).to(self.device)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.buffer = ReplayBuffer()
        self._step = 0

        # (B) timing gate: shared across coordinate/independent, one decision
        # per environment step (see _attack_states). Counts are summarised every
        # log_every episodes in train() instead of logged per fire.
        self.timing_gate = TimingGate(cfg.timing_budget) if cfg.timing_budget else None
        self._current_episode = 0
        self._timing_counts: Counter = Counter()

    # -- perturb every compromised agent's observation this step -----------
    def _attack_states(self, states: List[np.ndarray], explore: bool
                       ) -> Tuple[List[np.ndarray], List[Tuple]]:
        """Returns (perturbed_states, transitions). Each transition is
        (group_key, clean_obs, applied_delta, link_state) for the replay
        buffer: group_key is a single topo_idx in independent mode, or the
        list of every topo_idx in the coordinated group when cfg.coordinate
        is set; link_state is the GLOBAL link vector in that case, else None.
        """
        # (B) timing gate: spend the L0 budget only on high-leverage steps. A
        # skipped step returns the CLEAN states untouched and pushes NO replay
        # transition — the buffer only ever sees steps the adversary chose to
        # act on. Fire/skip + why (severity band) are tallied and flushed as one
        # summary line every log_every episodes in train(), not logged per step.
        if self.timing_gate is not None:
            fire, score, threshold = self.timing_gate.should_attack(self.env.engine)
            self._timing_counts["fired" if fire else "skipped"] += 1
            if not fire:
                return list(states), []
            self._timing_counts[self.timing_gate.severity(score)] += 1
        if self.cfg.coordinate:
            return self._attack_states_joint(states, explore)
        return self._attack_states_independent(states, explore)

    def _attack_states_independent(self, states: List[np.ndarray], explore: bool
                                   ) -> Tuple[List[np.ndarray], List[Tuple]]:
        adv_states = list(states)
        transitions = []
        for topo_idx in self.trainable_indices:
            orig = np.asarray(states[topo_idx], dtype=np.float32)
            o = torch.as_tensor(orig, device=self.device).unsqueeze(0)
            with torch.no_grad():
                d = self.actor(o).squeeze(0).cpu().numpy()
            if explore:
                d = d + np.random.normal(0, self.cfg.explore_noise, size=d.shape)
                d = np.clip(d, -1.0, 1.0)
            applied = self.adv._project(orig, orig + d * self.cfg.epsilon)
            adv_states[topo_idx] = applied
            transitions.append((topo_idx, orig, (applied - orig) / self.cfg.epsilon, None))
        return adv_states, transitions

    def _attack_states_joint(self, states: List[np.ndarray], explore: bool
                             ) -> Tuple[List[np.ndarray], List[Tuple]]:
        """(A) Coordinated multi-agent perturbation: concatenate every
        compromised agent's observation into one joint vector, run it through
        a SINGLE actor so the emitted delta is one coherent joint decision
        (not N independent ones), and score it with a critic that sees the
        GLOBAL link state (env.engine.get_central_state) rather than any one
        agent's local neighbourhood. This is the mechanism a per-agent
        gradient cannot express: it lets the adversary push several flows
        onto ONE shared surviving link instead of each agent independently
        reacting to its own view of the network.
        """
        adv_states = list(states)
        blocks = [np.asarray(states[i], dtype=np.float32) for i in self.trainable_indices]
        joint_orig = np.concatenate(blocks)
        o = torch.as_tensor(joint_orig, device=self.device).unsqueeze(0)
        with torch.no_grad():
            d = self.actor(o).squeeze(0).cpu().numpy()
        if explore:
            d = d + np.random.normal(0, self.cfg.explore_noise, size=d.shape)
            d = np.clip(d, -1.0, 1.0)
        applied_joint = self.adv._project(joint_orig, joint_orig + d * self.cfg.epsilon)
        applied_delta_joint = (applied_joint - joint_orig) / self.cfg.epsilon

        obs_dim = blocks[0].shape[0]
        for pos, topo_idx in enumerate(self.trainable_indices):
            adv_states[topo_idx] = applied_joint[pos * obs_dim:(pos + 1) * obs_dim]

        link_state = self.env.engine.get_central_state(self.group_hosts).astype(np.float32)
        transitions = [(list(self.trainable_indices), joint_orig, applied_delta_joint, link_state)]
        return adv_states, transitions

    def _victim_actions(self, states: List[np.ndarray]):
        t_states = [states[i] for i in self.trainable_indices]
        t_actions = self.victim.choose_action(t_states)
        return self.build_full_actions(t_actions, self.n_total_hosts,
                                       self.trainable_indices, self.n_actions)

    def _update(self):
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return
        for _ in range(self.cfg.updates_per_step):
            obs, delta, r, nobs, done, link_state, next_link_state = \
                self.buffer.sample(self.cfg.batch_size, self.device)
            # critic: TD target uses the adversary's own next-step action
            with torch.no_grad():
                nd = self.actor_t(nobs)
                y = r + self.cfg.gamma * (1 - done) * self.critic_t(nobs, nd, next_link_state)
            q = self.critic(obs, delta, link_state)
            loss_c = F.mse_loss(q, y)
            self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()
            # actor: ascend Q (maximise victim loss)
            loss_a = -self.critic(obs, self.actor(obs), link_state).mean()
            self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()
            self._soft_update(self.actor_t, self.actor)
            self._soft_update(self.critic_t, self.critic)

    def _soft_update(self, target, online):
        with torch.no_grad():
            for tp, p in zip(target.parameters(), online.parameters()):
                tp.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p)

    def train(self, n_episodes: int, t_per_ep: int = 256,
              offered_load_factor: float = 2.0, n_link_failures: int = 0,
              log_every: int = 10) -> Dict:
        """Main SA-MDP loop. Returns a small history dict for plotting."""
        history = {"episode": [], "victim_pdr": [], "attack_loss_reward": []}
        for ep in range(n_episodes):
            self._current_episode = ep
            self.env.engine.reset_with_load(offered_load_factor=offered_load_factor)
            if n_link_failures:
                # reuse the runner's failure injector at the call site if you want
                # failure-regime adversaries; left off by default here.
                pass
            states = [self.env.engine.get_state(h) for h in self.hosts]
            ep_reward = 0.0
            explore = self._step < self.cfg.warmup_steps or True  # keep light noise
            for _ in range(t_per_ep):
                adv_states, transitions = self._attack_states(states, explore)
                actions = self._victim_actions(adv_states)
                next_states, _rewards, info = self.env.step(actions)
                # SA-MDP reward: victim per-step packet loss fraction (attacker gain)
                loss_frac = float(info.get("packet_loss_rate", 0.0)) / 100.0
                for (group_key, clean_obs, applied_delta, link_state) in transitions:
                    if isinstance(group_key, list):
                        next_obs = np.concatenate(
                            [next_states[i] for i in group_key]).astype(np.float32)
                        next_link_state = self.env.engine.get_central_state(
                            self.group_hosts).astype(np.float32)
                    else:
                        next_obs = np.asarray(next_states[group_key], np.float32)
                        next_link_state = None
                    self.buffer.push(clean_obs, applied_delta, loss_frac, next_obs, 0.0,
                                     link_state, next_link_state)
                ep_reward += loss_frac
                states = next_states
                self._step += 1
                self._update()
            pdr = float(self.env.get_stats().get("end_to_end_pdr", 0.0))
            history["episode"].append(ep)
            history["victim_pdr"].append(pdr)
            history["attack_loss_reward"].append(ep_reward / t_per_ep)
            if ep % log_every == 0:
                print(f"[adv] ep {ep:4d}  victim PDR {pdr:6.2f}%  "
                      f"mean step-loss reward {ep_reward / t_per_ep:.4f}")
                if self.timing_gate is not None:
                    _log_timing_summary("[adv-timing]", self._timing_counts,
                                        self.timing_gate.budget, log_every)
        return history

    def save(self, path: str):
        self.adv.save(path)