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

Two extension points are stubbed with TODO(student) — they are the parts that make
the attack *specific to this routing scenario* and are the intended research
contributions:
  (A) COORDINATED multi-agent perturbation  (steer flows onto a SHARED bottleneck)
  (B) STRATEGICALLY-TIMED / critical-state attacks  (spend an L0 budget only at the
      high-leverage moments — congestion onset, immediately post-failure)

Eval: `LearnedObservationAdversary` exposes `generate_adversarial_state(...)` with
the SAME signature as FGSMAttackFramework, so a trained adversary drops straight
into `standalone_experiment_runner._attack_episodes` via attack_type='learned'
and is scored by the existing damage-ceiling / random-control / action-flip metrics.

Run the trainer with  tools/train_adversary.py  (wires up env + frozen victim).
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Q(obs, delta) for the DDPG update of the adversary."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, delta], dim=-1))


# ─────────────────────────── replay ──────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, obs, delta, reward, next_obs, done):
        self.buf.append((obs, delta, reward, next_obs, done))

    def sample(self, batch: int, device):
        idx = random.sample(range(len(self.buf)), batch)
        obs, delta, r, nobs, done = zip(*(self.buf[i] for i in idx))
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
        return (t(obs), t(delta), t(r).unsqueeze(-1), t(nobs), t(done).unsqueeze(-1))

    def __len__(self):
        return len(self.buf)


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
    timing_budget: Optional[float] = 0,2  # (B) fraction of steps the attacker may act
    timing_score_metric: str = "utilization"  # (B) "utilization" | "q_saliency"
    timing_window: int = 200         # (B) rolling window for adaptive threshold
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────── eval-time interface (FGSM-compatible) ───────────────────
class LearnedObservationAdversary:
    """
    Wraps a trained AdversaryActor behind the SAME interface as
    FGSMAttackFramework, so it drops into `_attack_episodes` unchanged. Set the
    runner's `attack_framework` to an instance of this to evaluate a learned
    adversary with the existing damage-ceiling / random-control / flip metrics.
    """

    def __init__(self, obs_dim: int, cfg: AdversaryConfig,
                 bandwidth_indices: Optional[Sequence[int]] = None):
        self.cfg = cfg
        self.epsilon = cfg.epsilon           # runner sets this per case; kept in sync
        self.attack_type = "learned"
        self.device = torch.device(cfg.device)
        self.actor = AdversaryActor(obs_dim, cfg.hidden).to(self.device)
        self.actor.eval()
        self.bandwidth_indices = list(bandwidth_indices) if bandwidth_indices else None
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
    def perturb(self, state: np.ndarray) -> np.ndarray:
        orig = np.asarray(state, dtype=np.float32)
        o = torch.as_tensor(orig, device=self.device).unsqueeze(0)
        delta = self.actor(o).squeeze(0).cpu().numpy() * self.epsilon
        return self._project(orig, orig + delta)

    def generate_adversarial_state(self, state, agent_network=None,
                                   network_engine=None, agent_index: int = 0,
                                   bandwidth_indices=None) -> np.ndarray:
        """FGSM-compatible entry point. agent_network/engine are unused by the
        grey-box adversary (it acts on the observation alone) but kept in the
        signature so the runner call site does not change."""
        return self.perturb(state)

    # -- persistence -------------------------------------------------------
    def save(self, path: str):
        torch.save({"actor": self.actor.state_dict(),
                    "epsilon": self.cfg.epsilon}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()
        return self


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

        self.adv = LearnedObservationAdversary(obs_dim, cfg, bandwidth_indices)
        self.actor = self.adv.actor
        self.actor_t = AdversaryActor(obs_dim, cfg.hidden).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic = AdversaryCritic(obs_dim, cfg.hidden).to(self.device)
        self.critic_t = AdversaryCritic(obs_dim, cfg.hidden).to(self.device)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.buffer = ReplayBuffer()
        self._step = 0

        # (B) timing-budget bookkeeping
        self._score_hist: deque = deque(maxlen=cfg.timing_window)
        self._ep_budget_steps: int = 0
        self._ep_used: int = 0
            # -- perturb every compromised agent's observation this step -----------
    def _attack_states(self, states: List[np.ndarray], explore: bool
                       ) -> Tuple[List[np.ndarray], List[Tuple[int, np.ndarray, np.ndarray]]]:
        """Returns (perturbed_states, transitions) where each transition is
        (topo_idx, clean_obs, applied_delta) for the replay buffer."""
        adv_states = list(states)
        transitions = []

        # (B) strategically-timed / critical-state gate: decide ONCE per step
        # whether this is a high-leverage moment worth spending L0 budget on.
        # Scoring is cheap (max utilisation) or uses the critic (Q-saliency).
        # If cfg.timing_budget is None the gate is disabled and every step is
        # attacked (original behaviour).
        score = self._critical_score(states)
        if not self._should_attack(score):
            return adv_states, transitions  # clean pass-through, no transitions logged
        self._ep_used += 1

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
            transitions.append((topo_idx, orig, (applied - orig) / self.cfg.epsilon))
        # TODO(student A — coordinate): replace the independent per-agent loop above
        # with a JOINT perturbation. Concatenate the compromised agents' observations,
        # let a single actor emit a joint delta, and share a critic that sees the
        # global link state so the adversary can push multiple flows onto ONE shared
        # surviving link (the mechanism that actually reaches the damage ceiling).
        return adv_states, transitions

    # -- (B) critical-state scoring ----------------------------------------
    def _critical_score(self, states: List[np.ndarray]) -> float:
        """Higher = more worth attacking right now. Only used when
        cfg.timing_budget is set."""
        if self.cfg.timing_budget is None:
            return 0.0
        scores = []
        for topo_idx in self.trainable_indices:
            obs = np.asarray(states[topo_idx], dtype=np.float32)
            if self.cfg.timing_score_metric == "q_saliency":
                o = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    zero = torch.zeros_like(o)
                    q_clean = self.critic(o, zero)
                    q_pert = self.critic(o, self.actor(o))
                scores.append(float((q_pert - q_clean).abs().item()))
            else:  # "utilization" -- cheap proxy, no forward pass needed
                if self.adv.bandwidth_indices is not None:
                    scores.append(float(obs[self.adv.bandwidth_indices].max()))
                else:
                    scores.append(float(obs.max()))
        return max(scores) if scores else 0.0

    def _should_attack(self, score: float) -> bool:
        """(B) Adaptive-threshold L0 gate. Attacks the top (1 - timing_budget)
        fraction of observed critical-state scores, tracked against a rolling
        window, and stops early once the per-episode budget is spent."""
        if self.cfg.timing_budget is None:
            return True  # gate disabled -> attack every step (original behaviour)

        if self._ep_used >= self._ep_budget_steps:
            return False  # budget for this episode is exhausted

        self._score_hist.append(score)
        if len(self._score_hist) < max(10, self.cfg.timing_window // 4):
            threshold = 0.0  # not enough history yet -> attack to seed the buffer
        else:
            q = max(0.0, min(1.0, 1.0 - self.cfg.timing_budget))
            threshold = float(np.quantile(self._score_hist, q))

        return score >= threshold

    def _victim_actions(self, states: List[np.ndarray]):
        t_states = [states[i] for i in self.trainable_indices]
        t_actions = self.victim.choose_action(t_states)
        return self.build_full_actions(t_actions, self.n_total_hosts,
                                       self.trainable_indices, self.n_actions)

    def _update(self):
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return
        for _ in range(self.cfg.updates_per_step):
            obs, delta, r, nobs, done = self.buffer.sample(self.cfg.batch_size, self.device)
            # critic: TD target uses the adversary's own next-step action
            with torch.no_grad():
                nd = self.actor_t(nobs)
                y = r + self.cfg.gamma * (1 - done) * self.critic_t(nobs, nd)
            q = self.critic(obs, delta)
            loss_c = F.mse_loss(q, y)
            self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()
            # actor: ascend Q (maximise victim loss)
            loss_a = -self.critic(obs, self.actor(obs)).mean()
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
            self.env.engine.reset_with_load(offered_load_factor=offered_load_factor)
            # (B) reset the per-episode L0 budget so timing gates apply fresh
            # each episode (budget is a FRACTION of t_per_ep, not cumulative).
            if self.cfg.timing_budget is not None:
                self._ep_budget_steps = max(1, int(np.ceil(self.cfg.timing_budget * t_per_ep)))
                self._ep_used = 0
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
                for (topo_idx, clean_obs, applied_delta) in transitions:
                    self.buffer.push(clean_obs, applied_delta, loss_frac,
                                     np.asarray(next_states[topo_idx], np.float32), 0.0)
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
        return history

    def save(self, path: str):
        self.adv.save(path)