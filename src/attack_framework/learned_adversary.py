"""
Learned worst-case observation adversary (SA-MDP / SA-DDPG style) for the MADDPG
routing victim. This is the *scaffold* for the follow-up MSc project: it turns the
myopic single-step FGSM attack into a trained adversary whose objective is the
victim's packet loss over whole trajectories, so we can test whether the robustness
measured under FGSM is fundamental or just an artefact of a weak (myopic, per-agent)
attacker.

Design (mirrors the paper's threat model exactly, so results stay comparable):
* Threat model: observation-space, L-inf bounded by epsilon, bandwidth/utilisation
  features re-projected to [0,1]. Identical ball to FGSMAttackFramework.
* Adversary: a deterministic policy pi_adv(obs) -> delta in the epsilon-ball,
  trained by DDPG with the *victim's per-step packet loss* as reward (SA-MDP: the
  optimal observation adversary is itself an RL agent, Zhang et al. NeurIPS 2020).
* Victim: a FROZEN trained MADDPG variant. The adversary never sees the victim's
  weights except through forward passes at attack time (grey-box); set
  `white_box=True` to also let the DDPG critic condition on victim internals.

Two extension points are stubbed with TODO(student) -- they are the parts that make
the attack *specific to this routing scenario* and are the intended research
contributions:
  (A) COORDINATED multi-agent perturbation (steer flows onto a SHARED bottleneck)
  (B) STRATEGICALLY-TIMED / critical-state attacks (spend an L0 budget only at the
      high-leverage moments -- congestion onset, immediately post-failure)

  (B) IMPORTANT FIX (this revision): the timing-budget gate used to live ONLY on
  AdversaryTrainer (self._should_attack / self._attack_states), which is exercised
  during train(). LearnedObservationAdversary.generate_adversarial_state() -- the
  FGSM-compatible entry point actually called at EVALUATION time by
  standalone_experiment_runner -- bypassed the gate entirely and perturbed on every
  call, so a trained adversary would attack 100% of steps at eval regardless of
  `timing_budget`. The gating logic (quantile score, event timer, per-episode
  budget bookkeeping) now lives on LearnedObservationAdversary itself, and BOTH
  AdversaryTrainer (joint, multi-agent view) and generate_adversarial_state
  (per-agent view, since the FGSM-compatible signature only gets one agent's obs
  at a time) route through it -- so training and evaluation always agree on
  whether a step gets attacked.

  Two selectable timing strategies:
    - "quantile": adaptive-threshold gate (attack the top-k% of steps by a
      rolling critical-state score).
    - "event":    fires only on a congestion-onset rising edge or during a fixed
      post-failure recovery window (CongestionFailureTimer below).
    - "both":     event gate OR quantile gate.

  Call `adversary.new_episode(t_per_ep=...)` at the start of every eval episode
  so the per-episode budget resets correctly (otherwise the budget keeps
  accumulating against the last-seen t_per_ep across the whole eval run).

  Every step's decision is logged (trigger label + underlying signals) via
  CongestionFailureTimer.log and the mode-agnostic attack_log, both plain
  lists/dicts (no pandas DataFrames), so they are JSON-serialisable directly.

Eval: `LearnedObservationAdversary` exposes `generate_adversarial_state(...)` with
the SAME signature as FGSMAttackFramework, so a trained adversary drops straight
into `standalone_experiment_runner._attack_episodes` via attack_type='learned'
and is scored by the existing damage-ceiling / random-control / action-flip metrics.

Run the trainer with tools/train_adversary.py (wires up env + frozen victim).
"""
from __future__ import annotations

import csv
import json
import os
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
    epsilon: float = 0.30          # L-inf budget (match the FGSM sweep)
    hidden: int = 256
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.95
    tau: float = 5e-3              # target soft-update
    batch_size: int = 256
    warmup_steps: int = 2_000
    updates_per_step: int = 1
    explore_noise: float = 0.10    # exploration noise on the perturbation direction

    # --- extension flags (see TODO(student) blocks) ---
    coordinate: bool = False               # (A) joint multi-agent perturbation
    timing_budget: Optional[float] = 0.2   # (B) fraction of steps the attacker may act
    timing_score_metric: str = "utilization"  # (B) "utilization" | "q_saliency"
    timing_window: int = 200               # (B) rolling window for adaptive threshold

    # --- (B) event-driven timing: congestion onset / post-failure recovery ---
    timing_mode: str = "both"          # "quantile" | "event" | "both"
    onset_rise_threshold: float = 0.90  # utilisation level that defines "congested"
    onset_hysteresis: float = 0.10      # must drop this far below before re-arming
    failure_loss_spike: float = 0.15    # jump in loss_frac vs rolling mean -> failure
    failure_baseline_window: int = 50   # rolling window for the loss baseline
    post_failure_steps: int = 15        # steps to keep attacking after a failure

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────── (B) event-driven timing gate ────────────────────────
class CongestionFailureTimer:
    """
    Stateful, event-driven trigger for the strategically-timed attacker.

    Fires on:
      * congestion onset -- a RISING EDGE of max utilisation through
        `onset_rise_threshold` (with hysteresis so a sustained plateau only
        fires once per onset, not every step).
      * post-failure recovery -- a fixed-length window of `post_failure_steps`
        steps immediately after a detected failure (a sudden jump in per-step
        packet loss above a rolling baseline, `failure_loss_spike`). Swap this
        proxy for an explicit `info["link_failure"]` flag if your env exposes
        one -- it will be more reliable than inferring failures from loss.

    Every step's decision is appended to `self.log` (a plain list of dicts --
    trigger label + the underlying signals, all JSON-serialisable) so
    training/eval runs can be analysed post-hoc, e.g. correlating trigger type
    with victim PDR or plotting utilisation/loss with markers at each trigger
    event.
    """

    def __init__(self, cfg: AdversaryConfig):
        self.cfg = cfg
        self.log: List[Dict] = []
        self.reset()

    def reset(self):
        """Call at the start of every episode: clears state AND the log."""
        self._armed = True
        self._post_failure_left = 0
        self._loss_hist: deque = deque(maxlen=self.cfg.failure_baseline_window)
        self.log = []

    def _max_utilization(self, states: List[np.ndarray], trainable_indices,
                          bandwidth_indices) -> float:
        vals = []
        for i in trainable_indices:
            obs = np.asarray(states[i], dtype=np.float32)
            vals.append(float(obs[bandwidth_indices].max()) if bandwidth_indices
                        else float(obs.max()))
        return max(vals) if vals else 0.0

    def update_and_check(self, step: int, states: List[np.ndarray],
                          trainable_indices: Sequence[int],
                          bandwidth_indices: Optional[Sequence[int]],
                          loss_frac: float,
                          ground_truth_failure: bool = False) -> bool:
        """
        `ground_truth_failure`: True if a real failure-injector fired THIS
        step (wired in via AdversaryTrainer.train(failure_injector=...)). When
        available, it OR's into the failure trigger directly (no need to wait
        for a loss-rate spike), and is also logged separately so you can
        compare detection latency / false positives of the loss-spike proxy
        against ground truth.
        """
        cfg = self.cfg
        util = self._max_utilization(states, trainable_indices, bandwidth_indices)

        onset_fired = False
        if util >= cfg.onset_rise_threshold and self._armed:
            onset_fired = True
            self._armed = False
        elif util <= cfg.onset_rise_threshold - cfg.onset_hysteresis:
            self._armed = True

        proxy_fired = False
        have_baseline = len(self._loss_hist) >= max(5, cfg.failure_baseline_window // 4)
        baseline = float(np.mean(self._loss_hist)) if have_baseline else None
        if baseline is not None and (loss_frac - baseline) >= cfg.failure_loss_spike:
            proxy_fired = True
        self._loss_hist.append(loss_frac)

        failure_fired = proxy_fired or ground_truth_failure
        if failure_fired:
            self._post_failure_left = cfg.post_failure_steps

        in_recovery = self._post_failure_left > 0
        if in_recovery:
            self._post_failure_left -= 1

        if onset_fired:
            trigger = "congestion_onset"
        elif failure_fired:
            trigger = "failure_detected"
        elif in_recovery:
            trigger = "post_failure_recovery"
        else:
            trigger = "none"

        self.log.append({
            "step": step,
            "utilization": util,
            "loss_frac": loss_frac,
            "baseline": baseline,
            "trigger": trigger,
            "attacked": trigger != "none",
            "proxy_failure_fired": proxy_fired,
            "ground_truth_failure": ground_truth_failure,
        })
        return trigger != "none"

    def trigger_counts(self) -> Dict[str, int]:
        """Plain dict of {trigger_label: count} for this episode's log."""
        counts: Dict[str, int] = {}
        for row in self.log:
            counts[row["trigger"]] = counts.get(row["trigger"], 0) + 1
        return counts


# ──────────────────── eval-time interface (FGSM-compatible) ───────────────────
class LearnedObservationAdversary:
    """
    Wraps a trained AdversaryActor behind the SAME interface as
    FGSMAttackFramework, so it drops into `_attack_episodes` unchanged. Set the
    runner's `attack_framework` to an instance of this to evaluate a learned
    adversary with the existing damage-ceiling / random-control / flip metrics.

    THE TIMING-BUDGET GATE NOW LIVES HERE (previously only on AdversaryTrainer,
    which meant eval-time calls to generate_adversarial_state() bypassed it and
    attacked 100% of steps). Both AdversaryTrainer and generate_adversarial_state
    route through the SAME should_attack()/quantile/event machinery below, so
    training and evaluation are always consistent.

    NOTE on the event/onset trigger at eval time: the FGSM-compatible signature
    `generate_adversarial_state(state, ...)` only receives ONE agent's
    observation per call (no joint view across all compromised agents like
    AdversaryTrainer has during training). The congestion-onset utilisation
    check and the quantile score therefore use THIS agent's own observation as
    the proxy, rather than the max across all compromised agents. This is a
    documented approximation required by the fixed eval interface; if you want
    the exact joint-max semantics at eval, call `should_attack_joint(states,
    trainable_indices, agent_index)` directly with the full state list instead
    of going through generate_adversarial_state().

    Call `new_episode(t_per_ep=...)` at the start of every eval episode so the
    per-episode budget resets (otherwise it accumulates against the last-seen
    t_per_ep across the whole eval run).
    """

    def __init__(self, obs_dim: int, cfg: AdversaryConfig,
                 bandwidth_indices: Optional[Sequence[int]] = None):
        self.cfg = cfg
        self.epsilon = cfg.epsilon  # runner sets this per case; kept in sync
        self.attack_type = "learned"
        self.device = torch.device(cfg.device)
        self.actor = AdversaryActor(obs_dim, cfg.hidden).to(self.device)
        self.actor.eval()
        self.bandwidth_indices = list(bandwidth_indices) if bandwidth_indices else None
        # stats block kept for API parity with FGSMAttackFramework
        self.attack_stats: Dict = {"total_attacks": 0, "attack_success_count": 0}

        # optional: trainer wires this in so q_saliency scoring can use the
        # DDPG critic; if None (pure eval-only load), falls back to "utilization"
        self.critic: Optional[AdversaryCritic] = None

        # --- (B) timing-budget gate state (single source of truth) ---
        self.event_timer = CongestionFailureTimer(cfg) if cfg.timing_mode in ("event", "both") else None
        self._score_hist: deque = deque(maxlen=cfg.timing_window)
        self._ep_budget_steps: int = 0
        self._ep_used: int = 0
        self._cur_step_in_ep: int = 0
        self._cur_episode: int = -1
        self._t_per_ep: Optional[int] = None
        self._last_loss_frac: float = 0.0
        self._last_ground_truth_failure: bool = False
        self.attack_log: List[Dict] = []

    # -- episode bookkeeping -------------------------------------------------
    def new_episode(self, episode_idx: Optional[int] = None, t_per_ep: Optional[int] = None):
        """Reset the per-episode L0 budget and event-timer state. MUST be
        called at the start of every episode (training AND evaluation) or the
        budget will not reset and the gate will drift out of sync with actual
        episode boundaries."""
        self._cur_episode = episode_idx if episode_idx is not None else self._cur_episode + 1
        self._cur_step_in_ep = 0
        self._ep_used = 0
        if t_per_ep is not None:
            self._t_per_ep = t_per_ep
        if self.cfg.timing_budget is not None and self._t_per_ep:
            self._ep_budget_steps = max(1, int(np.ceil(self.cfg.timing_budget * self._t_per_ep)))
        else:
            self._ep_budget_steps = 0
        if self.event_timer is not None:
            self.event_timer.reset()
        self._last_loss_frac = 0.0
        self._last_ground_truth_failure = False

    def update_step_context(self, loss_frac: float = 0.0, ground_truth_failure: bool = False):
        """Feed the latest per-step packet-loss fraction (and optional
        ground-truth failure flag) into the gate. Call this once per step,
        BEFORE the next call to generate_adversarial_state / should_attack, so
        the event timer's failure detector has fresh data."""
        self._last_loss_frac = loss_frac
        self._last_ground_truth_failure = ground_truth_failure

    # -- (B) critical-state scoring (single-observation approximation) ------
    def _critical_score_single(self, obs: np.ndarray) -> float:
        if self.cfg.timing_budget is None:
            return 0.0
        if self.cfg.timing_score_metric == "q_saliency" and self.critic is not None:
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                zero = torch.zeros_like(o)
                q_clean = self.critic(o, zero)
                q_pert = self.critic(o, self.actor(o))
            return float((q_pert - q_clean).abs().item())
        if self.bandwidth_indices is not None:
            return float(obs[self.bandwidth_indices].max())
        return float(obs.max())

    def _quantile_gate(self, score: float) -> bool:
        self._score_hist.append(score)
        if len(self._score_hist) < max(10, self.cfg.timing_window // 4):
            threshold = 0.0  # not enough history yet -> attack to seed the buffer
        else:
            q = max(0.0, min(1.0, 1.0 - self.cfg.timing_budget))
            threshold = float(np.quantile(self._score_hist, q))
        return score >= threshold

    def should_attack(self, obs: np.ndarray) -> Tuple[bool, float]:
        """Single-observation gate used by generate_adversarial_state (eval
        path). Returns (attacked, score). Advances the internal step counter."""
        if self.cfg.timing_budget is None:
            self._log_decision(True, 0.0)
            return True, 0.0

        if self._ep_budget_steps and self._ep_used >= self._ep_budget_steps:
            self._log_decision(False, 0.0)
            self._cur_step_in_ep += 1
            return False, 0.0

        score = self._critical_score_single(obs)
        mode = self.cfg.timing_mode
        event_hit = False
        if mode in ("event", "both") and self.event_timer is not None:
            event_hit = self.event_timer.update_and_check(
                self._cur_step_in_ep, [obs], [0], self.bandwidth_indices,
                self._last_loss_frac, ground_truth_failure=self._last_ground_truth_failure)
            if mode == "event":
                attacked = event_hit
                if attacked:
                    self._ep_used += 1
                self._log_decision(attacked, score)
                self._cur_step_in_ep += 1
                return attacked, score

        quantile_hit = self._quantile_gate(score)
        attacked = (quantile_hit or event_hit) if mode == "both" else quantile_hit
        if attacked:
            self._ep_used += 1
        self._log_decision(attacked, score)
        self._cur_step_in_ep += 1
        return attacked, score

    def should_attack_joint(self, states: List[np.ndarray], trainable_indices: Sequence[int],
                             agent_index: int) -> Tuple[bool, float]:
        """Exact joint-max-utilisation gate, matching AdversaryTrainer's
        training-time semantics precisely. Use this instead of should_attack()
        if your eval harness can supply the full states list (all compromised
        agents), not just one agent's observation."""
        if self.cfg.timing_budget is None:
            self._log_decision(True, 0.0)
            return True, 0.0

        if self._ep_budget_steps and self._ep_used >= self._ep_budget_steps:
            self._log_decision(False, 0.0)
            self._cur_step_in_ep += 1
            return False, 0.0

        obs = np.asarray(states[agent_index], dtype=np.float32)
        score = self._critical_score_single(obs)
        mode = self.cfg.timing_mode
        event_hit = False
        if mode in ("event", "both") and self.event_timer is not None:
            event_hit = self.event_timer.update_and_check(
                self._cur_step_in_ep, states, trainable_indices, self.bandwidth_indices,
                self._last_loss_frac, ground_truth_failure=self._last_ground_truth_failure)
            if mode == "event":
                attacked = event_hit
                if attacked:
                    self._ep_used += 1
                self._log_decision(attacked, score)
                self._cur_step_in_ep += 1
                return attacked, score

        quantile_hit = self._quantile_gate(score)
        attacked = (quantile_hit or event_hit) if mode == "both" else quantile_hit
        if attacked:
            self._ep_used += 1
        self._log_decision(attacked, score)
        self._cur_step_in_ep += 1
        return attacked, score

    def _log_decision(self, attacked: bool, score: float):
        self.attack_log.append({
            "episode": self._cur_episode,
            "step": self._cur_step_in_ep,
            "attacked": attacked,
            "score": score,
            "timing_mode": self.cfg.timing_mode if self.cfg.timing_budget is not None else "disabled",
            "ep_used": self._ep_used,
            "ep_budget_steps": self._ep_budget_steps,
        })

    # -- projection into the admissible perturbation set --------------------
    def _project(self, orig: np.ndarray, adv: np.ndarray) -> np.ndarray:
        """L-inf ball around orig, then domain clamp (obs features live in [0,1])."""
        adv = np.clip(adv, orig - self.epsilon, orig + self.epsilon)
        if self.bandwidth_indices is not None:
            adv[self.bandwidth_indices] = np.clip(adv[self.bandwidth_indices], 0.0, 1.0)
        else:
            adv = np.clip(adv, 0.0, 1.0)  # env observations are normalised
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
        signature so the runner call site does not change.

        THIS NOW RESPECTS cfg.timing_budget: if the gate says "don't attack
        this step" (budget exhausted, or neither the quantile nor event
        trigger fired), the CLEAN state is returned unchanged. Previously this
        method perturbed unconditionally, which is why timing_budget appeared
        to have no effect at evaluation time."""
        orig = np.asarray(state, dtype=np.float32)
        attacked, _score = self.should_attack(orig)
        if not attacked:
            return orig
        return self.perturb(orig)

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
      victim: object with .choose_action(list_of_states) and .agents
      env: NetworkEnv (with .engine.get_state / .step / reset)
      trainable_indices: topology indices that carry a learning actor
      hosts: env.engine.get_all_hosts()
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
        self.build_full_actions = build_full_actions  # runner._build_full_actions
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
        self.adv.critic = self.critic  # let q_saliency scoring see the live critic
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.buffer = ReplayBuffer()
        self._step = 0

        # (B) the gate itself now lives on self.adv (single source of truth,
        # shared with generate_adversarial_state at eval time). Training uses
        # self.adv.should_attack_joint(...) so it keeps the exact joint-max
        # utilisation semantics across all compromised agents.
        self._pending_states: Optional[List[np.ndarray]] = None
        self.trigger_logs: List[List[Dict]] = []  # one list-of-dicts per completed episode

    # -- perturb every compromised agent's observation this step -----------
    def _attack_states(self, states: List[np.ndarray], explore: bool
                        ) -> Tuple[List[np.ndarray], List[Tuple[int, np.ndarray, np.ndarray]]]:
        """Returns (perturbed_states, transitions) where each transition is
        (topo_idx, clean_obs, applied_delta) for the replay buffer."""
        adv_states = list(states)
        transitions = []

        # (B) strategically-timed / critical-state gate: decide ONCE per step
        # whether this is a high-leverage moment worth spending L0 budget on.
        # Delegates to self.adv (the same gate used at eval time), using the
        # JOINT view across all compromised agents for accurate onset detection.
        attacked, _score = self.adv.should_attack_joint(
            states, self.trainable_indices, self.trainable_indices[0] if self.trainable_indices else 0)

        if not attacked:
            return adv_states, transitions  # clean pass-through, no transitions logged

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
        # TODO(student A -- coordinate): replace the independent per-agent loop above
        # with a JOINT perturbation. Concatenate the compromised agents' observations,
        # let a single actor emit a joint delta, and share a critic that sees the
        # global link state so the adversary can push multiple flows onto ONE shared
        # surviving link (the mechanism that actually reaches the damage ceiling).
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

    @staticmethod
    def _write_rows_csv(path: str, rows: List[Dict]):
        """Plain csv.DictWriter -- no pandas required."""
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def train(self, n_episodes: int, t_per_ep: int = 256,
              offered_load_factor: float = 2.0, n_link_failures: int = 0,
              failure_injector: Optional[Callable[["object", int], Optional[Sequence]]] = None,
              log_every: int = 10, log_dir: Optional[str] = None) -> Dict:
        """Main SA-MDP loop. Returns a small history dict for plotting -- all
        JSON-serialisable (plain lists/dicts, no DataFrames).

        `failure_injector(env, step) -> failed_links | None` is an optional
        callable wired in at the call site (e.g. the runner's existing failure
        injector). If it returns a non-empty/non-None result on a given step,
        that step is tagged as a GROUND-TRUTH failure event and compared
        against the loss-spike proxy in the trigger log (see the
        `ground_truth_failure` column). If `n_link_failures > 0` but no
        `failure_injector` is supplied, ground truth is left unavailable and
        only the proxy signal drives the post-failure gate.

        If `log_dir` is given:
          - "event"/"both" modes also write f"{log_dir}/trigger_log_ep{ep:04d}.csv"
            (from CongestionFailureTimer) and keep them in history["trigger_logs"].
          - ALL modes write f"{log_dir}/attack_log_ep{ep:04d}.csv" -- the
            mode-agnostic per-step attack decision log, so you can always see
            which episode(s) contained attacks.

        Console output (always printed, no extra calls needed):
          - every `log_every` episodes: victim PDR, mean step-loss reward,
            attacked-steps count, and (in "event"/"both" modes) a trigger
            breakdown for that episode.
          - after training finishes: a full attacks-per-episode table, total
            attacked-steps percentage, all-episode trigger counts, and a
            TP/FN/FP comparison of the loss-spike proxy vs ground-truth
            failures (if a failure_injector was used)."""
        history = {"episode": [], "victim_pdr": [], "attack_loss_reward": []}
        if self.cfg.timing_mode in ("event", "both"):
            history["trigger_logs"] = []

        for ep in range(n_episodes):
            self.env.engine.reset_with_load(offered_load_factor=offered_load_factor)
            # (B) reset the per-episode L0 budget + event timer on self.adv --
            # the single shared gate used by both training and eval.
            self.adv.new_episode(episode_idx=ep, t_per_ep=t_per_ep)
            states = [self.env.engine.get_state(h) for h in self.hosts]
            ep_reward = 0.0
            explore = self._step < self.cfg.warmup_steps or True  # keep light noise
            for t in range(t_per_ep):
                # ground-truth failure injection (if a real injector is wired in)
                gt_failure = False
                if n_link_failures and failure_injector is not None:
                    injected = failure_injector(self.env, t)
                    gt_failure = bool(injected)
                self.adv.update_step_context(loss_frac=self.adv._last_loss_frac,
                                              ground_truth_failure=gt_failure)

                adv_states, transitions = self._attack_states(states, explore)
                actions = self._victim_actions(adv_states)
                next_states, _rewards, info = self.env.step(actions)
                # SA-MDP reward: victim per-step packet loss fraction (attacker gain)
                loss_frac = float(info.get("packet_loss_rate", 0.0)) / 100.0
                self.adv._last_loss_frac = loss_frac  # feeds the event timer next step
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

            ep_trigger_log: List[Dict] = []
            if self.cfg.timing_mode in ("event", "both") and self.adv.event_timer is not None:
                ep_trigger_log = [dict(row, episode=ep) for row in self.adv.event_timer.log]
                self.trigger_logs.append(ep_trigger_log)
                history["trigger_logs"].append(ep_trigger_log)
                if log_dir is not None:
                    os.makedirs(log_dir, exist_ok=True)
                    self._write_rows_csv(f"{log_dir}/trigger_log_ep{ep:04d}.csv", ep_trigger_log)

            # mode-agnostic attack log rows for this episode
            ep_attack_rows = [r for r in self.adv.attack_log if r["episode"] == ep]
            ep_attacked_steps = sum(1 for r in ep_attack_rows if r["attacked"])

            if log_dir is not None:
                os.makedirs(log_dir, exist_ok=True)
                self._write_rows_csv(f"{log_dir}/attack_log_ep{ep:04d}.csv", ep_attack_rows)

            if ep % log_every == 0:
                msg = (f"[adv] ep {ep:4d} victim PDR {pdr:6.2f}% "
                       f"mean step-loss reward {ep_reward / t_per_ep:.4f} "
                       f"attacked {ep_attacked_steps}/{len(ep_attack_rows)} steps")
                if ep_trigger_log:
                    counts: Dict[str, int] = {}
                    for row in ep_trigger_log:
                        if row["trigger"] != "none":
                            counts[row["trigger"]] = counts.get(row["trigger"], 0) + 1
                    if counts:
                        breakdown = ", ".join(f"{k}={v}" for k, v in counts.items())
                        msg += f" [{breakdown}]"
                print(msg)

        # end-of-training summary: always shown, no extra calls needed
        print("\n" + "=" * 60)
        print("Attack timing summary (all episodes)")
        print("=" * 60)
        summary = self.attacks_per_episode()
        if summary:
            for row in summary:
                print(f"  ep {row['episode']:4d}  attacked {row['steps_attacked']:4d}/"
                      f"{row['total_steps']:4d}  ({row['attacked_fraction']:.1%})  "
                      f"budget={row['ep_budget_steps']}")
            total_attacked = sum(r["steps_attacked"] for r in summary)
            total_steps = sum(r["total_steps"] for r in summary)
            if total_steps:
                print(f"\nTotal steps attacked: {total_attacked} / {total_steps} "
                      f"({total_attacked / total_steps:.1%})")

        if self.cfg.timing_mode in ("event", "both"):
            all_trig = self.all_trigger_logs()
            if all_trig:
                print("\nTrigger breakdown (all episodes):")
                trig_counts: Dict[str, int] = {}
                tp = fn = fp = 0
                has_failure_data = False
                for row in all_trig:
                    trig_counts[row["trigger"]] = trig_counts.get(row["trigger"], 0) + 1
                    gt, proxy = row["ground_truth_failure"], row["proxy_failure_fired"]
                    if gt or proxy:
                        has_failure_data = True
                    if gt and proxy:
                        tp += 1
                    elif gt and not proxy:
                        fn += 1
                    elif not gt and proxy:
                        fp += 1
                for k, v in trig_counts.items():
                    print(f"  {k}: {v}")
                if has_failure_data:
                    print(f"\nFailure-proxy vs ground truth: TP={tp} FN={fn} FP={fp}")
        print("=" * 60 + "\n")

        return history

    def all_trigger_logs(self) -> List[Dict]:
        """Flatten every episode's trigger log collected so far into one list
        of dicts -- handy for a single budget-vs-damage / trigger-type
        analysis. JSON-serialisable."""
        flat: List[Dict] = []
        for ep_log in self.trigger_logs:
            flat.extend(ep_log)
        return flat

    def attacks_per_episode(self) -> List[Dict]:
        """Convenience summary: how many steps were attacked in each episode,
        and what fraction of that episode's budget was used. Returns a plain
        list of per-episode dicts, JSON-serialisable. Reads from
        self.adv.attack_log (the single shared gate log)."""
        by_episode: Dict[int, Dict] = {}
        for row in self.adv.attack_log:
            ep = row["episode"]
            entry = by_episode.setdefault(ep, {
                "episode": ep, "steps_attacked": 0, "total_steps": 0,
                "ep_budget_steps": row["ep_budget_steps"],
            })
            entry["total_steps"] += 1
            if row["attacked"]:
                entry["steps_attacked"] += 1
            entry["ep_budget_steps"] = max(entry["ep_budget_steps"], row["ep_budget_steps"])
        summary = []
        for ep in sorted(by_episode):
            entry = by_episode[ep]
            entry["attacked_fraction"] = (
                entry["steps_attacked"] / entry["total_steps"] if entry["total_steps"] else 0.0
            )
            summary.append(entry)
        return summary

    def save_logs_json(self, path: str):
        """Dump attack_log + all_trigger_logs to a single JSON file -- both
        are already plain lists/dicts, so this is a direct json.dump."""
        payload = {
            "attack_log": self.adv.attack_log,
            "trigger_logs": self.all_trigger_logs(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def save(self, path: str):
        self.adv.save(path)