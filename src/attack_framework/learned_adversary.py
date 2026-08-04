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

  (B) is now implemented with two selectable timing strategies:
    - "quantile": the original adaptive-threshold gate (attack the top-k% of
      steps by a rolling critical-state score).
    - "event":    fires only on a congestion-onset rising edge or during a fixed
      post-failure recovery window (CongestionFailureTimer below).
    - "both":     event gate OR quantile gate.
  Every step's decision is logged (trigger label + underlying signals) via
  CongestionFailureTimer.log for post-hoc analysis, AND a mode-agnostic
  per-step attack_log records every attack decision tagged with its episode
  number regardless of timing_mode (see AdversaryTrainer.attack_log_df /
  attacks_per_episode). Both logs are now also PRINTED automatically during
  and after train() -- no extra calls needed to see what happened.

Eval: `LearnedObservationAdversary` exposes `generate_adversarial_state(...)` with
the SAME signature as FGSMAttackFramework, so a trained adversary drops straight
into `standalone_experiment_runner._attack_episodes` via attack_type='learned'
and is scored by the existing damage-ceiling / random-control / action-flip metrics.

Run the trainer with tools/train_adversary.py (wires up env + frozen victim).
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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

    Every step's decision is appended to `self.log` (trigger label + the
    underlying signals) so training/eval runs can be analysed post-hoc, e.g.
    correlating trigger type with victim PDR or plotting utilisation/loss
    with markers at each trigger event.
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

    def log_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)


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
        self.epsilon = cfg.epsilon  # runner sets this per case; kept in sync
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
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.buffer = ReplayBuffer()
        self._step = 0

        # (B) timing-budget bookkeeping
        self._score_hist: deque = deque(maxlen=cfg.timing_window)
        self._ep_budget_steps: int = 0
        self._ep_used: int = 0

        # (B) event-driven timing (congestion onset / post-failure)
        self.event_timer = CongestionFailureTimer(cfg)
        self._pending_states: Optional[List[np.ndarray]] = None
        self._last_loss_frac: float = 0.0
        self._last_ground_truth_failure: bool = False
        self._cur_step_in_ep: int = 0
        self.trigger_logs: List[pd.DataFrame] = []  # one df per completed episode

        # (B) mode-agnostic attack log: records EVERY step's attack decision
        # tagged with its episode, regardless of timing_mode (quantile/event/
        # both). Use this for "which episode did it attack in" analysis --
        # trigger_logs above only exists for "event"/"both" modes.
        self._cur_episode: int = -1
        self.attack_log: List[Dict] = []

    # -- perturb every compromised agent's observation this step -----------
    def _attack_states(self, states: List[np.ndarray], explore: bool
                        ) -> Tuple[List[np.ndarray], List[Tuple[int, np.ndarray, np.ndarray]]]:
        """Returns (perturbed_states, transitions) where each transition is
        (topo_idx, clean_obs, applied_delta) for the replay buffer."""
        adv_states = list(states)
        transitions = []
        self._pending_states = states  # let the event timer see this step's obs

        # (B) strategically-timed / critical-state gate: decide ONCE per step
        # whether this is a high-leverage moment worth spending L0 budget on.
        # Scoring is cheap (max utilisation) or uses the critic (Q-saliency),
        # or event-driven (congestion onset / post-failure recovery window).
        # If cfg.timing_budget is None the gate is disabled and every step is
        # attacked (original behaviour).
        score = self._critical_score(states)
        attacked = self._should_attack(score)

        # mode-agnostic per-step attack log, tagged with the current episode
        # (works for quantile/event/both, unlike CongestionFailureTimer.log
        # which is only populated in "event"/"both" modes).
        self.attack_log.append({
            "episode": self._cur_episode,
            "step": self._cur_step_in_ep,
            "attacked": attacked,
            "score": score,
            "timing_mode": self.cfg.timing_budget is not None and self.cfg.timing_mode or "disabled",
            "ep_used": self._ep_used,
            "ep_budget_steps": self._ep_budget_steps,
        })

        if not attacked:
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
        # TODO(student A -- coordinate): replace the independent per-agent loop above
        # with a JOINT perturbation. Concatenate the compromised agents' observations,
        # let a single actor emit a joint delta, and share a critic that sees the
        # global link state so the adversary can push multiple flows onto ONE shared
        # surviving link (the mechanism that actually reaches the damage ceiling).
        return adv_states, transitions

    # -- (B) critical-state scoring ----------------------------------------
    def _critical_score(self, states: List[np.ndarray]) -> float:
        """Higher = more worth attacking right now. Only used when
        cfg.timing_budget is set and timing_mode uses the quantile gate."""
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

    def _quantile_gate(self, score: float) -> bool:
        """Original adaptive-threshold L0 gate. Attacks the top
        (1 - timing_budget) fraction of observed critical-state scores,
        tracked against a rolling window."""
        self._score_hist.append(score)
        if len(self._score_hist) < max(10, self.cfg.timing_window // 4):
            threshold = 0.0  # not enough history yet -> attack to seed the buffer
        else:
            q = max(0.0, min(1.0, 1.0 - self.cfg.timing_budget))
            threshold = float(np.quantile(self._score_hist, q))
        return score >= threshold

    def _should_attack(self, score: float) -> bool:
        """(B) L0 gate combining an episode budget with either the quantile
        critical-state gate, the event-driven congestion/failure gate, or both."""
        if self.cfg.timing_budget is None:
            return True  # gate disabled -> attack every step (original behaviour)

        if self._ep_used >= self._ep_budget_steps:
            return False  # budget for this episode is exhausted

        mode = self.cfg.timing_mode
        event_hit = False
        if mode in ("event", "both"):
            event_hit = self.event_timer.update_and_check(
                self._cur_step_in_ep, self._pending_states, self.trainable_indices,
                self.adv.bandwidth_indices, self._last_loss_frac,
                ground_truth_failure=self._last_ground_truth_failure)
            if mode == "event":
                return event_hit

        quantile_hit = self._quantile_gate(score)
        if mode == "both":
            return quantile_hit or event_hit
        return quantile_hit

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
              failure_injector: Optional[Callable[["object", int], Optional[Sequence]]] = None,
              log_every: int = 10, log_dir: Optional[str] = None) -> Dict:
        """Main SA-MDP loop. Returns a small history dict for plotting.

        `failure_injector(env, step) -> failed_links | None` is an optional
        callable wired in at the call site (e.g. the runner's existing failure
        injector). If it returns a non-empty/non-None result on a given step,
        that step is tagged as a GROUND-TRUTH failure event and compared
        against the loss-spike proxy in the trigger log (see the
        `ground_truth_failure` column). If `n_link_failures > 0` but no
        `failure_injector` is supplied, ground truth is left unavailable and
        only the proxy signal drives the post-failure gate (previous
        behaviour, now made explicit rather than silently a no-op).

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
            self._cur_episode = ep  # tag every attack-log row with the right episode
            self.env.engine.reset_with_load(offered_load_factor=offered_load_factor)
            # (B) reset the per-episode L0 budget so timing gates apply fresh
            # each episode (budget is a FRACTION of t_per_ep, not cumulative).
            if self.cfg.timing_budget is not None:
                self._ep_budget_steps = max(1, int(np.ceil(self.cfg.timing_budget * t_per_ep)))
                self._ep_used = 0
            self.event_timer.reset()
            self._last_loss_frac = 0.0
            self._last_ground_truth_failure = False
            states = [self.env.engine.get_state(h) for h in self.hosts]
            ep_reward = 0.0
            explore = self._step < self.cfg.warmup_steps or True  # keep light noise
            for t in range(t_per_ep):
                self._cur_step_in_ep = t

                # ground-truth failure injection (if a real injector is wired in)
                gt_failure = False
                if n_link_failures and failure_injector is not None:
                    injected = failure_injector(self.env, t)
                    gt_failure = bool(injected)
                self._last_ground_truth_failure = gt_failure

                adv_states, transitions = self._attack_states(states, explore)
                actions = self._victim_actions(adv_states)
                next_states, _rewards, info = self.env.step(actions)
                # SA-MDP reward: victim per-step packet loss fraction (attacker gain)
                loss_frac = float(info.get("packet_loss_rate", 0.0)) / 100.0
                self._last_loss_frac = loss_frac  # feeds the event timer next step
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

            ep_log = pd.DataFrame()
            if self.cfg.timing_mode in ("event", "both"):
                ep_log = self.event_timer.log_df()
                ep_log["episode"] = ep
                self.trigger_logs.append(ep_log)
                history["trigger_logs"].append(ep_log)
                if log_dir is not None:
                    import os
                    os.makedirs(log_dir, exist_ok=True)
                    ep_log.to_csv(f"{log_dir}/trigger_log_ep{ep:04d}.csv", index=False)

            # mode-agnostic attack log: always written if log_dir is set, so
            # you can see which episode(s) an attack fired in even under pure
            # "quantile" mode (where trigger_logs above stays empty).
            if log_dir is not None:
                import os
                os.makedirs(log_dir, exist_ok=True)
                ep_attack_rows = [r for r in self.attack_log if r["episode"] == ep]
                pd.DataFrame(ep_attack_rows).to_csv(
                    f"{log_dir}/attack_log_ep{ep:04d}.csv", index=False)

            # per-episode attack breakdown (mode-agnostic; always available)
            ep_attack_rows = [r for r in self.attack_log if r["episode"] == ep]
            ep_attacked_steps = sum(1 for r in ep_attack_rows if r["attacked"])

            if ep % log_every == 0:
                msg = (f"[adv] ep {ep:4d} victim PDR {pdr:6.2f}% "
                       f"mean step-loss reward {ep_reward / t_per_ep:.4f} "
                       f"attacked {ep_attacked_steps}/{len(ep_attack_rows)} steps")
                if self.cfg.timing_mode in ("event", "both") and not ep_log.empty:
                    counts = ep_log["trigger"].value_counts().to_dict()
                    counts.pop("none", None)
                    if counts:
                        breakdown = ", ".join(f"{k}={v}" for k, v in counts.items())
                        msg += f" [{breakdown}]"
                print(msg)

        # end-of-training summary: always shown, no extra calls needed
        print("\n" + "=" * 60)
        print("Attack timing summary (all episodes)")
        print("=" * 60)
        summary = self.attacks_per_episode()
        if not summary.empty:
            print(summary.to_string(index=False))
            print(f"\nTotal steps attacked: {int(summary['steps_attacked'].sum())} / "
                  f"{int(summary['total_steps'].sum())} "
                  f"({summary['steps_attacked'].sum() / summary['total_steps'].sum():.1%})")
        if self.cfg.timing_mode in ("event", "both"):
            trig_df = self.all_trigger_logs()
            if not trig_df.empty:
                print("\nTrigger breakdown (all episodes):")
                print(trig_df["trigger"].value_counts().to_string())
                if trig_df["ground_truth_failure"].any() or trig_df["proxy_failure_fired"].any():
                    tp = int((trig_df.ground_truth_failure & trig_df.proxy_failure_fired).sum())
                    fn = int((trig_df.ground_truth_failure & ~trig_df.proxy_failure_fired).sum())
                    fp = int((~trig_df.ground_truth_failure & trig_df.proxy_failure_fired).sum())
                    print(f"\nFailure-proxy vs ground truth: TP={tp} FN={fn} FP={fp}")
        print("=" * 60 + "\n")

        return history

    def all_trigger_logs(self) -> pd.DataFrame:
        """Concatenate every episode's trigger log collected so far into one
        DataFrame -- handy for a single budget-vs-damage / trigger-type analysis."""
        if not self.trigger_logs:
            return pd.DataFrame()
        return pd.concat(self.trigger_logs, ignore_index=True)

    def attack_log_df(self) -> pd.DataFrame:
        """Mode-agnostic per-step attack decision log (works for quantile,
        event, and both modes). Columns: episode, step, attacked, score,
        timing_mode, ep_used, ep_budget_steps."""
        return pd.DataFrame(self.attack_log)

    def attacks_per_episode(self) -> pd.DataFrame:
        """Convenience summary: how many steps were attacked in each episode,
        and what fraction of that episode's budget was used."""
        df = self.attack_log_df()
        if df.empty:
            return df
        summary = (df.groupby("episode")
                     .agg(steps_attacked=("attacked", "sum"),
                          total_steps=("attacked", "count"),
                          ep_budget_steps=("ep_budget_steps", "max"))
                     .reset_index())
        summary["attacked_fraction"] = summary["steps_attacked"] / summary["total_steps"]
        return summary

    def save(self, path: str):
        self.adv.save(path)