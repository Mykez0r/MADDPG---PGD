#!/usr/bin/env python
"""Measure how much the domain clamp costs the attacker.

The FGSM baseline clamped only the first min(4, obs_dim) observation slots to
[0,1]; the learned adversary clamps EVERY feature (see
LearnedObservationAdversary._project, which falls back to a full clamp when no
bandwidth_indices are supplied). The full clamp is therefore a STRICT SUBSET of
the baseline's admissible set, so it can only weaken the attack - but by how
much depends entirely on where the observations actually sit. A clamp at 1.0
costs nothing if no feature is ever near 1.0, and costs the attacker everything
on a feature that is pinned at saturation.

This script samples real on-policy observations from the frozen victim and
reports, for a given epsilon, how much of the perturbation range the full clamp
removes relative to FGSM parity.

    python tools/diagnose_clamp.py --config reward_fix_full_config.json \
        --variant CC-Duelling-GNN --victim-models data/results/reward_fix/models \
        --episodes 5 --steps 256 --epsilon 0.30
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "src/attack_framework")
sys.path.insert(0, "src/maddpg_clean")

from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402

sys.path.insert(0, "tools")
from train_adversary import build_victim_and_env  # noqa: E402


def collect_observations(runner, maddpg, env, trainable, n_episodes, n_steps,
                         load):
    """Roll the FROZEN victim forward and record every trainable agent's obs."""
    hosts = env.engine.get_all_hosts()
    n_total = getattr(env.engine, "n_total_hosts", len(hosts))
    n_actions = maddpg.n_actions
    seen = []
    for _ in range(n_episodes):
        env.engine.reset_with_load(offered_load_factor=load)
        states = [env.engine.get_state(h) for h in hosts]
        for _t in range(n_steps):
            seen.extend(np.asarray(states[i], dtype=np.float32)
                        for i in trainable)
            t_states = [states[i] for i in trainable]
            t_actions = maddpg.choose_action(t_states)
            actions = runner._build_full_actions(t_actions, n_total, trainable,
                                                 n_actions)
            states, _r, _info = env.step(actions)
    return np.stack(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", default="CC-Duelling-GNN")
    ap.add_argument("--victim-models", default="data/results/reward_fix/models")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--load", type=float, default=2.0)
    ap.add_argument("--epsilon", type=float, default=0.30)
    ap.add_argument("--parity-slots", type=int, default=4,
                    help="how many leading slots the FGSM baseline clamped")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runner = StandaloneExperimentRunner(config_path=args.config,
                                        results_dir="data/results/_clampdiag")
    maddpg, env, trainable, obs_dim = build_victim_and_env(
        runner, args.variant, args.victim_models)

    X = collect_observations(runner, maddpg, env, trainable,
                             args.episodes, args.steps, args.load)
    eps = args.epsilon
    k = min(args.parity_slots, X.shape[1])

    # Admissible per-feature width under each rule.
    #   full   : [max(0, x-eps), min(1, x+eps)]           for every feature
    #   parity : the same for the first k slots, and the UNCLAMPED [x-eps, x+eps]
    #            (width 2*eps) for every feature beyond them
    lo_full = np.maximum(0.0, X - eps)
    hi_full = np.minimum(1.0, X + eps)
    w_full = hi_full - lo_full

    w_parity = np.full_like(w_full, 2.0 * eps)
    w_parity[:, :k] = w_full[:, :k]          # identical on the shared slots

    tail = slice(k, X.shape[1])               # where the two rules differ
    ratio = w_full[:, tail].sum() / w_parity[:, tail].sum()

    binds = w_full[:, tail] < (2.0 * eps - 1e-6)
    pinned_hi = X[:, tail] >= 1.0 - 1e-6
    pinned_lo = X[:, tail] <= 1e-6

    # Where does the loss concentrate? Report the worst feature indices.
    per_feat_ratio = (w_full[:, tail].mean(axis=0) / (2.0 * eps))
    order = np.argsort(per_feat_ratio)

    report = {
        "variant": args.variant,
        "epsilon": eps,
        "obs_dim": int(X.shape[1]),
        "parity_slots": k,
        "samples": int(X.shape[0]),
        "feature_value_distribution": {
            "at_zero_pct": float(100 * (X <= 1e-6).mean()),
            "at_one_pct": float(100 * (X >= 1.0 - 1e-6).mean()),
            "interior_pct": float(100 * ((X > 1e-6) & (X < 1.0 - 1e-6)).mean()),
            "mean": float(X.mean()), "median": float(np.median(X)),
        },
        "clamp_effect_on_unshared_features": {
            "width_ratio_full_over_parity": float(ratio),
            "range_removed_pct": float(100 * (1 - ratio)),
            "features_where_clamp_binds_pct": float(100 * binds.mean()),
            "pinned_at_one_pct": float(100 * pinned_hi.mean()),
            "pinned_at_zero_pct": float(100 * pinned_lo.mean()),
        },
        "tightest_features": [
            {"index": int(k + i), "mean_width_frac_of_2eps":
             float(per_feat_ratio[i])} for i in order[:8]
        ],
        "loosest_features": [
            {"index": int(k + i), "mean_width_frac_of_2eps":
             float(per_feat_ratio[i])} for i in order[-4:]
        ],
    }

    print(json.dumps(report, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
