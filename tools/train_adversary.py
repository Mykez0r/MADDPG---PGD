#!/usr/bin/env python
"""
Entry point for the learned worst-case observation adversary (MSc follow-up).

It reuses the existing experiment runner to build the environment and load a
FROZEN trained victim, then trains a SA-MDP DDPG adversary against it and (optionally)
evaluates the trained adversary through the SAME attack loop used for FGSM, so the
numbers are directly comparable to the FGSM/PGD results in the paper.

Usage (inside the maddpg-exp docker image, GPU optional):
    python tools/train_adversary.py \
        --config reward_fix_full_config.json \
        --variant CC-Simple \
        --episodes 300 --load 2.0 --failures 0 \
        --epsilon 0.30 \
        --out host_data/results/learned_adv/CC-Simple

Then evaluate the trained adversary against the damage ceiling / random control:
    python tools/train_adversary.py ... --eval-only --adv-ckpt <path>/adversary.pt

This is a SCAFFOLD. The two research extensions (coordinated, timed) are flagged
with TODO(student) in src/attack_framework/learned_adversary.py; enabling them is
the point of the project.
"""
import argparse
import json
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "src/attack_framework")
sys.path.insert(0, "src/maddpg_clean")

from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402
from learned_adversary import (  # noqa: E402
    AdversaryConfig, AdversaryTrainer, LearnedObservationAdversary,
    RandomControlAdversary,
)

# two-sided t critical values at 95%, indexed by degrees of freedom (n-1)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 24: 2.064, 29: 2.045, 39: 2.023, 59: 2.001}


def _t95(df: int) -> float:
    if df in _T95:
        return _T95[df]
    smaller = [k for k in _T95 if k <= df]
    return _T95[max(smaller)] if smaller else 1.96


def _paired_diff(a_series, b_series):
    """Mean and 95% CI of the PAIRED per-episode difference (a_i - b_i).

    The arms are run on identical traffic and identical link failures (same
    traffic_seed), so episode i of one arm and episode i of another are the same
    network under a different attack. Differencing within the pair removes the
    episode-to-episode variance that otherwise swamps the effect — which is the
    whole reason the pairing has to be preserved upstream. An unpaired CI over
    these means would be far too wide to resolve the gap.
    """
    if not a_series or not b_series or len(a_series) != len(b_series):
        return None, None, None
    d = [float(x) - float(y) for x, y in zip(a_series, b_series)]
    n = len(d)
    mean = statistics.fmean(d)
    if n < 2:
        return mean, None, None
    half = _t95(n - 1) * statistics.stdev(d) / math.sqrt(n)
    return mean, mean - half, mean + half


def _link_victim_models(runner: "StandaloneExperimentRunner", victim_models: str):
    """Make <results_dir>/models resolve to the trained victim weights.

    _make_variant loads each agent from `<results_dir>/models/<variant>/...`, so
    the trained checkpoints (which live OUTSIDE the repo, e.g. reward_fix/models)
    must be linked in. Mirrors the symlink guard used by the FGSM run scripts.
    """
    victim_models = os.path.abspath(victim_models)
    if not os.path.isdir(victim_models):
        raise SystemExit(
            f"victim weights not found at {victim_models}. The trained MADDPG "
            "checkpoints are NOT in the repo (host_data/ is gitignored) — obtain "
            "them from the server/shared drive and pass --victim-models <dir>.")
    link = os.path.join(runner.results_dir, "models")
    if os.path.islink(link) or not os.path.exists(link):
        if os.path.islink(link):
            os.unlink(link)
        os.symlink(victim_models, link)
    elif os.path.realpath(link) != victim_models:
        raise SystemExit(
            f"{link} exists and is not a symlink to {victim_models}; refusing to "
            "overwrite. Remove it or choose a fresh --out directory.")


def build_victim_and_env(runner: "StandaloneExperimentRunner", variant_name: str,
                         victim_models: str):
    """Load the named trained victim + an attack env, mirroring fgsm_probe()."""
    _link_victim_models(runner, victim_models)
    vcfg = next(v for v in runner.config["variants"] if v["name"] == variant_name)
    maddpg, _, _ = runner._make_variant(vcfg)

    # HARD GUARD: load_checkpoint() no-ops silently if the file is missing, so an
    # untrained victim would train the adversary against garbage (PDR ~55% vs the
    # trained ~87%). Verify a checkpoint actually exists before proceeding.
    a0 = maddpg.agents[0]
    best = list(getattr(a0, "best_checkpoint_files", {}).values())
    final = getattr(a0.actor, "checkpoint_file", None)
    have = (best and all(os.path.exists(p) for p in best)) or \
           (final and os.path.exists(final))
    if not have:
        raise SystemExit(
            f"No victim checkpoint for '{variant_name}' under "
            f"{runner.results_dir}/models/{variant_name}. Point --victim-models at "
            "the directory that contains the trained per-variant weights.")
    runner._load_variant_checkpoint(maddpg, variant_name)

    # freeze the victim
    for ag in maddpg.agents:
        ag.actor.eval()
        for p in ag.actor.parameters():
            p.requires_grad_(False)
    hotspot = (runner.config.get("attack_eval", {}) or {}).get("hotspot")
    env = runner._make_attack_env(hotspot)
    trainable = env.engine.trainable_host_indices
    obs_dim = len(env.engine.get_state(env.engine.get_all_hosts()[trainable[0]]))
    return maddpg, env, trainable, obs_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", default="CC-Simple")
    # Defaults are CONTAINER-relative: the maddpg-exp helper mounts host_data/ as
    # /workspace/data, so inside the container the weights are at data/... (not
    # host_data/...). All documented runs are in-container.
    ap.add_argument("--victim-models", default="data/results/reward_fix/models",
                    help="dir with the trained victim weights (NOT in the repo; "
                         "get from the server/shared drive). Linked to <out>/models. "
                         "Path is relative to the container workdir (data/ = host_data/).")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--load", type=float, default=2.0)
    ap.add_argument("--failures", type=int, default=0)
    ap.add_argument("--epsilon", type=float, default=0.30)
    ap.add_argument("--out", default="data/results/learned_adv")  # container-relative
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--adv-ckpt", default=None)
    ap.add_argument("--eval-episodes", type=int, default=15,
                    help="paired episodes per arm (clean/random/learned) at eval")
    ap.add_argument("--traffic-seed", type=int, default=20240601,
                    help="shared seed that keeps the three eval arms PAIRED on the "
                         "same traffic and the same sampled link failures; the "
                         "paired CIs are only valid if every arm uses it")
    ap.add_argument("--coordinate", action="store_true",
                    help="(A) joint multi-agent perturbation. Training only — at "
                         "eval the mode is read from the checkpoint.")
    ap.add_argument("--timing-budget", type=float, default=None,
                    help="TODO(student B): fraction of steps the attacker may act")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runner = StandaloneExperimentRunner(config_path=args.config,
                                        results_dir=args.out)
    maddpg, env, trainable, obs_dim = build_victim_and_env(
        runner, args.variant, args.victim_models)
    cfg = AdversaryConfig(epsilon=args.epsilon,
                          coordinate=args.coordinate,
                          timing_budget=args.timing_budget)

    if args.eval_only:
        assert args.adv_ckpt, "--eval-only needs --adv-ckpt"
        # How the checkpoint was TRAINED (coordinate / group_size) is recorded in
        # the checkpoint itself. Derive the adversary's shape from that rather
        # than from the CLI flags, so an eval can never silently mismatch the
        # trained actor — forgetting --coordinate used to fail with a bare
        # "group_size=N does not match 1", and passing it failed for want of
        # group_hosts. Neither flag is needed at eval time now.
        meta = torch.load(args.adv_ckpt, map_location="cpu")
        cfg.coordinate = bool(meta.get("coordinate", False))
        group_hosts = None
        if cfg.coordinate:
            all_hosts = env.engine.get_all_hosts()
            # SAME ordering the trainer used (AdversaryTrainer.__init__), so
            # position i is the agent_index the runner passes for host i.
            group_hosts = [all_hosts[i] for i in trainable]
            ckpt_group = int(meta.get("group_size", len(group_hosts)))
            if ckpt_group != len(group_hosts):
                raise SystemExit(
                    f"checkpoint was trained on a group of {ckpt_group} agents but "
                    f"this env exposes {len(group_hosts)} trainable agents for "
                    f"variant '{args.variant}'. A coordinated actor is sized for the "
                    "JOINT observation, so it can only be evaluated on the same "
                    "group — check that --variant/--config match the training run.")
        adv = LearnedObservationAdversary(obs_dim, cfg,
                                          group_hosts=group_hosts).load(args.adv_ckpt)
        mode = "coordinated" if cfg.coordinate else "independent"
        print(f"[eval] mode={mode} group_size={adv.group_size} "
              f"epsilon={args.epsilon} timing_budget={args.timing_budget} "
              f"episodes={args.eval_episodes} seed={args.traffic_seed}")

        # PAIRING: every arm gets the same traffic_seed, load and failure count.
        # _attack_episodes re-seeds the global RNGs from traffic_seed at entry, so
        # the injected traffic AND the sampled link failures are identical across
        # clean / random / learned. Without this the per-episode differences below
        # would be comparing different networks and the CIs would be meaningless.
        common = dict(offered_load_factor=args.load,
                      n_link_failures=args.failures,
                      traffic_seed=args.traffic_seed)
        n_eval = args.eval_episodes

        runner.attack_framework = adv
        clean = runner._attack_episodes(maddpg, env, n_eval, args.steps,
                                        attack=False, **common)

        # RANDOM CONTROL: same epsilon-ball and same timing gate, random direction.
        # Uses its own Generator so it consumes no entropy from the seeded global
        # streams and the pairing survives.
        rnd = RandomControlAdversary(obs_dim, cfg, seed=args.traffic_seed)
        runner.attack_framework = rnd
        random_arm = runner._attack_episodes(maddpg, env, n_eval, args.steps,
                                             attack=True, attack_type="random",
                                             epsilon=args.epsilon,
                                             measure_flips=True, **common)

        runner.attack_framework = adv
        learned_arm = runner._attack_episodes(maddpg, env, n_eval, args.steps,
                                              attack=True, attack_type="learned",
                                              epsilon=args.epsilon,
                                              measure_flips=True, **common)

        c_pdr = clean["mean_end_to_end_pdr"]
        r_pdr = random_arm["mean_end_to_end_pdr"]
        l_pdr = learned_arm["mean_end_to_end_pdr"]
        # HEADLINE: adversarial-specific gap = how much the LEARNED direction cost
        # the victim beyond what a random perturbation of the same size already
        # costs. Paired per-episode, so the CI is valid.
        gap_mean, gap_lo, gap_hi = _paired_diff(random_arm["pdr_series"],
                                                learned_arm["pdr_series"])
        raw_mean, raw_lo, raw_hi = _paired_diff(clean["pdr_series"],
                                                learned_arm["pdr_series"])
        rnd_mean, rnd_lo, rnd_hi = _paired_diff(clean["pdr_series"],
                                                random_arm["pdr_series"])

        result = {
            "variant": args.variant,
            "mode": mode,
            "coordinate": cfg.coordinate,
            "group_size": adv.group_size,
            "epsilon": args.epsilon,
            "timing_budget": args.timing_budget,
            "offered_load_factor": args.load,
            "n_link_failures": args.failures,
            "episodes": n_eval,
            "steps_per_episode": args.steps,
            "traffic_seed": args.traffic_seed,
            "paired": True,
            "pdr": {
                "clean": c_pdr,
                "random": r_pdr,
                "learned": l_pdr,
                "clean_series": clean["pdr_series"],
                "random_series": random_arm["pdr_series"],
                "learned_series": learned_arm["pdr_series"],
            },
            # OUTCOMES — what actually happened to delivery.
            "outcomes": {
                "adversarial_gap_pp": gap_mean,
                "adversarial_gap_ci95": [gap_lo, gap_hi],
                "raw_drop_pp": raw_mean,
                "raw_drop_ci95": [raw_lo, raw_hi],
                "random_drop_pp": rnd_mean,
                "random_drop_ci95": [rnd_lo, rnd_hi],
            },
            # DECISIONS — how much routing the attack changed. Deliberately kept
            # separate from outcomes: a high flip rate with a flat PDR means the
            # network ABSORBED the attack, it does not mean the attack worked.
            "decisions": {
                "learned_action_flip_rate": learned_arm.get("action_flip_rate"),
                "random_action_flip_rate": random_arm.get("action_flip_rate"),
            },
            "warnings": [],
        }

        w = result["warnings"]
        if gap_lo is not None and gap_lo <= 0.0 <= gap_hi:
            # Report the resolution too: "CI includes 0" at small n can mean
            # "no effect" OR "underpowered". At n=15 this test only resolves a
            # ~2pp gap about two thirds of the time, so the two must not be
            # conflated — raise --eval-episodes before concluding "no effect".
            half = (gap_hi - gap_lo) / 2.0
            result["outcomes"]["min_detectable_effect_pp"] = half
            w.append(
                f"adversarial gap CI includes 0 — no evidence the learned attack "
                f"beats the random control at this budget. NOTE this run only "
                f"resolves effects larger than ~{half:.2f}pp (n={n_eval}); a "
                f"smaller true gap would be invisible here, so this is not proof "
                f"of no effect. Do NOT report the raw drop as attack damage.")
        if gap_hi is not None and gap_hi < 0.0:
            w.append(
                f"adversarial gap is significantly NEGATIVE ({gap_mean:+.2f}pp, CI "
                f"[{gap_lo:+.2f}, {gap_hi:+.2f}]) — the learned adversary is WORSE than "
                "random noise of the same size. That indicts the TRAINING RUN, not the "
                "victim's robustness: check that attack_loss_reward actually rose in "
                "train_history.json before drawing any conclusion about the ceiling.")
        flip = learned_arm.get("action_flip_rate")
        rflip = random_arm.get("action_flip_rate")
        if flip is not None and flip >= 0.05 and gap_mean is not None and gap_mean <= 0.5:
            extra = f" (random control flips {rflip:.1%})" if rflip is not None else ""
            w.append(
                f"flip rate {flip:.1%}{extra} but adversarial gap only {gap_mean:+.2f}pp "
                "— the network is ABSORBING the flips (K-path redundancy). Decisions "
                "changed is NOT outcomes changed; do not report the flip rate as success.")
        if c_pdr < 60.0:
            w.append(
                f"clean PDR is only {c_pdr:.1f}% — this is a FAILURE-DOMINATED cell "
                "(the network self-collapses without any attack). Not "
                "attack-informative; the raw drop here is mostly the topology.")
        if n_eval < 10:
            w.append(
                f"n={n_eval} episodes is small — CIs are wide and this cell is "
                "indicative only. The degenerate high-failure cells (n~6) in the "
                "FGSM study were exactly this.")

        print(json.dumps({k: v for k, v in result.items() if k != "pdr"}, indent=2))
        print(f"\n[eval] clean {c_pdr:.2f}%  random {r_pdr:.2f}%  learned {l_pdr:.2f}%")
        print(f"[eval] ADVERSARIAL GAP (random-learned) {gap_mean:+.2f}pp "
              f"95%CI [{gap_lo:+.2f}, {gap_hi:+.2f}]"
              if gap_lo is not None else
              f"[eval] ADVERSARIAL GAP (random-learned) {gap_mean:+.2f}pp")
        for line in w:
            print(f"[eval][WARN] {line}")

        # Mode-tagged file so a coordinated eval does not clobber the independent
        # one you are comparing it against; the stable name is kept for continuity.
        json.dump(result, open(os.path.join(args.out, "learned_adv_eval.json"), "w"),
                  indent=2)
        json.dump(result, open(os.path.join(args.out, f"learned_adv_eval_{mode}.json"), "w"),
                  indent=2)
        return

    trainer = AdversaryTrainer(
        victim=maddpg, env=env, trainable_indices=trainable, obs_dim=obs_dim,
        cfg=cfg, build_full_actions=runner._build_full_actions,
    )
    history = trainer.train(n_episodes=args.episodes, t_per_ep=args.steps,
                            offered_load_factor=args.load,
                            n_link_failures=args.failures)
    ckpt = os.path.join(args.out, "adversary.pt")
    trainer.save(ckpt)
    json.dump(history, open(os.path.join(args.out, "train_history.json"), "w"))
    print(f"saved adversary -> {ckpt}")
    print("next: re-run with --eval-only --adv-ckpt", ckpt,
          "to score it against the damage ceiling / random control.")


if __name__ == "__main__":
    main()
