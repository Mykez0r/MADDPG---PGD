#!/usr/bin/env python
"""Compare a variant's eval under the full clamp against the same eval at FGSM parity.

The two runs share a traffic seed, episode count and checkpoint, differing only
in the admissible set, so their per-episode PDR series are PAIRED. That allows a
confidence interval on the difference itself rather than the much weaker check of
whether two separately-computed gap intervals happen to overlap.

Reported per matched (variant, timing budget):
  * the adversarial gap under each clamp, with its own 95% CI
  * the PAIRED difference in attack-arm PDR, full minus parity. Positive means
    the parity attack drove delivery LOWER, i.e. the full clamp was costing the
    attacker real damage.
  * the flip rate under each clamp

    python tools/compare_clamps.py --results-root host_data/results/learned_adv
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

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


def paired_diff(a, b):
    """Mean and 95% CI of the paired difference (a_i - b_i)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return (d[0] if d else 0.0), None, None
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    h = _t95(n - 1) * se
    return m, m - h, m + h


def load(results_root, mode):
    """{(variant, budget): {clamp: record}} over every eval JSON found."""
    out = {}
    files = set(glob.glob(os.path.join(results_root, "*", "**",
                                       f"learned_adv_eval_{mode}.json"),
                          recursive=True))
    files |= set(glob.glob(os.path.join(results_root, "*",
                                        f"learned_adv_eval_{mode}.json")))
    for f in sorted(files):
        if "smoke" in f:
            continue
        d = json.load(open(f, encoding="utf-8"))
        key = (d["variant"], d.get("timing_budget"))
        out.setdefault(key, {})[d.get("domain_clamp", "full")] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="host_data/results/learned_adv")
    ap.add_argument("--mode", default="independent")
    ap.add_argument("--epsilon", type=float, default=None,
                    help="restrict to one epsilon; default compares all")
    args = ap.parse_args()

    data = load(args.results_root, args.mode)
    rows = []
    for (variant, budget), byclamp in sorted(data.items(), key=lambda kv: str(kv[0])):
        if "full" not in byclamp or "fgsm_parity" not in byclamp:
            continue
        full, par = byclamp["full"], byclamp["fgsm_parity"]
        if args.epsilon is not None and abs(full["epsilon"] - args.epsilon) > 1e-9:
            continue
        # Pairing is only meaningful if the two runs really did see the same
        # episodes; refuse to compare them otherwise rather than report a number
        # that looks paired and is not.
        if (full["traffic_seed"] != par["traffic_seed"]
                or full["episodes"] != par["episodes"]):
            print(f"  !! {variant} tb={budget}: seed/episode mismatch, not paired "
                  f"({full['traffic_seed']}/{full['episodes']} vs "
                  f"{par['traffic_seed']}/{par['episodes']}) — skipped")
            continue
        m, lo, hi = paired_diff(full["pdr"]["attack_series"],
                                par["pdr"]["attack_series"])
        rows.append((variant, budget, full, par, m, lo, hi))

    if not rows:
        raise SystemExit(f"no variant under {args.results_root} has BOTH a full "
                         f"and an fgsm_parity {args.mode} eval to compare")

    hdr = (f"{'variant':17s} {'budget':7s} {'gap full':>9s} {'gap parity':>11s} "
           f"{'parity-full':>12s} {'95% CI':>20s}  {'flips f/p':>13s}")
    print(hdr)
    print("-" * len(hdr))
    for variant, budget, full, par, m, lo, hi in rows:
        gf = full["outcomes"]["adversarial_gap_pp"]
        gp = par["outcomes"]["adversarial_gap_pp"]
        ff = 100 * full["decisions"]["attack_action_flip_rate"]
        fp = 100 * par["decisions"]["attack_action_flip_rate"]
        ci = f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None else "n/a"
        verdict = "" if lo is None or (lo <= 0 <= hi) else "  <-- real"
        print(f"{variant:17s} {str(budget):7s} {gf:+9.3f} {gp:+11.3f} "
              f"{gp - gf:+12.3f} {ci:>20s}  {ff:5.2f}/{fp:5.2f}{verdict}")

    print()
    print("'parity-full' is the change in adversarial gap when the attacker is")
    print("given the FGSM baseline's looser admissible set. The CI is on the")
    print("PAIRED per-episode difference in attack-arm PDR (full minus parity):")
    print("positive and excluding zero means parity delivered strictly more damage.")


if __name__ == "__main__":
    main()
