#!/usr/bin/env python
"""Heatmap of why the timing gate fired, for every variant and timing budget.

Reads the `timing.<arm>.reasons` block that tools/train_adversary.py --eval-only
writes into each gated eval JSON (runs from commit 00341c2 on). Every attacked
step is attributed to one trigger: the rule that admitted it (its score was
above the rolling threshold, or it tied the threshold and tie rationing let it
through) crossed with the congestion band its score fell in. One panel per
trigger that occurred in at least one run; rows are victim variants, columns
the requested budget, and each cell holds that trigger's share of the run's
attacked steps, so the panels of one cell sum to 100%.

    python tools/plot_trigger_reasons.py --results-root host_data/results/learned_adv_parity \
        --mode independent --domain-clamp fgsm_parity --out trigger_reasons.png

Design notes (dataviz skill):
  * magnitude on a grid, so a sequential single-hue ramp (reference blue,
    steps 100-700) on one shared 0-100% scale across panels, with a scale legend.
  * cell text carries the share and the raw count; share alone hides that a
    constant number of near-saturation states fires at every budget, which is
    why its share falls as the budget grows.
  * ungated runs have no gate and are skipped; triggers that never fired in any
    run are left out and named in the footnote instead of drawn as empty panels.
  * a CSV table view is written next to the PNG.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

VARIANT_ORDER = ["CC-Simple", "CC-Duelling", "CC-Simple-GNN", "CC-Duelling-GNN",
                 "LC-Simple", "LC-Duelling", "LC-Duelling-GNN"]
RULES = ("above_threshold", "tie_admitted")
BANDS = ("saturated", "near_saturation", "high_congestion", "moderate")
RULE_LABEL = {"above_threshold": "Score above threshold",
              "tie_admitted": "Tie at threshold, admitted by rationing"}

# Reference sequential blue, steps 100 -> 700.
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"


def band_label(band, cutoffs):
    hi = {"saturated": None, "near_saturation": cutoffs["saturated"],
          "high_congestion": cutoffs["near_saturation"], "moderate": cutoffs["high_congestion"]}
    lo = {"saturated": cutoffs["saturated"], "near_saturation": cutoffs["near_saturation"],
          "high_congestion": cutoffs["high_congestion"], "moderate": None}
    name = band.replace("_", " ")
    if hi[band] is None:
        return f"{name} (score ≥ {lo[band]:.2f})"
    if lo[band] is None:
        return f"{name} (score < {hi[band]:.2f})"
    return f"{name} ({lo[band]:.2f} ≤ score < {hi[band]:.2f})"


def load(results_root, epsilon, mode, arm, domain_clamp=None):
    """Return ({(variant, budget): reasons+meta}, clamp, episodes, skipped_files)."""
    cells, clamps, episodes, skipped = {}, set(), set(), []
    pattern = os.path.join(results_root, "*", "**", f"learned_adv_eval_{mode}.json")
    for f in sorted(set(glob.glob(pattern, recursive=True))):
        if "smoke" in f:
            continue
        d = json.load(open(f, encoding="utf-8"))
        if abs(float(d.get("epsilon", -1)) - epsilon) > 1e-9:
            continue
        clamp = d.get("domain_clamp", "full")
        if domain_clamp is not None and clamp != domain_clamp:
            continue
        t = (d.get("timing") or {}).get(arm) or {}
        if not t.get("gated"):
            continue                                   # ungated: no gate, no reasons
        if "reasons" not in t:
            skipped.append(f)                          # gated, but predates reason logging
            continue
        key = (d["variant"], d.get("timing_budget"))
        if key in cells:
            raise SystemExit(f"two eval files for {key[0]} at budget {key[1]}:\n"
                             f"  {cells[key]['file']}\n  {f}")
        clamps.add(clamp)
        episodes.add(d.get("episodes"))
        cells[key] = {"file": f, "reasons": t["reasons"], "attacked": t.get("attacked_steps"),
                      "realised": t.get("realised_rate")}
    if len(clamps) > 1:
        raise SystemExit(f"eval files use different domain clamps {sorted(clamps)}; "
                         "pass --domain-clamp")
    return cells, (clamps.pop() if clamps else domain_clamp), episodes, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="host_data/results/learned_adv")
    ap.add_argument("--epsilon", type=float, default=0.30)
    ap.add_argument("--mode", default="independent", help="independent | coordinated")
    ap.add_argument("--arm", default="attack", choices=["attack", "random"],
                    help="which arm's gate to read; both use the same gate rule")
    ap.add_argument("--domain-clamp", choices=["full", "fgsm_parity"], default=None)
    ap.add_argument("--out", default="trigger_reasons.png")
    args = ap.parse_args()

    cells, clamp, episodes, skipped = load(args.results_root, args.epsilon, args.mode,
                                           args.arm, args.domain_clamp)
    for f in skipped:
        print("skipped (no reasons recorded):", f)
    if not cells:
        raise SystemExit("no gated eval JSONs with recorded trigger reasons found")
    clamp_label = {"full": "full clamp", "fgsm_parity": "FGSM-parity clamp"}.get(clamp, clamp)

    variants = [v for v in VARIANT_ORDER if any(k[0] == v for k in cells)]
    budgets = sorted({k[1] for k in cells})
    cutoffs = next(iter(cells.values()))["reasons"]["score_band_cutoffs"]

    def count(cell, rule, band):
        return cell["reasons"]["fired_by_rule_and_band"].get(rule, {}).get(band, 0)

    def fired(cell):
        return sum(count(cell, r, b) for r in RULES for b in BANDS)

    triggers = [(r, b) for r in RULES for b in BANDS
                if any(count(c, r, b) for c in cells.values())]
    absent = [b for b in BANDS if not any(t[1] == b for t in triggers)]

    cmap = LinearSegmentedColormap.from_list("seq_blue", RAMP)
    n_p = len(triggers)
    fig, axes = plt.subplots(1, n_p, figsize=(3.6 * n_p + 1.6, 0.52 * len(variants) + 2.2),
                             dpi=200, squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor(SURFACE)

    for ax, (rule, band) in zip(axes, triggers):
        grid = [[None] * len(budgets) for _ in variants]
        for i, v in enumerate(variants):
            for j, b in enumerate(budgets):
                c = cells.get((v, b))
                if c and fired(c):
                    n = count(c, rule, band)
                    grid[i][j] = (100.0 * n / fired(c), n)
        vals = [[(g[0] if g else float("nan")) for g in row] for row in grid]
        mesh = ax.pcolormesh(vals, cmap=cmap, vmin=0, vmax=100,
                             edgecolors=SURFACE, linewidth=2)
        for i, row in enumerate(grid):
            for j, g in enumerate(row):
                if g is None:
                    ax.text(j + 0.5, i + 0.5, "not run", ha="center", va="center",
                            fontsize=7, color=MUTED)
                    continue
                ink = "white" if g[0] >= 45 else INK
                ax.text(j + 0.5, i + 0.42, f"{g[0]:.0f}%", ha="center", va="center",
                        fontsize=9, color=ink)
                ax.text(j + 0.5, i + 0.72, f"{g[1]}", ha="center", va="center",
                        fontsize=6.5, color=ink, alpha=0.85)
        ax.set_facecolor(SURFACE)
        ax.invert_yaxis()
        ax.set_xticks([j + 0.5 for j in range(len(budgets))])
        ax.set_xticklabels([f"{b:.2f}" for b in budgets], fontsize=9, color=MUTED)
        ax.set_yticks([i + 0.5 for i in range(len(variants))])
        ax.set_yticklabels(variants if ax is axes[0] else [], fontsize=9, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("Timing budget (requested)", fontsize=9, color=INK)
        ax.set_title(f"{RULE_LABEL[rule]}\n{band_label(band, cutoffs)}",
                     fontsize=10, color=INK, loc="left", pad=8)

    cbar = fig.colorbar(mesh, ax=list(axes), fraction=0.025, pad=0.02)
    cbar.set_label("Share of attacked steps (%)", fontsize=9, color=INK)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=MUTED, length=0)

    n_ep = ", ".join(str(e) for e in sorted(episodes))
    fig.suptitle(f"Why the timing gate fired, {args.mode} scope, {clamp_label}  "
                 f"({args.arm} arm, ε = {args.epsilon:g}, n = {n_ep} episodes)",
                 fontsize=12, color=INK, x=0.02, ha="left", y=1.0)
    foot = "Cell: share of that run's attacked steps (count below). Panels of one cell sum to 100%."
    if absent:
        foot += "  No attack fired in any run in the " + " or ".join(
            band_label(b, cutoffs) for b in absent) + " band."
    fig.text(0.02, -0.02, foot, fontsize=7.5, color=MUTED, ha="left", wrap=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", args.out)

    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "scope", "domain_clamp", "arm", "timing_budget", "realised_rate",
                    "attacked_steps", "rule", "band", "count", "share_pct",
                    "ties_rejected", "mean_score_when_fired", "mean_threshold_when_fired"])
        for v in variants:
            for b in budgets:
                c = cells.get((v, b))
                if not c:
                    continue
                r, tot = c["reasons"], fired(c)
                for rule in RULES:
                    for band in BANDS:
                        n = count(c, rule, band)
                        w.writerow([v, args.mode, clamp, args.arm, b, c["realised"], c["attacked"],
                                    rule, band, n, f"{100 * n / tot:.2f}" if tot else "",
                                    r["decision_outcomes"].get("skipped_tie_rejected"),
                                    r.get("mean_score_when_fired"),
                                    r.get("mean_threshold_when_fired")])
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
