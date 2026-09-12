#!/usr/bin/env python
"""Plot the adversarial gap against the timing budget and against the flip rate.

Reads the paired-eval JSONs written by tools/train_adversary.py --eval-only and
draws two panels:

  left   gap (random - attack, in pp) with its 95% CI, for every timing budget
         plus the ungated run as the 100% reference.
  right  the same gap against the fraction of routing decisions the attack
         actually flipped - the panel that separates "changed the routing" from
         "did damage".

    python tools/plot_timing_sweep.py --results-root host_data/results/learned_adv \
        --epsilon 0.30 --out host_data/results/learned_adv/timing_sweep_gap.png

Design notes (dataviz skill):
  * LEFT x is CATEGORICAL, so the ungated 100% point sits beside 0.10-0.25
    without a broken or log axis.
  * tick labels carry the REALISED attack rate, which is the honest x.
  * colour is fixed per variant (never by rank), taken in slot order from the
    reference categorical palette; filled marker = CI excludes 0.
  * RIGHT is drawn as connected TRAJECTORIES, not a point cloud. A 7-series
    scatter would be judged on the all-pairs pairlist, where only the first
    three palette slots clear the floors (slot 4 puts yellow beside orange at
    normal-vision dE 13.7, under the hard floor of 15). Joining each variant's
    five budgets into one labelled path keeps identity on position + direct
    label, with colour reinforcing the key the left panel already taught.
  * a CSV table view is written next to the PNG: three of the light-mode slots
    sit below 3:1 contrast, so the numbers must be reachable without hue.
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

# Reference categorical palette, light mode, in slot order. Assigned to variants
# by NAME so a variant keeps its colour no matter which subset is plotted.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
VARIANT_ORDER = ["CC-Simple", "CC-Duelling", "CC-Simple-GNN", "CC-Duelling-GNN",
                 "LC-Simple", "LC-Duelling", "LC-Duelling-GNN"]
COLOR = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(VARIANT_ORDER)}

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"


def load(results_root: str, epsilon: float, mode: str):
    """Return {variant: {budget_or_None: row}} for one epsilon and eval mode."""
    out: dict = {}
    pattern = os.path.join(results_root, "*", "**", f"learned_adv_eval_{mode}.json")
    files = glob.glob(pattern, recursive=True) + glob.glob(
        os.path.join(results_root, "*", f"learned_adv_eval_{mode}.json"))
    for f in sorted(set(files)):
        if "smoke" in f:
            continue
        d = json.load(open(f, encoding="utf-8"))
        if abs(float(d.get("epsilon", -1)) - epsilon) > 1e-9:
            continue
        o, t = d.get("outcomes", {}), (d.get("timing") or {}).get("attack", {})
        dec = d.get("decisions") or {}
        lo, hi = (o.get("adversarial_gap_ci95") or [None, None])
        out.setdefault(d["variant"], {})[d.get("timing_budget")] = {
            "gap": o.get("adversarial_gap_pp"), "lo": lo, "hi": hi,
            "realised": t.get("realised_rate", 1.0 if not t.get("gated", False) else None),
            "atk_flip": dec.get("attack_action_flip_rate"),
            "rnd_flip": dec.get("random_action_flip_rate"),
            "episodes": d.get("episodes"),
        }
    return out


def place_end_labels(ax, ends, pad_frac=0.018, max_shift_frac=None):
    """Direct end-labels, nudged just enough to clear one line of text.

    `ends` is [[y, x, variant, colour], ...]. The nudge is ~one label height,
    NOT a big fraction of the span: an over-large gap drags a label away from
    the line it names, which misreads worse than no label. When
    `max_shift_frac` is set, a label that still cannot sit near its own series
    is dropped - the legend and the CSV table view carry those.
    """
    ends = sorted(ends)
    span = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
    min_gap = pad_frac * span
    max_shift = None if max_shift_frac is None else max_shift_frac * span
    prev = None
    for e in ends:
        e.append(e[0])                                   # remember the true y
        if prev is not None and e[0] - prev < min_gap:
            e[0] = prev + min_gap
        prev = e[0]
    for y, x, variant, c, true_y in ends:
        if max_shift is not None and abs(y - true_y) > max_shift:
            continue
        ax.annotate(variant, xy=(x, y), color=c, fontsize=8,
                    va="center", ha="left", annotation_clip=False)


def style_axes(ax):
    """Recessive chrome: horizontal grid only, no top/right spines."""
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(axis="y", labelsize=9, colors=MUTED, length=0)
    ax.tick_params(axis="x", labelsize=9, colors=MUTED, length=0)
    ax.axhline(0, color=BASELINE, linewidth=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="host_data/results/learned_adv")
    ap.add_argument("--epsilon", type=float, default=0.30)
    ap.add_argument("--mode", default="independent",
                    help="which eval files to read: independent | coordinated")
    ap.add_argument("--out", default="timing_sweep_gap.png")
    args = ap.parse_args()

    data = load(args.results_root, args.epsilon, args.mode)
    if not data:
        raise SystemExit(f"no {args.mode} eval JSONs with epsilon={args.epsilon} "
                         f"under {args.results_root}")

    budgets = sorted({b for v in data.values() for b in v if b is not None})
    columns = budgets + [None]                      # None = ungated, the 100% point
    xs = list(range(len(columns)))

    # tick label = requested budget + the realised rate averaged over variants
    labels = []
    for b in columns:
        rs = [v[b]["realised"] for v in data.values() if b in v and v[b]["realised"]]
        mean_r = f"{100 * sum(rs) / len(rs):.0f}%" if rs else "?"
        labels.append(f"{b:g}\nactual {mean_r}" if b is not None else f"none\nactual {mean_r}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    style_axes(ax2)

    # ── left panel: gap vs requested budget ──────────────────────────────────
    ends = []
    for variant in VARIANT_ORDER:
        if variant not in data:
            continue
        series = data[variant]
        px = [i for i, b in enumerate(columns) if b in series and series[b]["gap"] is not None]
        py = [series[columns[i]]["gap"] for i in px]
        if not px:
            continue
        err = [[series[columns[i]]["gap"] - (series[columns[i]]["lo"] or series[columns[i]]["gap"])
                for i in px],
               [(series[columns[i]]["hi"] or series[columns[i]]["gap"]) - series[columns[i]]["gap"]
                for i in px]]
        c = COLOR[variant]
        ax.errorbar(px, py, yerr=err, color=c, linewidth=2.0, elinewidth=1.2,
                    capsize=3, alpha=0.95, zorder=3, label=variant)
        # filled marker = the CI excludes zero (a real effect); hollow = not significant
        for i in px:
            r = series[columns[i]]
            sig = r["lo"] is not None and not (r["lo"] <= 0 <= r["hi"])
            ax.plot(i, r["gap"], marker="o", markersize=7, zorder=4, color=c,
                    markerfacecolor=(c if sig else SURFACE),
                    markeredgecolor=c, markeredgewidth=2)
        ends.append([py[-1], xs[-1] + 0.08, variant, c])

    place_end_labels(ax, ends, pad_frac=0.018, max_shift_frac=0.05)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9, color=MUTED)
    ax.set_xlim(-0.3, len(columns) - 0.35)
    ax.set_xlabel("Timing budget (requested, with mean realised attack rate)",
                  fontsize=10, color=INK, labelpad=10)
    ax.set_ylabel("Adversarial gap (pp)   random − attack", fontsize=10, color=INK)
    ax.set_title("Gap grows with the budget\nfilled marker = 95% CI excludes zero",
                 fontsize=11, color=INK, loc="left", pad=12)

    # ── right panel: gap vs decisions actually flipped ───────────────────────
    # One connected path per variant, walking 0.10 -> ungated. Within a variant
    # flips and damage rise together; ACROSS variants the same flip rate lands at
    # wildly different damage, which is the whole point of the panel.
    ends2, flips_all = [], []
    for variant in VARIANT_ORDER:
        if variant not in data:
            continue
        series = data[variant]
        pts = [(100 * series[b]["atk_flip"], series[b]["gap"]) for b in columns
               if b in series and series[b]["gap"] is not None
               and series[b].get("atk_flip") is not None]
        if not pts:
            continue
        c = COLOR[variant]
        fx, fy = [p[0] for p in pts], [p[1] for p in pts]
        ax2.plot(fx, fy, color=c, linewidth=2.0, alpha=0.95, zorder=3,
                 marker="o", markersize=6, markerfacecolor=c, markeredgecolor=c)
        ends2.append([fy[-1], fx[-1] * 1.09, variant, c])
        flips_all.extend(fx)

    # Flip rates span ~3 decades (0.018% to 17%), so a linear x crushes four of
    # the seven variants against the origin. Log x is the honest fix: it is a
    # rate, it never reaches zero here, and the panel's argument is the VERTICAL
    # spread at matched x, which any monotone x-scale preserves.
    ax2.set_xscale("log")
    ax2.xaxis.grid(True, color=GRID, linewidth=0.8)

    # No max_shift here: this panel has no error bars to fall back on, so every
    # variant must keep its label even when the flat ones pile up around zero.
    place_end_labels(ax2, ends2, pad_frac=0.045)

    lo_f, hi_f = min(flips_all), max(flips_all)
    ax2.set_xlim(lo_f * 0.55, hi_f * 3.4)
    ax2.set_xticks([0.01, 0.1, 1.0, 10.0])
    ax2.set_xticklabels(["0.01%", "0.1%", "1%", "10%"])
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel("Routing decisions flipped by the attack (log scale)",
                   fontsize=10, color=INK, labelpad=10)
    ax2.set_ylabel("Adversarial gap (pp)   random − attack", fontsize=10, color=INK)
    ax2.set_title("But flipping decisions does not buy damage\n"
                  "paths walk 0.10 → ungated; equal flips, opposite outcomes",
                  fontsize=11, color=INK, loc="left", pad=12)

    n = next(iter(next(iter(data.values())).values()))["episodes"]
    fig.suptitle(f"Learned adversary vs victim architecture  "
                 f"(ε = {args.epsilon:g}, n = {n} paired episodes)",
                 fontsize=12, color=INK, x=0.035, ha="left", y=1.02)
    handles, lbls = ax.get_legend_handles_labels()
    fig.legend(handles, lbls, frameon=False, fontsize=8, ncol=7,
               loc="lower center", bbox_to_anchor=(0.5, -0.06), labelcolor=INK)

    fig.tight_layout(w_pad=4.0)
    fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", args.out)

    # table view — the numbers must be readable without relying on colour
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "timing_budget", "realised_rate", "gap_pp",
                    "ci95_lo", "ci95_hi", "significant",
                    "attack_flip_pct", "random_flip_pct"])
        for variant in VARIANT_ORDER:
            for b in columns:
                r = data.get(variant, {}).get(b)
                if not r:
                    continue
                sig = r["lo"] is not None and not (r["lo"] <= 0 <= r["hi"])
                w.writerow([variant, "none" if b is None else b,
                            f"{r['realised']:.3f}" if r["realised"] else "",
                            f"{r['gap']:+.3f}", f"{r['lo']:+.3f}" if r["lo"] is not None else "",
                            f"{r['hi']:+.3f}" if r["hi"] is not None else "", int(sig),
                            f"{100 * r['atk_flip']:.2f}" if r.get("atk_flip") is not None else "",
                            f"{100 * r['rnd_flip']:.2f}" if r.get("rnd_flip") is not None else ""])
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
