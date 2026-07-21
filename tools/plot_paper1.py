"""
Paper-1 figure generator — MADDPG routing-variant architecture study.

CANONICAL result set = reward_fix (all 7 variants trained at 2x hotspot with the
shared flow reward, mean_util_weight = 0.1).  Produces the 10 headline figures
(F1-F10) of the rewritten results section:

  F1  training reward (7 variants, rolling mean over raw signal)
  F2  training packet loss
  F3  end-to-end PDR vs offered load under hotspot demand (7 variants + EVPN-SP)
  F4  PDR gap to EVPN-SP vs load (hotspot)
  F5  baseline envelope vs load: CC-Simple policy vs greedy / random / SP / worst
  F6  paired per-episode significance: policy - random (delta p95, delta PDR)
  F7  per-variant damage-ceiling bars at 2x hotspot (architecture ordering)
  F8  OOD failure-severity sweep: PDR vs n random link failures + gap-to-greedy
  F9  GNN paired delta (GNN - non-GNN), intact vs dual-link failure
  F10 training-regime / reward ablation (1x w0.8 -> 2x w0.8 -> 2x w0.1)

NOTE: sweep_baselines/load_*/ policy rows are NOT used anywhere — those runs
loaded final (not best) checkpoints via a broken models symlink; only their
checkpoint-independent oracle rules (greedy/random/sp/worst) are read.

Also prints the exact statistics quoted in the paper (Wilcoxon, sign counts,
effect sizes) to stdout.

Usage (server, inside the maddpg-exp container with host_data -> /workspace/data):
    RESULTS_ROOT=/workspace/data/results python tools/plot_paper1.py
Locally:
    python tools/plot_paper1.py            # ROOT defaults to host_data/results
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.environ.get("RESULTS_ROOT", os.path.join("host_data", "results"))
FIG_DIR = os.path.join(ROOT, "reward_fix", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Variant styling (family colour; GNN dashed) ─────────────────────────────
STYLE = {
    "CC-Simple":       dict(color="#1f77b4", ls="-",  marker="o"),
    "CC-Simple-GNN":   dict(color="#1f77b4", ls="--", marker="o"),
    "CC-Duelling":     dict(color="#17becf", ls="-",  marker="s"),
    "CC-Duelling-GNN": dict(color="#17becf", ls="--", marker="s"),
    "LC-Duelling":     dict(color="#2ca02c", ls="-",  marker="^"),
    "LC-Duelling-GNN": dict(color="#2ca02c", ls="--", marker="^"),
    "LC-Simple":       dict(color="#ff7f0e", ls="-",  marker="D"),
    "EVPN_SP":         dict(color="black",   ls=":",  marker="x"),
}
# Fixed routing-rule styling (greedy is a myopic heuristic, NOT a flow optimum)
RULE_STYLE = {
    "greedy": dict(color="dimgray",  ls="--", marker="v", label="Greedy (least-util)"),
    "random": dict(color="#9467bd",  ls="-",  marker="P", label="Random spreading"),
    "sp":     dict(color="black",    ls=":",  marker="x", label="EVPN-SP"),
    "worst":  dict(color="#bbbbbb",  ls="-",  marker=".", label="Worst (adversarial)"),
}
VARIANTS = ["CC-Simple", "CC-Duelling", "CC-Simple-GNN", "CC-Duelling-GNN",
            "LC-Simple", "LC-Duelling", "LC-Duelling-GNN"]
GNN_PAIRS = [("CC-Simple", "CC-Simple-GNN"),
             ("CC-Duelling", "CC-Duelling-GNN"),
             ("LC-Duelling", "LC-Duelling-GNN")]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 9,
})

PDR = "mean_end_to_end_pdr"          # delivered / injected (headline metric)


def jload(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return json.load(f)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def rolling_mean(y, window=41):
    """Centred rolling mean; shrinks window at the edges to avoid NaNs."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = np.empty(n)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


# ═════ F1 / F2 — training convergence ════════════════════════════════════════
def f1_f2():
    tr = jload("reward_fix", "phase1_training_results.json")
    for name, key, ylab, loc in [
            ("F1_training_reward", "rewards", "Episode reward (shared flow reward)", "lower right"),
            ("F2_training_pkt_loss", "pkt_losses", "Packet loss (%)", "upper right")]:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for v in VARIANTS:
            raw = np.asarray(tr[v][key], dtype=float)
            ax.plot(raw, color=STYLE[v]["color"], alpha=0.10, lw=0.6)
            ax.plot(rolling_mean(raw), label=v, lw=1.8,
                    color=STYLE[v]["color"], ls=STYLE[v]["ls"])
        ax.set_xlabel("Training episode")
        ax.set_ylabel(ylab)
        ax.legend(ncol=2, loc=loc)
        save(fig, name)


# ═════ F3 / F4 — hotspot load sweep, variants vs EVPN-SP ═════════════════════
def f3_f4():
    sw = jload("reward_fix", "phase2_hotspot_sweep_results.json")
    loads = sw["meta"]["loads"]

    def series(m):
        return [sw["methods"][m][f"load_{l:.2f}"][PDR] for l in loads]

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for m in VARIANTS + ["EVPN_SP"]:
        ax.plot(loads, series(m), label=m.replace("EVPN_SP", "EVPN-SP"),
                lw=2.2 if m in ("CC-Simple", "EVPN_SP") else 1.4,
                ms=4, **STYLE[m])
    ax.set_xlabel("Offered load factor (hotspot demand)")
    ax.set_ylabel("End-to-end PDR (%)")
    ax.legend(ncol=2, loc="lower left")
    save(fig, "F3_pdr_vs_load_hotspot")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    sp = np.array(series("EVPN_SP"))
    for m in VARIANTS:
        ax.plot(loads, np.array(series(m)) - sp, label=m, lw=1.6, ms=4, **STYLE[m])
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Offered load factor (hotspot demand)")
    ax.set_ylabel("PDR gap to EVPN-SP (pp)")
    ax.legend(ncol=2, loc="upper left")
    save(fig, "F4_gap_vs_sp_hotspot")
    cc = np.array(series("CC-Simple"))
    print("  [F3] CC-Simple over SP (pp):",
          {f"{l:g}x": round(g, 1) for l, g in zip(loads, cc - sp)})


# ═════ F5 — baseline envelope (oracle rules) + CC-Simple policy ══════════════
def f5():
    sw = jload("reward_fix", "phase2_hotspot_sweep_results.json")
    loads = sw["meta"]["loads"]
    pol = [sw["methods"]["CC-Simple"][f"load_{l:.2f}"][PDR] for l in loads]

    rules = {r: [] for r in ("greedy", "random", "sp", "worst")}
    for l in loads:
        d = jload("reward_fix", "sweep_baselines", f"load_{l:.2f}", "damage_ceiling.json")
        for r in rules:
            rules[r].append(d[r][PDR])

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for r in ("greedy", "random", "sp", "worst"):
        ax.plot(loads, rules[r], lw=1.5, ms=4, **RULE_STYLE[r])
    ax.plot(loads, pol, color=STYLE["CC-Simple"]["color"], marker="o", ms=5,
            lw=2.4, label="CC-Simple (learned)")
    ax.annotate("", xy=(3.0, rules["greedy"][-1]), xytext=(3.0, rules["random"][-1]),
                arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.2))
    ax.text(2.93, (rules["greedy"][-1] + rules["random"][-1]) / 2,
            "adaptivity\nheadroom", ha="right", va="center", fontsize=8, color="dimgray")
    ax.set_xlabel("Offered load factor (hotspot demand)")
    ax.set_ylabel("End-to-end PDR (%)")
    ax.legend(loc="lower left")
    save(fig, "F5_baseline_envelope")
    print("  [F5] random->greedy gap:",
          {f"{l:g}x": round(g - r, 1) for l, g, r in zip(loads, rules["greedy"], rules["random"])})


# ═════ F6 — paired per-episode significance (policy vs random) ═══════════════
def f6():
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.7))
    stats_out = {}
    rng = np.random.default_rng(7)
    for j, (metric, key, ylab) in enumerate([
            ("p95", "p95_series", r"$\Delta$ 95th-pct delay (steps)"),
            ("pdr", "pdr_series", r"$\Delta$ end-to-end PDR (pp)")]):
        ax = axes[j]
        for i, load in enumerate(("2.00", "3.00")):
            d = jload("sigrun", f"load_{load}", "damage_ceiling.json")
            dd = np.array(d["policy"][key]) - np.array(d["random"][key])
            x = np.full(len(dd), float(i)) + rng.uniform(-0.13, 0.13, len(dd))
            ax.scatter(x, dd, s=9, alpha=0.45, color="#1f77b4", zorder=3)
            ax.boxplot([dd], positions=[i], widths=0.42, showfliers=False,
                       medianprops=dict(color="black"))
            try:
                p = float(stats.wilcoxon(dd).pvalue)
            except ValueError:
                p = float("nan")
            neg, pos, zero = int((dd < 0).sum()), int((dd > 0).sum()), int((dd == 0).sum())
            dz = float(np.mean(dd) / np.std(dd, ddof=1))
            stats_out[(metric, load)] = dict(mean=float(np.mean(dd)), p=p,
                                             neg=neg, pos=pos, zero=zero, dz=dz)
        ax.axhline(0, color="black", lw=1)
        # p-value annotations after axes limits settle
        for i, load in enumerate(("2.00", "3.00")):
            s = stats_out[(metric, load)]
            ax.text(i, ax.get_ylim()[1] * 0.97, f"p={s['p']:.0e}",
                    ha="center", va="top", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([r"$2\times$ load", r"$3\times$ load"])
        ax.set_ylabel(ylab)
        ax.set_title(r"policy $-$ random (paired, $n{=}60$)", fontsize=10)
    fig.tight_layout()
    save(fig, "F6_paired_significance")
    for k, v in stats_out.items():
        print(f"  [F6] {k}: mean={v['mean']:+.3f}  p={v['p']:.2e}  dz={v['dz']:+.2f}  "
              f"neg/pos/zero={v['neg']}/{v['pos']}/{v['zero']}")


# ═════ F7 — per-variant damage ceiling at 2x hotspot (with 95% CIs) ══════════
def f7():
    # ceil2x_v2 reruns ceil2x with per-episode series so we can show a 95% CI
    # (normal approx, n=20 episodes) on each variant's paired PDR estimate.
    n_eps = 20
    z = 1.96
    vals, cis = {}, {}
    for v in VARIANTS:
        d = jload("ceil2x_v2", v, "damage_ceiling.json")["policy"]
        vals[v] = d[PDR]
        cis[v] = z * d["std_end_to_end_pdr"] / np.sqrt(n_eps)
    base = jload("ceil2x_v2", "CC-Simple", "damage_ceiling.json")   # rules identical across dirs
    order = sorted(vals, key=vals.get)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    y = np.arange(len(order))
    ax.barh(y, [vals[v] for v in order],
            xerr=[cis[v] for v in order], capsize=3,
            error_kw=dict(lw=1.2, ecolor="black"),
            color=[STYLE[v]["color"] for v in order],
            hatch=["//" if v.endswith("GNN") else "" for v in order],
            edgecolor="white")
    for i, v in enumerate(order):
        ax.text(vals[v] + cis[v] + 0.3, i, f"{vals[v]:.1f}", va="center", fontsize=9)
    # Rule labels anchored just above the axes (axes-fraction y), so they never
    # collide with a bar's value annotation regardless of where the CI lands.
    for r in ("sp", "random", "greedy"):
        ax.axvline(base[r][PDR], color=RULE_STYLE[r]["color"], ls=RULE_STYLE[r]["ls"], lw=1.4)
        ax.text(base[r][PDR], 1.01, RULE_STYLE[r]["label"].split(" (")[0],
                transform=ax.get_xaxis_transform(),
                rotation=90, va="bottom", ha="center", fontsize=8,
                color=RULE_STYLE[r]["color"])
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlim(60, 100)
    ax.set_xlabel(r"End-to-end PDR (%) at $2\times$ hotspot (paired episodes, 95% CI)")
    save(fig, "F7_variant_ceiling_2x")
    print("  [F7] 95% CI half-widths (pp):",
          {v: round(cis[v], 2) for v in order})


# ═════ F8 — OOD failure-severity sweep ═══════════════════════════════════════
def f8():
    ns = [0, 1, 2, 4, 6, 8]
    data = {r: [] for r in ("policy", "greedy", "random", "sp", "worst")}
    for n in ns:
        d = jload("failsev", f"n_{n}", "damage_ceiling.json")
        for r in data:
            data[r].append(d[r][PDR])

    fig, (ax, axg) = plt.subplots(1, 2, figsize=(8.2, 3.8),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    for r in ("greedy", "random", "sp", "worst"):
        ax.plot(ns, data[r], lw=1.5, ms=4.5, **RULE_STYLE[r])
    ax.plot(ns, data["policy"], color=STYLE["CC-Simple"]["color"], marker="o",
            ms=5, lw=2.4, label="CC-Simple (learned)")
    ax.axvline(2.15, color="#2ca02c", ls="--", lw=1, alpha=0.8)
    ax.text(2.35, 97, "crossover\n$n\\approx 2$", fontsize=8, color="#2ca02c", va="top")
    ax.set_xlabel("Random link failures per episode ($n$)")
    ax.set_ylabel("End-to-end PDR (%)")
    ax.legend(loc="lower left", fontsize=8)

    gaps = np.array(data["greedy"]) - np.array(data["policy"])
    colors = ["#c44e52" if g < 0 else "dimgray" for g in gaps]
    axg.bar(range(len(ns)), gaps, color=colors, width=0.6)
    for i, g in enumerate(gaps):
        axg.text(i, g + (0.5 if g >= 0 else -0.5), f"{g:+.1f}",
                 ha="center", va="bottom" if g >= 0 else "top", fontsize=8)
    axg.axhline(0, color="black", lw=1)
    axg.set_xticks(range(len(ns)))
    axg.set_xticklabels(ns)
    axg.set_xlabel("Random link failures ($n$)")
    axg.set_ylabel(r"Greedy $-$ policy (pp)")
    axg.set_ylim(float(min(gaps)) - 4, float(max(gaps)) + 4)
    fig.tight_layout()
    save(fig, "F8_failure_severity")
    print("  [F8] policy drop 0->8:", round(data["policy"][0] - data["policy"][-1], 1),
          "| random drop:", round(data["random"][0] - data["random"][-1], 1))


# ═════ F9 — GNN paired delta (intact vs dual-link failure), paired 95% CI ════
def f9():
    # ceil2x_v2 / ceil2x_fail_v2 share an identical per-episode traffic seed
    # across variant directories (verified: 'random'/'sp' pdr_series are
    # byte-identical across variants), so the GNN - non-GNN delta can be
    # computed per-episode and given a proper paired CI, not just a point value.
    def paired_delta(dirname, base, gnn):
        a = np.array(jload(dirname, base, "damage_ceiling.json")["policy"]["pdr_series"])
        b = np.array(jload(dirname, gnn, "damage_ceiling.json")["policy"]["pdr_series"])
        d = b - a
        return float(d.mean()), 1.96 * float(d.std(ddof=1)) / np.sqrt(len(d))

    labels = [b for b, _ in GNN_PAIRS]
    intact = [paired_delta("ceil2x_v2", b, g) for b, g in GNN_PAIRS]
    failed = [paired_delta("ceil2x_fail_v2", b, g) for b, g in GNN_PAIRS]
    intact_m, intact_ci = zip(*intact)
    failed_m, failed_ci = zip(*failed)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    ax.bar(x - 0.18, intact_m, 0.36, yerr=intact_ci, capsize=3,
           error_kw=dict(lw=1.1, ecolor="black"),
           label="intact topology", color="#4c72b0")
    ax.bar(x + 0.18, failed_m, 0.36, yerr=failed_ci, capsize=3,
           error_kw=dict(lw=1.1, ecolor="black"),
           label="dual-link failure", color="#c44e52")
    for xi, v, e in list(zip(x - 0.18, intact_m, intact_ci)) + list(zip(x + 0.18, failed_m, failed_ci)):
        off = 0.25 if v >= 0 else -0.25
        ax.text(xi, v + e + off if v >= 0 else v - e + off, f"{v:+.1f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", lw=1)
    lo = min(min(intact_m[i] - intact_ci[i] for i in range(3)),
             min(failed_m[i] - failed_ci[i] for i in range(3)))
    hi = max(max(intact_m[i] + intact_ci[i] for i in range(3)),
             max(failed_m[i] + failed_ci[i] for i in range(3)))
    ax.set_ylim(lo - 1.0, hi + 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PDR change from adding GNN (pp)")
    ax.legend()
    save(fig, "F9_gnn_paired_delta")
    print("  [F9] intact:", [(round(m, 2), round(c, 2)) for m, c in intact],
          "| failed:", [(round(m, 2), round(c, 2)) for m, c in failed])
    print("  [F9] CI excludes zero?",
          {lab: (abs(m) > c) for lab, (m, c) in zip(labels, intact)})


# ═════ F10 — training-regime / reward ablation (CC-Simple), unified protocol ═
def f10():
    # All three bars now come from the SAME paired 20-episode ceiling protocol
    # (previously main/stress used a 30-ep sweep and reward_fix a paired
    # ceiling — different eval protocols are not comparable point-for-point).
    n_eps = 20
    z = 1.96
    paths = [("main_CC-Simple", "trained $1\\times$\n$w_u{=}0.8$", "#aec7e8", "//"),
             ("stress_CC-Simple", "trained $2\\times$\n$w_u{=}0.8$", "#6baed6", ""),
             ("CC-Simple", "trained $2\\times$\n$w_u{=}0.1$", "#1f77b4", "")]
    vals, cis = [], []
    for key, *_ in paths:
        dirname = "ablation_ceil" if key != "CC-Simple" else "ceil2x_v2"
        d = jload(dirname, key, "damage_ceiling.json")["policy"]
        vals.append(d[PDR])
        cis.append(z * d["std_end_to_end_pdr"] / np.sqrt(n_eps))
    rf = jload("ceil2x_v2", "CC-Simple", "damage_ceiling.json")

    labels = [p[1] for p in paths]
    colors = [p[2] for p in paths]
    hatches = [p[3] for p in paths]
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    ax.bar(range(3), vals, 0.55, yerr=cis, capsize=4,
           error_kw=dict(lw=1.2, ecolor="black"),
           color=colors, hatch=hatches, edgecolor="white")
    for i, (v, c) in enumerate(zip(vals, cis)):
        ax.text(i, v + c + 0.3, f"{v:.1f}", ha="center", fontsize=10)
    for r in ("sp", "random"):
        ax.axhline(rf[r][PDR], color=RULE_STYLE[r]["color"], ls=RULE_STYLE[r]["ls"], lw=1.4)
        ax.text(2.55, rf[r][PDR] + 0.2, RULE_STYLE[r]["label"].split(" (")[0],
                fontsize=8, va="bottom", ha="right", color=RULE_STYLE[r]["color"])
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(65, 95)
    ax.set_ylabel("CC-Simple end-to-end PDR (%)\nat $2\\times$ hotspot (95% CI)")
    save(fig, "F10_ablation_regime")
    print(f"  [F10] bars (paired protocol): "
          f"main={vals[0]:.2f}±{cis[0]:.2f} stress={vals[1]:.2f}±{cis[1]:.2f} "
          f"rf={vals[2]:.2f}±{cis[2]:.2f}")


# ═════ F11 — canonical set at UNIFORM demand (reviewer M3) ═══════════════════
def f11():
    u = jload("unisweep", "phase2_hotspot_sweep_results.json")
    loads = u["meta"]["loads"]

    def series(m):
        return [u["methods"][m][f"load_{l:.2f}"][PDR] for l in loads]

    fig, (ax, axg) = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for m in VARIANTS + ["EVPN_SP"]:
        ax.plot(loads, series(m), label=m.replace("EVPN_SP", "EVPN-SP"),
                lw=2.2 if m in ("CC-Simple", "EVPN_SP") else 1.3,
                ms=4, **STYLE[m])
    ax.set_xlabel("Offered load factor (uniform demand)")
    ax.set_ylabel("End-to-end PDR (%)")
    ax.legend(ncol=2, loc="lower left", fontsize=7.5)

    sp = np.array(series("EVPN_SP"))
    worst_gap, best_gap = 0.0, -99.0
    for m in VARIANTS:
        gap = np.array(series(m)) - sp
        axg.plot(loads, gap, lw=1.6, ms=4, **STYLE[m])
        worst_gap = min(worst_gap, gap.min())
        best_gap = max(best_gap, gap.max())
    axg.axhline(0, color="black", lw=1)
    axg.set_xlabel("Offered load factor (uniform demand)")
    axg.set_ylabel("PDR gap to EVPN-SP (pp)")
    fig.tight_layout()
    save(fig, "F11_uniform_sweep")
    print(f"  [F11] stress-trained set at UNIFORM load: gap to SP in "
          f"[{worst_gap:.1f}, {best_gap:.1f}] pp across all variants/loads")


if __name__ == "__main__":
    print(f"ROOT = {ROOT}")
    for fn in (f1_f2, f3_f4, f5, f6, f7, f8, f9, f10, f11):
        print(f"── {fn.__name__} ──")
        fn()
    print("done.")
