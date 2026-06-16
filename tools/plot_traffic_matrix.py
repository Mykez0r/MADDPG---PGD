#!/usr/bin/env python3
"""
tools/plot_traffic_matrix.py
-----------------------------
Generates publication-quality figures of the traffic demand model used
in the MADDPG experiments.

Outputs (saved to tools/figures/):
  traffic_matrix_heatmap.pdf / .png  - 65x65 empirical demand heatmap
  traffic_flow_stats.pdf / .png      - flow count, packet dist, priority pie
  traffic_source_load.pdf / .png     - per-node outgoing flow load

Usage (local machine, no Docker needed):
  pip install matplotlib networkx numpy
  python tools/plot_traffic_matrix.py
"""

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_NODES = 65
HOSTS   = [f"H{i}" for i in range(N_NODES)]


# ---------------------------------------------------------------------------
# Traffic generator  (mirrors FlowManager.generate_random_flows exactly)
# ---------------------------------------------------------------------------

def generate_flows(n_flows: int = 500, rng: random.Random = None) -> list:
    """Generate flows using the same logic as the experiment environment."""
    if rng is None:
        rng = random.Random(RANDOM_SEED)
    flows = []
    for _ in range(n_flows):
        src = rng.choice(HOSTS)
        dst = rng.choice([h for h in HOSTS if h != src])
        flows.append({
            "src":      src,
            "dst":      dst,
            "packets":  rng.randint(5, 20),
            "priority": rng.choice(["high", "medium", "low"]),
        })
    return flows


def build_demand_matrix(flows: list) -> np.ndarray:
    """Build a N_NODES x N_NODES demand matrix (total packets per pair)."""
    idx = {h: i for i, h in enumerate(HOSTS)}
    M   = np.zeros((N_NODES, N_NODES), dtype=float)
    for f in flows:
        M[idx[f["src"]], idx[f["dst"]]] += f["packets"]
    return M


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, stem: str):
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Demand heatmap
# ---------------------------------------------------------------------------

def plot_traffic_heatmap(M: np.ndarray):
    print("Plotting traffic matrix heatmap ...")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(M, cmap="YlOrRd", interpolation="nearest", aspect="auto")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Total packets demanded", fontsize=10)

    # Tick labels: show every 10th node to avoid clutter
    tick_positions = list(range(0, N_NODES, 10))
    tick_labels    = [f"H{i}" for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)

    ax.set_xlabel("Destination node", fontsize=11)
    ax.set_ylabel("Source node", fontsize=11)
    ax.set_title(
        "Empirical Traffic Demand Matrix\n"
        f"(500 sampled flows, uniform random model, seed={RANDOM_SEED})",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, "traffic_matrix_heatmap")


# ---------------------------------------------------------------------------
# Figure 2: Flow statistics (3-panel)
# ---------------------------------------------------------------------------

def plot_flow_statistics(flows: list):
    print("Plotting flow statistics ...")

    packets   = [f["packets"]  for f in flows]
    priorities = [f["priority"] for f in flows]

    # Count flows per source
    src_counts = {h: 0 for h in HOSTS}
    for f in flows:
        src_counts[f["src"]] += 1
    src_values = list(src_counts.values())

    priority_counts = {
        "high":   priorities.count("high"),
        "medium": priorities.count("medium"),
        "low":    priorities.count("low"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- Panel 1: Packet count distribution ---
    ax = axes[0]
    ax.hist(packets, bins=range(4, 22), color="#2980b9",
            edgecolor="white", alpha=0.85, align="left")
    ax.set_xlabel("Packets per flow", fontsize=11)
    ax.set_ylabel("Number of flows", fontsize=11)
    ax.set_title("Packet Count Distribution", fontsize=12, fontweight="bold")
    ax.axvline(np.mean(packets), color="#e74c3c", linestyle="--",
               linewidth=1.5, label=f"Mean = {np.mean(packets):.1f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)

    # --- Panel 2: Priority distribution (pie) ---
    ax = axes[1]
    colours = {"high": "#e74c3c", "medium": "#f39c12", "low": "#3498db"}
    wedge_colours = [colours[p] for p in priority_counts]
    wedges, texts, autotexts = ax.pie(
        list(priority_counts.values()),
        labels=[p.capitalize() for p in priority_counts],
        colors=wedge_colours,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title("Flow Priority Distribution", fontsize=12, fontweight="bold")

    # --- Panel 3: Flows per source node ---
    ax = axes[2]
    sorted_counts = sorted(src_values, reverse=True)
    ax.bar(range(N_NODES), sorted_counts, color="#27ae60", alpha=0.8, width=0.9)
    ax.set_xlabel("Source node (sorted by load)", fontsize=11)
    ax.set_ylabel("Number of outgoing flows", fontsize=11)
    ax.set_title("Outgoing Flow Load per Source Node", fontsize=12, fontweight="bold")
    ax.axhline(np.mean(src_values), color="#e74c3c", linestyle="--",
               linewidth=1.5, label=f"Mean = {np.mean(src_values):.1f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)

    fig.suptitle(
        "Traffic Model Statistics  —  Uniform Random Demand (FlowManager)",
        fontsize=11, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    _save(fig, "traffic_flow_stats")


# ---------------------------------------------------------------------------
# Figure 3: Source-destination load heatmap (aggregated per tier)
# ---------------------------------------------------------------------------

def plot_tier_demand(M: np.ndarray):
    """Show aggregated demand between tiers (core/dist/access)."""
    print("Plotting tier-aggregated demand ...")

    # Tier boundaries (must match network_environment.py)
    n_nodes   = N_NODES
    core_end  = max(3, int(0.1 * n_nodes))         # 0..6  -> core
    dist_end  = core_end + int(0.3 * n_nodes)       # 7..25 -> distribution
    # 26..64 -> access

    tier_ranges = {
        "Core":         range(0,        core_end),
        "Distribution": range(core_end, dist_end),
        "Access":       range(dist_end, n_nodes),
    }
    tier_names = list(tier_ranges.keys())
    n_tiers    = len(tier_names)

    T = np.zeros((n_tiers, n_tiers))
    for i, t_src in enumerate(tier_names):
        for j, t_dst in enumerate(tier_names):
            T[i, j] = M[np.ix_(
                list(tier_ranges[t_src]),
                list(tier_ranges[t_dst])
            )].sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(T, cmap="Blues", interpolation="nearest")

    # Annotate cells
    for i in range(n_tiers):
        for j in range(n_tiers):
            ax.text(j, i, f"{T[i,j]:.0f}",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if T[i, j] > T.max() * 0.6 else "#2c3e50")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Total packets", fontsize=9)

    ax.set_xticks(range(n_tiers))
    ax.set_xticklabels(tier_names, fontsize=11)
    ax.set_yticks(range(n_tiers))
    ax.set_yticklabels(tier_names, fontsize=11)
    ax.set_xlabel("Destination tier", fontsize=11)
    ax.set_ylabel("Source tier", fontsize=11)
    ax.set_title(
        "Aggregated Demand Between Tiers",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, "traffic_tier_demand")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 58)
    print(" plot_traffic_matrix.py — Traffic Demand Model Figures")
    print("=" * 58)

    rng   = random.Random(RANDOM_SEED)
    flows = generate_flows(n_flows=500, rng=rng)
    M     = build_demand_matrix(flows)

    print(f"Generated {len(flows)} flows across {N_NODES} nodes")
    print(f"  Total packets    : {sum(f['packets'] for f in flows)}")
    print(f"  Avg packets/flow : {np.mean([f['packets'] for f in flows]):.2f}")
    print(f"  Non-zero demand pairs: {int((M > 0).sum())} / {N_NODES**2}")
    print()

    plot_traffic_heatmap(M)
    plot_flow_statistics(flows)
    plot_tier_demand(M)

    print()
    print(f"All figures saved to {FIGURES_DIR}/")
    print("Done.")
