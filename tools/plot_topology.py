#!/usr/bin/env python3
"""
tools/plot_topology.py
----------------------
Generates publication-quality figures of the MADDPG network topology.

Outputs (saved to tools/figures/):
  topology_full.pdf / .png      - full 65-node service-provider graph
  topology_tiers.pdf / .png     - same graph, nodes coloured by tier
  degree_distribution.pdf / .png - node degree histogram

Usage (local machine, no Docker needed):
  pip install matplotlib networkx numpy
  python tools/plot_topology.py
"""

import sys
import os
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ---------------------------------------------------------------------------
# Topology builder  (mirrors network_environment.py exactly)
# ---------------------------------------------------------------------------

def build_service_provider_topology(n_nodes: int = 65, seed: int = RANDOM_SEED) -> nx.Graph:
    """Recreate the service-provider topology used in experiments."""
    rng = random.Random(seed)
    G = nx.Graph()

    for i in range(n_nodes):
        G.add_node(f"H{i}")

    nodes = list(G.nodes())

    # Tier sizes (must match network_environment.py)
    core_size  = max(3, int(0.1 * n_nodes))          # ~7
    dist_size  = int(0.3 * n_nodes)                  # ~19
    core_nodes = nodes[:core_size]
    dist_nodes = nodes[core_size : core_size + dist_size]
    access_nodes = nodes[core_size + dist_size :]

    # Core full mesh
    for i in range(len(core_nodes)):
        for j in range(i + 1, len(core_nodes)):
            G.add_edge(core_nodes[i], core_nodes[j],
                       capacity=10.0, tier="core")

    # Distribution -> core
    for d in dist_nodes:
        c = rng.choice(core_nodes)
        G.add_edge(d, c, capacity=8.0, tier="dist_core")
        if rng.random() < 0.3:
            other = rng.choice([n for n in dist_nodes if n != d])
            if not G.has_edge(d, other):
                G.add_edge(d, other, capacity=5.0, tier="dist_dist")

    # Access -> distribution
    for a in access_nodes:
        d = rng.choice(dist_nodes)
        G.add_edge(a, d, capacity=2.0, tier="access_dist")
        if rng.random() < 0.1:
            c = rng.choice(core_nodes)
            if not G.has_edge(a, c):
                G.add_edge(a, c, capacity=1.0, tier="access_core")

    # Store tier membership as node attribute
    for n in core_nodes:   G.nodes[n]["tier"] = "core"
    for n in dist_nodes:   G.nodes[n]["tier"] = "distribution"
    for n in access_nodes: G.nodes[n]["tier"] = "access"

    return G, core_nodes, dist_nodes, access_nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIER_COLOURS = {
    "core":         "#e74c3c",   # red
    "distribution": "#f39c12",  # orange
    "access":       "#3498db",  # blue
}

TIER_LABELS = {
    "core":         "Core (full-mesh, cap=10)",
    "distribution": "Distribution (cap=5-8)",
    "access":       "Access (cap=1-2)",
}


def _capacity_to_width(cap: float) -> float:
    """Map link capacity to line width for visualisation."""
    return max(0.3, cap / 3.0)


def _save(fig, stem: str):
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Full topology coloured by tier
# ---------------------------------------------------------------------------

def plot_topology_tiers(G: nx.Graph):
    print("Plotting topology (tier colours) ...")

    pos = nx.spring_layout(G, seed=RANDOM_SEED, k=0.6)

    node_colours = [TIER_COLOURS[G.nodes[n]["tier"]] for n in G.nodes()]
    node_sizes   = [
        280 if G.nodes[n]["tier"] == "core" else
        160 if G.nodes[n]["tier"] == "distribution" else 80
        for n in G.nodes()
    ]

    edge_widths  = [_capacity_to_width(G[u][v]["capacity"]) for u, v in G.edges()]
    edge_colours = ["#2c3e50" for _ in G.edges()]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(
        f"Service-Provider Network Topology\n"
        f"{G.number_of_nodes()} nodes · {G.number_of_edges()} links",
        fontsize=14, fontweight="bold", pad=14
    )

    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=edge_widths, edge_color=edge_colours, alpha=0.55)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colours, node_size=node_sizes, alpha=0.92)

    # Label only core nodes (too crowded otherwise)
    core_labels = {n: n for n in G.nodes() if G.nodes[n]["tier"] == "core"}
    nx.draw_networkx_labels(G, pos, labels=core_labels, ax=ax,
                            font_size=7, font_color="white", font_weight="bold")

    # Legend
    patches = [mpatches.Patch(color=TIER_COLOURS[t], label=TIER_LABELS[t])
               for t in ("core", "distribution", "access")]
    ax.legend(handles=patches, loc="upper left", fontsize=9, framealpha=0.85)
    ax.axis("off")
    fig.tight_layout()
    _save(fig, "topology_tiers")


# ---------------------------------------------------------------------------
# Figure 2: Capacity heatmap on edges (spring layout, edge colour = capacity)
# ---------------------------------------------------------------------------

def plot_topology_capacity(G: nx.Graph):
    print("Plotting topology (edge capacity heatmap) ...")

    pos = nx.spring_layout(G, seed=RANDOM_SEED, k=0.6)

    capacities = np.array([G[u][v]["capacity"] for u, v in G.edges()])
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=capacities.min(), vmax=capacities.max())
    edge_colours = [cmap(norm(c)) for c in capacities]
    edge_widths  = [_capacity_to_width(c) for c in capacities]

    node_colours = [TIER_COLOURS[G.nodes[n]["tier"]] for n in G.nodes()]
    node_sizes   = [
        280 if G.nodes[n]["tier"] == "core" else
        160 if G.nodes[n]["tier"] == "distribution" else 80
        for n in G.nodes()
    ]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(
        "Service-Provider Topology — Link Capacity",
        fontsize=14, fontweight="bold", pad=14
    )

    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=edge_widths, edge_color=edge_colours, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colours, node_size=node_sizes, alpha=0.92)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Link capacity (normalised)", fontsize=9)

    # Tier legend
    patches = [mpatches.Patch(color=TIER_COLOURS[t], label=t.capitalize())
               for t in ("core", "distribution", "access")]
    ax.legend(handles=patches, loc="upper left", fontsize=9, framealpha=0.85)
    ax.axis("off")
    fig.tight_layout()
    _save(fig, "topology_capacity")


# ---------------------------------------------------------------------------
# Figure 3: Degree distribution
# ---------------------------------------------------------------------------

def plot_degree_distribution(G: nx.Graph):
    print("Plotting degree distribution ...")

    degrees = dict(G.degree())
    tier_degrees = {"core": [], "distribution": [], "access": []}
    for n, deg in degrees.items():
        tier_degrees[G.nodes[n]["tier"]].append(deg)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: overall histogram
    ax = axes[0]
    all_deg = list(degrees.values())
    ax.hist(all_deg, bins=range(min(all_deg), max(all_deg) + 2),
            color="#2980b9", edgecolor="white", alpha=0.85, align="left")
    ax.set_xlabel("Node degree", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Degree Distribution (all nodes)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)

    # Right: boxplot per tier
    ax = axes[1]
    data   = [tier_degrees["core"], tier_degrees["distribution"], tier_degrees["access"]]
    labels = ["Core", "Distribution", "Access"]
    colours = [TIER_COLOURS[t] for t in ("core", "distribution", "access")]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.85)
    ax.set_ylabel("Node degree", fontsize=11)
    ax.set_title("Degree Distribution by Tier", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)

    fig.suptitle(
        f"Network Topology — Degree Analysis  "
        f"(mean={np.mean(all_deg):.1f}, max={max(all_deg)})",
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    _save(fig, "degree_distribution")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print(" plot_topology.py — MADDPG Network Topology Figures")
    print("=" * 55)

    G, core_nodes, dist_nodes, access_nodes = build_service_provider_topology(n_nodes=65)

    print(f"Topology: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Core        : {len(core_nodes)} nodes")
    print(f"  Distribution: {len(dist_nodes)} nodes")
    print(f"  Access      : {len(access_nodes)} nodes")
    print(f"  Avg degree  : {np.mean([d for _, d in G.degree()]):.2f}")
    print()

    plot_topology_tiers(G)
    plot_topology_capacity(G)
    plot_degree_distribution(G)

    print()
    print(f"All figures saved to {FIGURES_DIR}/")
    print("Done.")
