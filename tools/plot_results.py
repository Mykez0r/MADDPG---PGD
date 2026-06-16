#!/usr/bin/env python3
"""Generate figures from experiment result JSON files."""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _resolve_output_dir(output_dir: Optional[Path]) -> Path:
    base_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent / "figures"
    figure_dir = base_dir / "figures" if base_dir.name != "figures" else base_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def _save(fig, output_dir: Optional[Path], stem: str):
    figure_dir = _resolve_output_dir(output_dir)
    for ext in ("png", "pdf"):
        path = figure_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"saved -> {path}")
    plt.close(fig)


def _load_json(path: Path) -> Dict:
    with open(path, "r") as handle:
        return json.load(handle)


def _variant_colors(names: Iterable[str]) -> Dict[str, Tuple[float, float, float, float]]:
    names = list(names)
    cmap = plt.cm.get_cmap("tab10" if len(names) <= 10 else "tab20", max(len(names), 1))
    return {name: cmap(idx) for idx, name in enumerate(names)}


def _rolling_mean(values: List[float], window: int = 5) -> np.ndarray:
    if not values:
        return np.array([])
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_attack_case_key(key: str) -> Optional[Tuple[str, float]]:
    if "_eps" not in key:
        return None
    attack_type, epsilon_text = key.rsplit("_eps", 1)
    try:
        return attack_type, float(epsilon_text)
    except ValueError:
        return None


def plot_phase1_training(results_dict: Dict, output_dir: Optional[Path] = None):
    if not results_dict:
        warnings.warn("Phase 1 results are empty; skipping Phase 1 plots.")
        return

    variants = sorted(name for name, payload in results_dict.items() if isinstance(payload, dict))
    if not variants:
        return

    colors = _variant_colors(variants)

    fig, ax = plt.subplots(figsize=(12, 6))
    for name in variants:
        rewards = results_dict[name].get("rewards", [])
        if rewards:
            ax.plot(_rolling_mean(rewards, window=7), label=name, linewidth=2.0, color=colors[name])
    ax.set_title("Phase 1 Training Reward Evolution")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Episode Reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    _save(fig, output_dir, "phase1_reward_curves")

    fig, ax = plt.subplots(figsize=(12, 6))
    for name in variants:
        losses = results_dict[name].get("pkt_losses", [])
        if losses:
            ax.plot(_rolling_mean(losses, window=7), label=name, linewidth=2.0, color=colors[name])
    ax.set_title("Phase 1 Training Packet Loss Evolution")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Packet Loss (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    _save(fig, output_dir, "phase1_packet_loss_curves")

    ranking_rows = []
    for name in variants:
        payload = results_dict[name]
        ranking_rows.append((name, _safe_float(payload.get("final_reward", np.mean(payload.get("rewards", [0.0]))))))
    ranking_rows.sort(key=lambda item: item[1], reverse=True)
    labels = [name for name, _ in ranking_rows]
    values = [value for _, value in ranking_rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, values, color=[colors[name] for name in labels], alpha=0.9)
    ax.invert_yaxis()
    ax.set_title("Phase 1 Final Reward Ranking")
    ax.set_xlabel("Final Reward")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "phase1_final_reward_ranking")


def plot_phase2_evaluation(results_dict: Dict,
                           output_dir: Optional[Path] = None,
                           ranking_dict: Optional[Dict] = None):
    if not results_dict:
        warnings.warn("Phase 2 results are empty; skipping Phase 2 plots.")
        return

    variants = sorted(name for name in results_dict.keys() if not str(name).startswith("SP"))
    baselines = sorted(name for name in results_dict.keys() if str(name).startswith("SP"))
    if "SP" in results_dict and "SP" not in baselines:
        baselines.append("SP")
    if not variants:
        return

    scenarios = [scenario for scenario in ("normal", "dual_link_failure") if any(
        isinstance(results_dict.get(name), dict) and scenario in results_dict.get(name, {})
        for name in results_dict.keys()
    )]
    colors = _variant_colors(variants + baselines)

    if ranking_dict and ranking_dict.get("scenarios"):
        profiles = sorted(ranking_dict.get("profiles", {}).keys())
        for scenario in scenarios:
            for profile in profiles:
                ranked = ranking_dict["scenarios"].get(scenario, {}).get(profile, [])
                if not ranked:
                    continue
                names = [row["name"] for row in ranked]
                scores = [_safe_float(row.get("composite_score", 0.0)) for row in ranked]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(names, scores, color=[colors.get(name, "#4c72b0") for name in names], alpha=0.9)
                ax.set_title(f"Phase 2 Composite Scores: {scenario} ({profile})")
                ax.set_ylabel("Composite Score")
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=25, ha="right")
                ax.grid(axis="y", alpha=0.3)
                fig.tight_layout()
                _save(fig, output_dir, f"phase2_composite_{scenario}_{profile}")

    metrics = [
        ("mean_end_to_end_pdr", "End-to-End PDR (%)"),
        ("mean_hop_delivery_frac", "Per-Hop Delivery Frac (%)"),
        ("mean_pkt_loss", "Packet Loss (%)"),
        ("mean_delay_p95", "Delay P95"),
        ("mean_hops_mean", "Mean Hops"),
        ("mean_backlog_end", "Backlog End"),
    ]
    for scenario in scenarios:
        row_names = variants + baselines
        matrix = []
        for name in row_names:
            payload = results_dict.get(name, {}).get(scenario, {})
            matrix.append([_safe_float(payload.get(metric, 0.0)) for metric, _ in metrics])
        matrix = np.asarray(matrix, dtype=float)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_title(f"Phase 2 KPI Matrix: {scenario}")
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([label for _, label in metrics], rotation=20, ha="right")
        ax.set_yticks(range(len(row_names)))
        ax.set_yticklabels(row_names)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        _save(fig, output_dir, f"phase2_kpi_matrix_{scenario}")

    if ranking_dict:
        seed_info = ranking_dict.get("seed_convergence", {})
        history = seed_info.get("stability_history", {})
        if history:
            ordered = sorted((int(seed), _safe_float(score)) for seed, score in history.items())
            seeds = [seed for seed, _ in ordered]
            taus = [score for _, score in ordered]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(seeds, taus, marker="o", linewidth=2.0)
            ax.axhline(_safe_float(seed_info.get("stability_threshold", 0.95), 0.95), color="red", linestyle="--")
            ax.set_title("Phase 2 Rank Stability Evolution")
            ax.set_xlabel("Evaluation Episodes")
            ax.set_ylabel("Kendall Tau")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            _save(fig, output_dir, "phase2_rank_stability")

    row_names = variants + baselines
    scenario_matrix = []
    for name in row_names:
        scenario_matrix.append([_safe_float(results_dict.get(name, {}).get(scenario, {}).get("mean_end_to_end_pdr", 0.0)) for scenario in scenarios])
    if scenario_matrix:
        matrix = np.asarray(scenario_matrix, dtype=float)
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(matrix, cmap="GnBu", aspect="auto")
        ax.set_title("Phase 2 Scenario End-to-End PDR Heatmap")
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios)
        ax.set_yticks(range(len(row_names)))
        ax.set_yticklabels(row_names)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        _save(fig, output_dir, "phase2_delivery_heatmap")


def plot_phase2_load_sweep(load_sweep_dict: Dict,
                           output_dir: Optional[Path] = None):
    if not load_sweep_dict:
        return

    methods = load_sweep_dict.get("methods", {})
    if not methods:
        return

    scenarios = ["normal"]
    meta = load_sweep_dict.get("meta", {})
    if bool(meta.get("include_failures", False)):
        scenarios.append("dual_link_failure")

    for scenario in scenarios:
        for metric, title, ylabel in [
            ("mean_goodput_per_step", "Goodput", "Goodput / step"),
            ("mean_pkt_loss", "Packet Loss", "Packet Loss (%)"),
            ("mean_delay_p95", "Delay P95", "Delay P95"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 6))
            for method_name, payload in sorted(methods.items()):
                scenario_rows = payload.get(scenario, {})
                points = []
                for load_key, row in scenario_rows.items():
                    try:
                        load = float(str(load_key).split("_")[-1])
                    except (TypeError, ValueError):
                        continue
                    points.append((load, _safe_float(row.get(metric, 0.0))))
                if not points:
                    continue
                points.sort(key=lambda x: x[0])
                ax.plot(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    marker="o",
                    linewidth=2.0,
                    label=method_name,
                )
            ax.set_title(f"Phase 2 Load Sweep: {title} ({scenario})")
            ax.set_xlabel("Offered Load Factor")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            _save(fig, output_dir, f"phase2_load_sweep_{scenario}_{metric}")

    summary = load_sweep_dict.get("summary", {})
    if summary:
        names = sorted(summary.keys())
        values = [_safe_float(summary[n].get("critical_load_normal", 0.0)) for n in names]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(names, values, alpha=0.9)
        ax.set_title("Phase 2 Critical Load Comparison (Normal)")
        ax.set_ylabel("Critical Load Factor")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        _save(fig, output_dir, "phase2_critical_load_normal")


def plot_phase3_fgsm(results_dict: Dict,
                     output_dir: Optional[Path] = None,
                     ranking_dict: Optional[Dict] = None):
    if not results_dict:
        warnings.warn("Phase 3 results are empty; skipping Phase 3 plots.")
        return

    variants = sorted(name for name, payload in results_dict.items() if not name.startswith("_") and isinstance(payload, dict))
    if not variants:
        return

    colors = _variant_colors(variants)
    by_attack_type: Dict[str, List[Tuple[float, str, Dict]]] = {}
    for name in variants:
        payload = results_dict[name]
        for key, case in payload.items():
            if not isinstance(case, dict) or key in {"surface", "attack_summary", "_runtime", "_pruned", "gnn_embedding_attack"}:
                continue
            parsed = _parse_attack_case_key(key)
            if parsed is None or "attacked" not in case:
                continue
            attack_type, epsilon = parsed
            by_attack_type.setdefault(attack_type, []).append((epsilon, name, case))

    for attack_type, rows in by_attack_type.items():
        grouped: Dict[str, List[Tuple[float, float]]] = {}
        for epsilon, name, case in rows:
            grouped.setdefault(name, []).append((epsilon, _safe_float(case["attacked"].get("mean_pkt_loss", 0.0))))
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, points in sorted(grouped.items()):
            points.sort(key=lambda item: item[0])
            ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=2.0,
                    color=colors[name], label=name)
        ax.set_title(f"Phase 3 Packet Loss vs Epsilon: {attack_type}")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Packet Loss (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        _save(fig, output_dir, f"phase3_epsilon_curve_{attack_type}")

    attack_types = sorted(by_attack_type.keys())
    if attack_types:
        fig, axes = plt.subplots(len(attack_types), 1, figsize=(10, max(4, 4 * len(attack_types))), squeeze=False)
        for row_idx, attack_type in enumerate(attack_types):
            ax = axes[row_idx][0]
            rows = by_attack_type[attack_type]
            grouped: Dict[str, List[Tuple[float, float]]] = {}
            for epsilon, name, case in rows:
                clean_loss = _safe_float(case.get("clean", {}).get("mean_pkt_loss", 0.0))
                attacked_loss = _safe_float(case.get("attacked", {}).get("mean_pkt_loss", 0.0))
                grouped.setdefault(name, []).append((epsilon, attacked_loss - clean_loss))
            for name, points in sorted(grouped.items()):
                points.sort(key=lambda item: item[0])
                ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=2.0,
                        color=colors[name], label=name)
            ax.set_title(f"FGSM Degradation Curve: {attack_type}")
            ax.set_xlabel("Epsilon")
            ax.set_ylabel("Packet Loss Increase (pp)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, output_dir, "phase3_degradation_curves")

    critical_rows = []
    for name in variants:
        summary = results_dict[name].get("attack_summary", {}).get("attack_types", {})
        for attack_type, payload in summary.items():
            critical = payload.get("critical_epsilon")
            if critical is not None:
                critical_rows.append((attack_type, name, _safe_float(critical)))
    if critical_rows:
        attack_types = sorted({row[0] for row in critical_rows})
        attack_index = {attack: idx for idx, attack in enumerate(attack_types)}
        fig, ax = plt.subplots(figsize=(10, 6))
        for attack_type, name, epsilon in critical_rows:
            ax.scatter(attack_index[attack_type], epsilon, s=90, color=colors[name], label=name)
        ax.set_title("Phase 3 Critical Epsilon Discovery")
        ax.set_xlabel("Attack Type")
        ax.set_ylabel("Critical Epsilon")
        ax.set_xticks(range(len(attack_types)))
        ax.set_xticklabels(attack_types, rotation=20, ha="right")
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8, ncol=2)
        fig.tight_layout()
        _save(fig, output_dir, "phase3_critical_epsilon")

    surface_rows = []
    for name in variants:
        surface = results_dict[name].get("surface")
        if not isinstance(surface, dict):
            continue
        surface_rows.append((
            name,
            _safe_float(surface.get("core_attacked", {}).get("results", {}).get("mean_pkt_loss", 0.0)),
            _safe_float(surface.get("dist_attacked", {}).get("results", {}).get("mean_pkt_loss", 0.0)),
            _safe_float(surface.get("access_attacked", {}).get("results", {}).get("mean_pkt_loss", 0.0)),
        ))
    if surface_rows:
        names = [row[0] for row in surface_rows]
        core_vals = [row[1] for row in surface_rows]
        dist_vals = [row[2] for row in surface_rows]
        access_vals = [row[3] for row in surface_rows]
        x = np.arange(len(names))
        width = 0.25
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(x - width, core_vals, width=width, label="Core", alpha=0.9)
        ax.bar(x, dist_vals, width=width, label="Distribution", alpha=0.9)
        ax.bar(x + width, access_vals, width=width, label="Access", alpha=0.9)
        ax.set_title("Phase 3 Attack Surface Sensitivity")
        ax.set_ylabel("Packet Loss (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        _save(fig, output_dir, "phase3_attack_surface")

    if ranking_dict and ranking_dict.get("overall"):
        for profile, ranked in sorted(ranking_dict["overall"].items()):
            if not ranked:
                continue
            names = [row["name"] for row in ranked]
            scores = [_safe_float(row.get("composite_score", 0.0)) for row in ranked]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(names, scores, color=[colors.get(name, "#4c72b0") for name in names], alpha=0.9)
            ax.set_title(f"Phase 3 Overall Robustness Ranking ({profile})")
            ax.set_ylabel("Composite Score")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            _save(fig, output_dir, f"phase3_overall_ranking_{profile}")


def plot_all_phases(results_dir: Path,
                    phases: Optional[List[int]] = None,
                    output_dir: Optional[Path] = None):
    results_dir = Path(results_dir)
    phases = phases or [1, 2, 3]
    output_dir = output_dir or results_dir

    if 1 in phases:
        phase1_path = results_dir / "phase1_training_results.json"
        if phase1_path.exists():
            plot_phase1_training(_load_json(phase1_path), output_dir)

    if 2 in phases:
        phase2_path = results_dir / "phase2_maddpg_results.json"
        phase2_rankings = results_dir / "phase2_rankings.json"
        phase2_load_sweep = results_dir / "phase2_load_sweep_results.json"
        if phase2_path.exists():
            plot_phase2_evaluation(
                _load_json(phase2_path),
                output_dir,
                _load_json(phase2_rankings) if phase2_rankings.exists() else None,
            )
        if phase2_load_sweep.exists():
            plot_phase2_load_sweep(
                _load_json(phase2_load_sweep),
                output_dir,
            )

    if 3 in phases:
        phase3_path = results_dir / "phase3_fgsm_results.json"
        phase3_rankings = results_dir / "phase3_rankings.json"
        if phase3_path.exists():
            plot_phase3_fgsm(
                _load_json(phase3_path),
                output_dir,
                _load_json(phase3_rankings) if phase3_rankings.exists() else None,
            )


def main():
    parser = argparse.ArgumentParser(description="Generate figures from experiment result JSON files.")
    parser.add_argument("--results-dir", required=True, help="Directory containing phase JSON files.")
    parser.add_argument("--phases", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--output-dir", default=None, help="Optional base directory for figures.")
    args = parser.parse_args()

    plot_all_phases(
        Path(args.results_dir),
        phases=args.phases,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
