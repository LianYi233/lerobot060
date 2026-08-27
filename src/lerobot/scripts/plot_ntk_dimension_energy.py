#!/usr/bin/env python

"""Plot task breadth and directional adaptation load for PI0.5.

Reads the multi-seed JSON produced by lerobot_ntk_gap.py and creates a
paper-facing two-panel figure:

(a) Task-effective dimension (spectral effective rank), measuring adaptation breadth.
(b) Directional Adaptation Load,

    L_m = Tr(K_m) / (|theta_m| * d_eff^m),

measuring parameter-normalized tangent energy carried by each effective task mode.

This decomposition makes the intended motivation explicit:
similar task breadth can coexist with strongly asymmetric adaptation load.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Multi-seed NTK-gap JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ntk_breadth_vs_adaptation_load.pdf"),
        help="Figure path (.pdf, .png, .svg, ...)",
    )
    parser.add_argument(
        "--log-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use logarithmic y-axis for adaptation load (recommended)",
    )
    parser.add_argument(
        "--show-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay per-seed points on top of the summary bars",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _mean_std(x: np.ndarray) -> tuple[float, float]:
    ddof = 1 if x.size > 1 else 0
    return float(x.mean()), float(x.std(ddof=ddof))


def _extract_module(data: dict, module: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranks: list[float] = []
    energies: list[float] = []
    loads: list[float] = []

    parameter_key = "vlm_num_parameters" if module == "vlm" else "action_num_parameters"

    for run in data["per_seed"]:
        rank = float(run[module]["effective_rank"])
        trace = float(run[module]["trace"])
        num_parameters = float(run[parameter_key])

        if rank <= 0:
            raise ValueError(f"Invalid effective rank for {module}: {rank}")
        if num_parameters <= 0:
            raise ValueError(f"Invalid parameter count for {module}: {num_parameters}")

        energy = trace / num_parameters
        load = energy / rank

        ranks.append(rank)
        energies.append(energy)
        loads.append(load)

    return (
        np.asarray(ranks, dtype=np.float64),
        np.asarray(energies, dtype=np.float64),
        np.asarray(loads, dtype=np.float64),
    )


def _plot_bar_with_points(
    ax,
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    ylabel: str,
    log_scale: bool,
    show_seeds: bool,
) -> None:
    labels = ["VLM", "Action Expert"]
    x = np.arange(2)

    mean_a, std_a = _mean_std(values_a)
    mean_b, std_b = _mean_std(values_b)
    means = [mean_a, mean_b]
    stds = [std_a, std_b]

    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        width=0.58,
        alpha=0.72,
    )

    if show_seeds:
        offsets = np.linspace(-0.09, 0.09, len(values_a)) if len(values_a) > 1 else np.array([0.0])
        ax.scatter(
            np.full_like(values_a, x[0], dtype=np.float64) + offsets,
            values_a,
            marker="o",
            s=24,
            alpha=0.75,
            zorder=3,
        )
        ax.scatter(
            np.full_like(values_b, x[1], dtype=np.float64) + offsets,
            values_b,
            marker="^",
            s=28,
            alpha=0.75,
            zorder=3,
        )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", which="both", alpha=0.22)

    for idx, (mean, std) in enumerate(zip(means, stds, strict=True)):
        y_text = mean + std if mean + std > 0 else mean
        label = f"{mean:.2e}" if log_scale else f"{mean:.2f}"
        ax.annotate(
            label,
            (x[idx], y_text),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main() -> None:
    args = _parse_args()
    data = json.loads(args.input.read_text())

    if "per_seed" not in data or not data["per_seed"]:
        raise ValueError("Input JSON does not contain non-empty 'per_seed' results")

    vlm_rank, vlm_energy, vlm_load = _extract_module(data, "vlm")
    action_rank, action_energy, action_load = _extract_module(data, "action")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    _plot_bar_with_points(
        axes[0],
        vlm_rank,
        action_rank,
        ylabel="Task-effective dimension",
        log_scale=False,
        show_seeds=args.show_seeds,
    )
    axes[0].set_title("(a) Adaptation breadth")

    _plot_bar_with_points(
        axes[1],
        vlm_load,
        action_load,
        ylabel=r"Directional adaptation load  $\mathrm{Tr}(K)/(|\theta| d_{\mathrm{eff}})$",
        log_scale=args.log_load,
        show_seeds=args.show_seeds,
    )
    axes[1].set_title("(b) Adaptation load")

    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    vlm_rank_mean, vlm_rank_std = _mean_std(vlm_rank)
    action_rank_mean, action_rank_std = _mean_std(action_rank)
    vlm_energy_mean, vlm_energy_std = _mean_std(vlm_energy)
    action_energy_mean, action_energy_std = _mean_std(action_energy)
    vlm_load_mean, vlm_load_std = _mean_std(vlm_load)
    action_load_mean, action_load_std = _mean_std(action_load)

    raw_trace_ratios = np.asarray(
        [
            float(run["action"]["trace"]) / max(float(run["vlm"]["trace"]), 1e-30)
            for run in data["per_seed"]
        ],
        dtype=np.float64,
    )
    normalized_energy_ratios = action_energy / np.maximum(vlm_energy, 1e-30)
    load_ratios = action_load / np.maximum(vlm_load, 1e-30)

    print("=== Task Breadth vs Directional Adaptation Load ===")
    print(
        f"VLM    effective_rank={vlm_rank_mean:.3f} +/- {vlm_rank_std:.3f}, "
        f"energy={vlm_energy_mean:.6e} +/- {vlm_energy_std:.6e}, "
        f"load={vlm_load_mean:.6e} +/- {vlm_load_std:.6e}"
    )
    print(
        f"Action effective_rank={action_rank_mean:.3f} +/- {action_rank_std:.3f}, "
        f"energy={action_energy_mean:.6e} +/- {action_energy_std:.6e}, "
        f"load={action_load_mean:.6e} +/- {action_load_std:.6e}"
    )
    print(
        "Raw trace ratio Action/VLM: "
        f"{raw_trace_ratios.mean():.3f} +/- {raw_trace_ratios.std(ddof=1):.3f}"
    )
    print(
        "Parameter-normalized energy ratio Action/VLM: "
        f"{normalized_energy_ratios.mean():.3f} +/- {normalized_energy_ratios.std(ddof=1):.3f}"
    )
    print(
        "Directional adaptation load ratio Action/VLM: "
        f"{load_ratios.mean():.3f} +/- {load_ratios.std(ddof=1):.3f}"
    )
    print(f"Load>1 fraction: {float(np.mean(load_ratios > 1.0)) * 100:.2f}%")
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
