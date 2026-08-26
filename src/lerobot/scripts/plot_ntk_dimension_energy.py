#!/usr/bin/env python

"""Plot PI0.5 module effective dimension versus tangent energy.

Reads the multi-seed JSON produced by ``lerobot_ntk_gap.py`` and creates a
paper-facing scatter plot comparing the VLM and action pathway.

The x-axis is spectral effective rank.  The y-axis is parameter-normalized
kernel trace

    E_m = Tr(K_m) / |theta_m|,

which is the mean squared tangent-gradient energy per parameter, up to the
CountSketch approximation used by the diagnostic.  Raw trace is also available
via ``--energy-mode raw`` but should be interpreted more cautiously because the
VLM and action pathway have very different parameter counts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Multi-seed NTK-gap JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ntk_effective_dimension_vs_tangent_energy.pdf"),
        help="Figure path (.pdf, .png, .svg, ...)",
    )
    parser.add_argument(
        "--energy-mode",
        choices=("per_parameter", "raw"),
        default="per_parameter",
        help="Use Tr(K)/num_parameters (recommended) or raw Tr(K)",
    )
    parser.add_argument(
        "--log-energy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a logarithmic y-axis (recommended)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _extract(data: dict, module: str, energy_mode: str) -> tuple[np.ndarray, np.ndarray]:
    ranks = []
    energies = []
    parameter_key = "vlm_num_parameters" if module == "vlm" else "action_num_parameters"

    for run in data["per_seed"]:
        ranks.append(float(run[module]["effective_rank"]))
        trace = float(run[module]["trace"])
        if energy_mode == "per_parameter":
            num_parameters = float(run[parameter_key])
            if num_parameters <= 0:
                raise ValueError(f"Invalid {parameter_key}={num_parameters}")
            trace /= num_parameters
        energies.append(trace)

    return np.asarray(ranks, dtype=np.float64), np.asarray(energies, dtype=np.float64)


def _mean_std(x: np.ndarray) -> tuple[float, float]:
    ddof = 1 if x.size > 1 else 0
    return float(x.mean()), float(x.std(ddof=ddof))


def main() -> None:
    args = _parse_args()
    data = json.loads(args.input.read_text())
    if "per_seed" not in data or not data["per_seed"]:
        raise ValueError("Input JSON does not contain non-empty 'per_seed' results")

    vlm_rank, vlm_energy = _extract(data, "vlm", args.energy_mode)
    action_rank, action_energy = _extract(data, "action", args.energy_mode)

    fig, ax = plt.subplots(figsize=(6.3, 4.8))

    # Use matplotlib's default color cycle so the figure remains theme-friendly.
    vlm_points = ax.scatter(
        vlm_rank,
        vlm_energy,
        marker="o",
        alpha=0.7,
        label="VLM (per seed)",
    )
    action_points = ax.scatter(
        action_rank,
        action_energy,
        marker="^",
        alpha=0.7,
        label="Action Expert (per seed)",
    )

    # Match each module's mean/error bars to its scatter color without hard-coding colors.
    vlm_color = vlm_points.get_facecolor()[0]
    action_color = action_points.get_facecolor()[0]

    vlm_rank_mean, vlm_rank_std = _mean_std(vlm_rank)
    vlm_energy_mean, vlm_energy_std = _mean_std(vlm_energy)
    action_rank_mean, action_rank_std = _mean_std(action_rank)
    action_energy_mean, action_energy_std = _mean_std(action_energy)

    ax.errorbar(
        vlm_rank_mean,
        vlm_energy_mean,
        xerr=vlm_rank_std,
        yerr=vlm_energy_std,
        fmt="o",
        markersize=9,
        capsize=4,
        linewidth=1.6,
        color=vlm_color,
        label="VLM mean ± std",
    )
    ax.errorbar(
        action_rank_mean,
        action_energy_mean,
        xerr=action_rank_std,
        yerr=action_energy_std,
        fmt="^",
        markersize=10,
        capsize=4,
        linewidth=1.6,
        color=action_color,
        label="Action Expert mean ± std",
    )

    if args.log_energy:
        ax.set_yscale("log")

    ax.set_xlabel("Task-effective dimension (spectral effective rank)")
    if args.energy_mode == "per_parameter":
        ax.set_ylabel(r"Parameter-normalized tangent energy  $\mathrm{Tr}(K)/|\theta|$")
    else:
        ax.set_ylabel(r"Tangent energy  $\mathrm{Tr}(K)$")

    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    raw_trace_ratios = np.asarray(
        [
            float(run["action"]["trace"]) / max(float(run["vlm"]["trace"]), 1e-30)
            for run in data["per_seed"]
        ],
        dtype=np.float64,
    )
    normalized_energy_ratios = action_energy / np.maximum(vlm_energy, 1e-30)

    print("=== Effective Dimension vs Tangent Energy ===")
    print(
        f"VLM    effective_rank={vlm_rank_mean:.3f} +/- {vlm_rank_std:.3f}, "
        f"energy={vlm_energy_mean:.6e} +/- {vlm_energy_std:.6e}"
    )
    print(
        f"Action effective_rank={action_rank_mean:.3f} +/- {action_rank_std:.3f}, "
        f"energy={action_energy_mean:.6e} +/- {action_energy_std:.6e}"
    )
    print(
        "Raw trace ratio Action/VLM: "
        f"{raw_trace_ratios.mean():.3f} +/- {raw_trace_ratios.std(ddof=1):.3f}"
    )
    print(
        "Parameter-normalized energy ratio Action/VLM: "
        f"{normalized_energy_ratios.mean():.3f} +/- {normalized_energy_ratios.std(ddof=1):.3f}"
    )
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
