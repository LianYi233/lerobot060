#!/usr/bin/env python

"""Plot relative task breadth and directional adaptation load for PI0.5.

Reads the multi-seed JSON produced by lerobot_ntk_gap.py and creates a
paper-facing two-panel figure using paired Action/VLM ratios:

(a) log10 task-effective breadth ratio
        log10(d_eff^Action / d_eff^VLM)

(b) log10 directional adaptation-load ratio
        log10(L_Action / L_VLM)

where

        L_m = Tr(K_m) / (|theta_m| * d_eff^m).

A value of zero is the no-mismatch baseline in both panels. Per-seed points are
shown together with the median and interquartile range (IQR), which is more
robust than mean +/- std for the long-tailed load-ratio distribution.
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
        default=Path("ntk_relative_breadth_vs_load.pdf"),
        help="Figure path (.pdf, .png, .svg, ...)",
    )
    parser.add_argument(
        "--show-seed-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Annotate each point with its seed id",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _extract_module(data: dict, module: str) -> tuple[np.ndarray, np.ndarray]:
    ranks: list[float] = []
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

        load = trace / (num_parameters * rank)
        ranks.append(rank)
        loads.append(load)

    return np.asarray(ranks, dtype=np.float64), np.asarray(loads, dtype=np.float64)


def _summary(x: np.ndarray) -> tuple[float, float, float]:
    median = float(np.median(x))
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    return median, q1, q3


def _plot_relative_panel(
    ax,
    values: np.ndarray,
    *,
    title: str,
    ylabel: str,
    seed_ids: list[int],
    show_seed_labels: bool,
) -> None:
    x = np.arange(len(values), dtype=np.float64)
    median, q1, q3 = _summary(values)

    ax.axhline(0.0, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.scatter(x, values, s=34, alpha=0.82, zorder=3)

    # Robust summary: median with IQR band.
    ax.axhspan(q1, q3, alpha=0.12, zorder=0)
    ax.axhline(median, linewidth=2.0, alpha=0.9)

    if show_seed_labels:
        for xi, yi, seed in zip(x, values, seed_ids, strict=True):
            ax.annotate(
                str(seed),
                (xi, yi),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=7,
                alpha=0.8,
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Seed")
    ax.set_xticks(x, [str(s) for s in seed_ids])
    ax.grid(True, axis="y", alpha=0.22)

    positive_fraction = float(np.mean(values > 0.0)) * 100.0
    ax.text(
        0.03,
        0.97,
        f"median={median:.3f}\nIQR=[{q1:.3f}, {q3:.3f}]\n>0: {positive_fraction:.0f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )


def main() -> None:
    args = _parse_args()
    data = json.loads(args.input.read_text())

    if "per_seed" not in data or not data["per_seed"]:
        raise ValueError("Input JSON does not contain non-empty 'per_seed' results")

    seed_ids = [int(run.get("seed", i)) for i, run in enumerate(data["per_seed"])]

    vlm_rank, vlm_load = _extract_module(data, "vlm")
    action_rank, action_load = _extract_module(data, "action")

    breadth_ratio = action_rank / np.maximum(vlm_rank, 1e-30)
    load_ratio = action_load / np.maximum(vlm_load, 1e-30)

    log_breadth_ratio = np.log10(breadth_ratio)
    log_load_ratio = np.log10(load_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))

    _plot_relative_panel(
        axes[0],
        log_breadth_ratio,
        title="(a) Relative adaptation breadth",
        ylabel=r"$\log_{10}(d_{\mathrm{eff}}^{A}/d_{\mathrm{eff}}^{V})$",
        seed_ids=seed_ids,
        show_seed_labels=args.show_seed_labels,
    )

    _plot_relative_panel(
        axes[1],
        log_load_ratio,
        title="(b) Relative adaptation load",
        ylabel=r"$\log_{10}(L_A/L_V)$",
        seed_ids=seed_ids,
        show_seed_labels=args.show_seed_labels,
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    breadth_median, breadth_q1, breadth_q3 = _summary(breadth_ratio)
    load_median, load_q1, load_q3 = _summary(load_ratio)

    breadth_log_median, breadth_log_q1, breadth_log_q3 = _summary(log_breadth_ratio)
    load_log_median, load_log_q1, load_log_q3 = _summary(log_load_ratio)

    print("=== Relative Task Breadth vs Directional Adaptation Load ===")
    print(
        "Breadth ratio Action/VLM: "
        f"median={breadth_median:.3f}, IQR=[{breadth_q1:.3f}, {breadth_q3:.3f}], "
        f">1 fraction={float(np.mean(breadth_ratio > 1.0)) * 100:.2f}%"
    )
    print(
        "Load ratio Action/VLM: "
        f"median={load_median:.3f}, IQR=[{load_q1:.3f}, {load_q3:.3f}], "
        f">1 fraction={float(np.mean(load_ratio > 1.0)) * 100:.2f}%"
    )
    print(
        "log10 breadth ratio: "
        f"median={breadth_log_median:.3f}, IQR=[{breadth_log_q1:.3f}, {breadth_log_q3:.3f}]"
    )
    print(
        "log10 load ratio: "
        f"median={load_log_median:.3f}, IQR=[{load_log_q1:.3f}, {load_log_q3:.3f}]"
    )
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
