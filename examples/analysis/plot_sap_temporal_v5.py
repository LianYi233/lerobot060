#!/usr/bin/env python3
"""Reproducible, appendix-ready summary of the pre-specified P0--P4 diagnostics.

Plots ALL tested variants, regardless of the ranking of SAP vs controls.
Run only after completing P0, P1, P2, P3 and P4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


MODELS = ("init", "sap", "shuffled")
COLORS = {"init": "#88929b", "sap": "#007d77", "shuffled": "#c5793a"}


def _load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing preceding analysis stage: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_axis(ax, ylabel, title):
    ax.set_title(title, fontsize=10, fontweight="semibold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="y", lw=.5, color="#e1e4e8", alpha=.8)


def _grouped(ax, names, values, *, model_names=MODELS, rotate=0):
    centers = np.arange(len(names), dtype=float)
    width = .22 if len(model_names) == 3 else .32
    for index, label in enumerate(model_names):
        xs = centers + (index - (len(model_names) - 1) / 2.) * width
        ys = [float(values[label][name]) if values[label][name] is not None else np.nan for name in names]
        ax.bar(xs, ys, width=width, color=COLORS.get(label, "#54657e"), label=label)
    ax.set_xticks(centers, names, rotation=rotate)


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite completed figure folder: {output}")
    sources = {
        "P0": root / "P0-data-check/data_order_preflight_v5.json",
        "P1": root / "P1-blocked/metrics_temporal_v5.json",
        "P2": root / "P2-synthetic/recovery/ar2_recovery_v5.json",
        "P3": root / "P3-probe/future_probe_v5.json",
        "P4": root / "P4-signature/signature_mmd_v5.json",
    }
    data = {name: _load_json(path) for name, path in sources.items()}
    p0, p1, p2, p3, p4 = (data[k] for k in ("P0", "P1", "P2", "P3", "P4"))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    horizons = sorted(p0["horizons"], key=int)
    _grouped(ax, horizons, {
        "ordered": {h: p0["horizons"][h]["ordered"]["relative_mse"] for h in horizons},
        "unordered": {h: p0["horizons"][h]["unordered"]["relative_mse"] for h in horizons},
    }, model_names=("ordered", "unordered"))
    # Both representations use identical ridge capacity, but different information.
    for rectangle in ax.patches[:len(horizons)]:
        rectangle.set_color(COLORS["sap"])
    for rectangle in ax.patches[len(horizons):]:
        rectangle.set_color(COLORS["shuffled"])
    _clean_axis(ax, "Relative future MSE ↓", "(a) Real action order information")
    ax.set_xlabel("Future horizon", fontsize=9)
    ax.legend(["ordered", "unordered"], frameon=False, fontsize=8)

    modes = list(p1["results"])
    ax = axes[0, 1]
    _grouped(ax, modes, {label: {mode: p1["results"][mode]["models"][label]["high_noise_flow_mse"]
                         for mode in modes} for label in MODELS}, rotate=15)
    _clean_axis(ax, "High-noise flow MSE ↓", "(b) Real held-out flow matching")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 2]
    if all("full_ode_masked_action_mse" in p1["results"][mode]["models"][label]
           for mode in modes for label in MODELS):
        _grouped(ax, modes, {
            label: {mode: p1["results"][mode]["models"][label]["full_ode_masked_action_mse"]
                    for mode in modes} for label in MODELS}, rotate=15)
    else:
        ax.text(.5, .5, "No full-ODE samples", ha="center", va="center", transform=ax.transAxes)
    _clean_axis(ax, "Full-ODE action MSE ↓", "(c) Pure-noise trajectory generation")

    ax = axes[1, 0]
    _grouped(ax, ["known AR(2)"], {label: {"known AR(2)": p2["results"][label]["generated"][
        "mean_coefficient_l2"]} for label in MODELS})
    oracle = p2["results"]["init"]["oracle_real"]["mean_coefficient_l2"]
    if oracle is not None:
        ax.axhline(oracle, color="#333333", lw=1.1, linestyle="--", label="real-data fit")
    _clean_axis(ax, "Coefficient L2 error ↓", "(d) Known synthetic dynamics")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    horizons = sorted({h for group in p3["results"].values() for h in group}, key=int)
    for label in MODELS:
        ys = [p3["results"][label][h]["R2_relative_to_intercept"] for h in horizons]
        ax.plot(np.asarray(horizons, dtype=int), ys, marker="o", lw=2.,
                color=COLORS[label], label=label)
    ax.axhline(0., color="#777", ls="--", lw=1.)
    ax.set_xticks([int(h) for h in horizons])
    ax.set_xlabel("Future lag", fontsize=9)
    _clean_axis(ax, "Held-out probe R² ↑", "(e) Frozen expert predictive-state probe")

    ax = axes[1, 2]
    scores = [p4["results"][label]["biased_product_mmd_squared"] for label in MODELS]
    ax.bar(MODELS, scores, color=[COLORS[name] for name in MODELS], width=.6)
    _clean_axis(ax, "Finite signature MMD² ↓", "(f) Context-weighted path distribution")

    fig.suptitle("SAP temporal-action diagnostics (controlled comparisons; report all results)",
                 fontweight="semibold", fontsize=13)
    output.mkdir(parents=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"sap_temporal_evidence_v5.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    reproducibility = {
        "protocol": "primingvla-v5-appended-figure-manifest",
        "interpretation": "Descriptive diagnostics; cannot prove a universal SAP superiority claim",
        "inputs": {label: {"path": str(path),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for label, path in sources.items()},
        "figures": ["sap_temporal_evidence_v5.png", "sap_temporal_evidence_v5.pdf"],
    }
    (output / "figure_manifest_v5.json").write_text(
        json.dumps(reproducibility, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[FIGURE] saved all six P0--P4 panels: {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    plot(parser.parse_args(argv))


if __name__ == "__main__":
    main()
