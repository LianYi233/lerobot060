"""Replot paired NTK measurements for staged and plain prompt learning, without a GPU.

Each point is one saved probe seed. No interpolation, smoothing, jitter, or invented
measurements are used. Checkpoints are aligned by cumulative updates, not stage names.
"""

import argparse
import csv
import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from lerobot.utils.ntk_protocol import (
    COMPARABLE_FIELDS,
    MEASUREMENT_FIELDS,
    measurement_identity,
    validate_measurement_pair,
)

COLORS = {"vlm": "#257F93", "action": "#D27452"}
LABELS = {"vlm": "VLM", "action": "Action expert"}
DEFAULT_STEPS = (0, 750, 1000, 4000)
METRICS = ("effective_rank", "parameter_normalized_energy")


@dataclass
class Measurements:
    path: Path
    label: str
    manifest: dict
    records: dict
    stages: dict
    signatures: dict
    group_metadata: dict
    measurement_digest: str
    file_digest: str


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def parse_steps(value):
    try:
        steps = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Steps must be comma-separated nonnegative integers") from error
    if not steps or any(step < 0 for step in steps) or list(steps) != sorted(set(steps)):
        raise argparse.ArgumentTypeError("Steps must be unique nonnegative integers in increasing order")
    return steps


def _required(mapping, fields, context):
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"{context}: missing {', '.join(missing)}")


def validate_document(document, path, label, steps=DEFAULT_STEPS, scope="backbone", file_digest=""):
    """Check completeness and signatures before any plot or CSV is written."""
    _required(document, ("manifest", "records"), str(path))
    manifest = document["manifest"]
    _required(manifest, (*MEASUREMENT_FIELDS, "device", "stages"), f"{label} manifest")
    identity = measurement_identity(manifest)
    if not identity:
        raise ValueError(f"{label}: unverified measurement source; rerun with the paired NTK analyzer")
    seeds = manifest["seeds"]
    indices = manifest["sample_indices"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"{label}: expected unique nonempty probe seeds")
    if len(indices) < 2 or len(set(indices)) != len(indices):
        raise ValueError(f"{label}: expected at least two unique sample indices")
    for seed in seeds:
        _integer(seed, "Probe seed")
    for index in indices:
        _integer(index, "Sample index")
    if not manifest["sample_digest"]:
        raise ValueError(f"{label}: missing processed-sample fingerprint")
    if manifest["scope"] not in ("backbone", "prompts", "both"):
        raise ValueError(f"{label}: unknown measured scope {manifest['scope']!r}")
    measured_scopes = ("backbone", "prompts") if manifest["scope"] == "both" else (manifest["scope"],)
    if scope not in measured_scopes:
        raise ValueError(f"{label}: results do not contain the requested {scope} scope")
    groups = {f"{part}/{module}" for part in measured_scopes for module in COLORS}
    stages, stage_names = {}, {}
    reference_config = None
    for stage in manifest["stages"]:
        _required(stage, ("name", "total_step", "local_step", "config"), f"{label} stage")
        step = _integer(stage["total_step"], "Cumulative update")
        _integer(stage["local_step"], "Local update")
        name = stage["name"]
        if not isinstance(name, str) or not name or step in stages or name in stage_names:
            raise ValueError(f"{label}: duplicate or invalid stage name/cumulative update")
        _required(stage["config"], COMPARABLE_FIELDS, f"{label} {name} config")
        config = {field: stage["config"][field] for field in COMPARABLE_FIELDS}
        if reference_config is not None and config != reference_config:
            raise ValueError(f"{label}: model/probe configuration differs across checkpoints")
        reference_config = config
        stages[step] = stage
        stage_names[name] = step
    if not stages or any(step not in stages for step in steps):
        missing = [step for step in steps if step not in stages]
        raise ValueError(
            f"{label}: missing cumulative updates {missing}; use --steps for a partial comparison"
        )
    records, signatures, group_metadata = {}, None, None
    for record in document["records"]:
        _required(
            record,
            ("stage", "total_step", "local_step", "seed", "groups", "parameter_groups"),
            f"{label} record",
        )
        step = _integer(record["total_step"], "Record cumulative update")
        seed = _integer(record["seed"], "Record probe seed")
        if record["stage"] not in stage_names or stage_names[record["stage"]] != step:
            raise ValueError(f"{label}: record stage and cumulative update disagree")
        if record["local_step"] != stages[step]["local_step"]:
            raise ValueError(f"{label}: record local update differs from its stage")
        key = (step, seed)
        if key in records:
            raise ValueError(f"{label}: duplicate record for cumulative update {step}, probe seed {seed}")
        if set(record["groups"]) != groups or set(record["parameter_groups"]) != groups:
            raise ValueError(f"{label}: missing or unexpected parameter groups")
        current_metadata = {}
        for group, values in record["groups"].items():
            _required(values, (*METRICS, "parameter_count", "parameter_representation"), f"{label} {group}")
            count = _integer(values["parameter_count"], f"{group} parameter count", minimum=1)
            for metric in METRICS:
                value = values[metric]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{label}: {group} {metric} is not numeric")
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{label}: {group} {metric} must be finite and positive")
            if not 1 - 1e-6 <= values["effective_rank"] <= len(indices) + 1e-6:
                raise ValueError(f"{label}: effective rank is outside the sampled-kernel dimension")
            signature = record["parameter_groups"][group]
            names, expected_count = set(), 0
            for member in signature:
                if len(member) != 2 or not isinstance(member[0], str) or member[0] in names:
                    raise ValueError(f"{label}: malformed or duplicate parameter signature in {group}")
                names.add(member[0])
                for dimension in member[1]:
                    _integer(dimension, "Parameter dimension", minimum=1)
                expected_count += math.prod(member[1])
            if not signature or expected_count != count:
                raise ValueError(f"{label}: {group} parameter count disagrees with its signature")
            current_metadata[group] = {
                "parameter_count": count,
                "parameter_representation": values["parameter_representation"],
            }
        if signatures is not None and signatures != record["parameter_groups"]:
            raise ValueError(f"{label}: parameter signatures differ across stages or probe seeds")
        if group_metadata is not None and group_metadata != current_metadata:
            raise ValueError(f"{label}: parameter counts/representations differ across stages or probe seeds")
        signatures = record["parameter_groups"]
        group_metadata = current_metadata
        records[key] = record
    expected = {(step, seed) for step in stages for seed in seeds}
    if set(records) != expected:
        raise ValueError(f"{label}: incomplete or unexpected stage/probe records")
    return Measurements(
        Path(path), label, manifest, records, stages, signatures, group_metadata, identity, file_digest
    )


def load_results(path, label, steps=DEFAULT_STEPS, scope="backbone"):
    path = Path(path).expanduser().resolve()
    content = path.read_bytes()
    return validate_document(json.loads(content), path, label, steps, scope, hashlib.sha256(content).hexdigest())


def validate_pair(staged, control):
    """Permit relocated files and different schedules, while preserving the measurement protocol."""
    validate_measurement_pair(staged.manifest, control.manifest)
    a = next(iter(staged.stages.values()))["config"]
    b = next(iter(control.stages.values()))["config"]
    for field in COMPARABLE_FIELDS:
        if a[field] != b[field]:
            raise ValueError(f"Cannot compare: model/probe configuration mismatch for {field}")
    if staged.signatures != control.signatures or staged.group_metadata != control.group_metadata:
        raise ValueError("Cannot compare: parameter signatures, counts, or representations differ")


def point_rows(datasets, steps, scope):
    """Export exactly the observations placed on the axes, including their paired seed identifiers."""
    return [
        {
            "method": dataset.label,
            "stage": dataset.records[(step, seed)]["stage"],
            "total_step": step,
            "local_step": dataset.records[(step, seed)]["local_step"],
            "probe_seed": seed,
            "group": f"{scope}/{module}",
            **{
                field: dataset.records[(step, seed)]["groups"][f"{scope}/{module}"][field]
                for field in (*METRICS, "parameter_count")
            },
        }
        for dataset in datasets
        for step in steps
        for seed in dataset.manifest["seeds"]
        for module in COLORS
    ]


def make_figure(datasets, steps, scope="backbone", y_scale="log", show_means=False, font_path=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager
    from matplotlib.lines import Line2D

    if font_path:
        path = Path(font_path).expanduser().resolve()
        font_manager.fontManager.addfont(str(path))
        font_family = font_manager.FontProperties(fname=str(path)).get_name()
    else:
        try:
            font_manager.findfont("Times New Roman", fallback_to_default=False)
            font_family = "Times New Roman"
        except ValueError:
            font_family = "DejaVu Serif"
            warnings.warn("Times New Roman unavailable; use --font-path to select it. Using DejaVu Serif.")
    style = {
        "font.family": font_family,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.edgecolor": "#AAB3BA",
        "axes.labelcolor": "#34414B",
        "text.color": "#34414B",
        "xtick.color": "#66747D",
        "ytick.color": "#66747D",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            len(datasets),
            len(steps),
            figsize=(3.0 * len(steps), 2.8 * len(datasets) + 1.05),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        rows = point_rows(datasets, steps, scope)
        energies = [row["parameter_normalized_energy"] for row in rows]
        if y_scale == "log":
            low, high = np.log10([min(energies), max(energies)])
            margin = max((high - low) * 0.12, 0.1)
            limits = (10 ** (low - margin), 10 ** (high + margin))
        else:
            limits = (0, max(energies) * 1.15)
        for row_index, dataset in enumerate(datasets):
            for column, step in enumerate(steps):
                axis = axes[row_index, column]
                for module, color in COLORS.items():
                    points = np.asarray(
                        [
                            [
                                dataset.records[(step, seed)]["groups"][f"{scope}/{module}"][metric]
                                for metric in METRICS
                            ]
                            for seed in dataset.manifest["seeds"]
                        ]
                    )
                    axis.scatter(
                        points[:, 0],
                        points[:, 1],
                        s=42,
                        c=color,
                        alpha=0.8,
                        edgecolors="white",
                        linewidths=0.55,
                        label=LABELS[module],
                        zorder=3,
                    )
                    if show_means:
                        axis.scatter(
                            *points.mean(axis=0),
                            marker="D",
                            s=65,
                            facecolors="none",
                            edgecolors=color,
                            linewidths=1.5,
                            zorder=4,
                        )
                axis.set_yscale(y_scale)
                axis.set_xlim(0, len(dataset.manifest["sample_indices"]))
                axis.set_ylim(limits)
                axis.spines[["top", "right"]].set_visible(False)
                axis.grid(axis="y", color="#E7EBEE", linewidth=0.65)
                axis.tick_params(axis="both", which="major", labelsize=9, length=3)
                if row_index == 0:
                    axis.set_title(f"{step:,} updates", pad=12)
                if column == 0:
                    axis.text(
                        0.025,
                        0.975,
                        dataset.label,
                        transform=axis.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                        fontweight="bold",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
                    )
        legend = [
            Line2D([], [], linestyle="none", marker="o", color=color, markersize=6, label=LABELS[module])
            for module, color in COLORS.items()
        ]
        if show_means:
            legend.append(
                Line2D(
                    [], [], linestyle="none", marker="D", markerfacecolor="none",
                    color="#66747D", markersize=6, label="Arithmetic mean",
                )
            )
        fig.legend(
            handles=legend, loc="upper center", ncol=len(legend), frameon=False, bbox_to_anchor=(0.54, 1.0)
        )
        fig.supxlabel("Spectral effective rank", y=0.065)
        fig.supylabel("Parameter-normalized tangent energy", x=0.015)
        fig.text(
            0.54,
            0.012,
            f"Each dot: one of {len(datasets[0].manifest['seeds'])} paired probe seeds; "
            "probe variation, not training-run variation.",
            ha="center",
            fontsize=8,
            color="#66747D",
        )
        fig.subplots_adjust(left=0.095, right=0.985, top=0.86, bottom=0.19, wspace=0.16, hspace=0.40)
    return fig, axes, font_family


def plot_comparison(
    control_results,
    output_dir,
    staged_results=None,
    steps=DEFAULT_STEPS,
    scope="backbone",
    y_scale="log",
    show_means=False,
    font_path=None,
    staged_label="Three-stage prompt learning",
    control_label="Plain prompt learning",
    dpi=300,
):
    """Write PNG/PDF/SVG, exact point CSV, and provenance; optional second method."""
    args = SimpleNamespace(
        control_results=control_results,
        output_dir=output_dir,
        staged_results=staged_results,
        steps=steps,
        scope=scope,
        y_scale=y_scale,
        show_means=show_means,
        font_path=font_path,
        staged_label=staged_label,
        control_label=control_label,
        dpi=dpi,
    )
    datasets = []
    if args.staged_results:
        datasets.append(load_results(args.staged_results, args.staged_label, args.steps, args.scope))
    datasets.append(load_results(args.control_results, args.control_label, args.steps, args.scope))
    if len(datasets) == 2:
        validate_pair(*datasets)
    rows = point_rows(datasets, args.steps, args.scope)
    fig, axes, font_family = make_figure(
        datasets, args.steps, args.scope, args.y_scale, args.show_means, args.font_path
    )
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / "ntk_prompt_comparison"
    import matplotlib.pyplot as plt

    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        for extension in ("png", "pdf", "svg"):
            fig.savefig(prefix.with_suffix(f".{extension}"), dpi=args.dpi, bbox_inches="tight")
    with (out / "ntk_prompt_comparison_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema": 1,
        "description": "Saved NTK observations, aligned by cumulative updates; no jitter or interpolation.",
        "scope": args.scope,
        "steps": list(args.steps),
        "y_scale": args.y_scale,
        "x_limits": list(axes[0, 0].get_xlim()),
        "y_limits": list(axes[0, 0].get_ylim()),
        "font_family": font_family,
        "show_arithmetic_means": args.show_means,
        "plotted_points": len(rows),
        "probe_seeds": datasets[0].manifest["seeds"],
        "uncertainty": "Probe/flow-noise replicates within each training run; not independent training runs.",
        "inputs": [
            {
                "method": dataset.label,
                "path": str(dataset.path),
                "sha256": dataset.file_digest,
                "measurement_digest": dataset.measurement_digest,
                "source_digest": dataset.manifest.get("source_digest"),
                "sample_digest": dataset.manifest["sample_digest"],
                "stage_names": {step: dataset.stages[step]["name"] for step in args.steps},
            }
            for dataset in datasets
        ],
    }
    (out / "ntk_prompt_comparison_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plt.close(fig)
    return out


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged-results", type=Path, help="Three-stage results.json; omit for control only")
    parser.add_argument(
        "--control-results", type=Path, required=True, help="Plain prompt-learning results.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--steps", type=parse_steps, default=DEFAULT_STEPS,
        help="Cumulative updates, default 0,750,1000,4000; use 0,750,1000 while training",
    )
    parser.add_argument("--scope", choices=("backbone", "prompts"), default="backbone")
    parser.add_argument("--staged-label", default="Three-stage prompt learning")
    parser.add_argument("--control-label", default="Plain prompt learning")
    parser.add_argument("--y-scale", choices=("log", "linear"), default="log")
    parser.add_argument(
        "--show-means", action="store_true", help="Overlay hollow diamonds for arithmetic means"
    )
    parser.add_argument("--font-path", type=Path, help="Optional font file, e.g. /path/to/times.ttf")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.staged_label == args.control_label and args.staged_results:
        parser.error("Method labels must differ")
    try:
        out = plot_comparison(**vars(args))
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Saved NTK scatter figures, observed points, and provenance to {out}")


if __name__ == "__main__":
    main()
