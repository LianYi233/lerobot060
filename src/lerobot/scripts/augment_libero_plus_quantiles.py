#!/usr/bin/env python

"""Add PI0.5-required quantile statistics to a local LIBERO-Plus dataset.

This utility is deliberately narrower and faster than the generic
``augment_dataset_quantile_stats.py``:

* reads only ``observation.state`` and ``action`` from the parquet files;
* never decodes videos;
* updates only the local ``meta/stats.json``;
* never pushes or tags anything on the Hugging Face Hub.

For ~2.24M LIBERO-Plus frames the two numeric arrays are small enough to keep
in host RAM while computing exact NumPy quantiles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

REQUIRED_FEATURES = ("observation.state", "action")
QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


def _has_required_quantiles(stats: dict) -> bool:
    for feature in REQUIRED_FEATURES:
        feature_stats = stats.get(feature, {})
        if "q01" not in feature_stats or "q99" not in feature_stats:
            return False
    return True


def _column_to_numpy(table, name: str) -> np.ndarray:
    """Convert a fixed-size/list parquet column into a dense float32 array."""
    values = table[name].combine_chunks().to_pylist()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    return array


def augment(root: Path, *, overwrite: bool = False) -> None:
    stats_path = root / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot stats file: {stats_path}")

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    if not overwrite and _has_required_quantiles(stats):
        print("LIBERO-Plus stats already contain q01/q99 for observation.state and action.")
        return

    parquet_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    chunks: dict[str, list[np.ndarray]] = {feature: [] for feature in REQUIRED_FEATURES}
    total_rows = 0
    print(f"Reading {len(parquet_files)} parquet files (state/action only; videos are not decoded)...")
    for path in tqdm(parquet_files, desc="Reading parquet"):
        schema_names = set(pq.read_schema(path).names)
        missing = [feature for feature in REQUIRED_FEATURES if feature not in schema_names]
        if missing:
            raise KeyError(f"{path} is missing required columns: {missing}")
        table = pq.read_table(path, columns=list(REQUIRED_FEATURES))
        total_rows += table.num_rows
        for feature in REQUIRED_FEATURES:
            chunks[feature].append(_column_to_numpy(table, feature))

    print(f"Loaded {total_rows:,} frames. Computing exact quantiles...")
    for feature in REQUIRED_FEATURES:
        values = np.concatenate(chunks[feature], axis=0)
        feature_stats = stats.setdefault(feature, {})
        for key, q in QUANTILES.items():
            feature_stats[key] = np.quantile(values, q, axis=0).astype(np.float32).tolist()
        print(
            f"  {feature}: shape={values.shape}, "
            f"q01={feature_stats['q01']}, q99={feature_stats['q99']}"
        )

    backup_path = stats_path.with_suffix(".json.before_quantiles")
    if not backup_path.exists():
        backup_path.write_text(stats_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backup written to {backup_path}")

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        f.write("\n")

    if not _has_required_quantiles(stats):
        raise RuntimeError("Quantile augmentation completed but required q01/q99 values are still missing")
    print(f"Updated local stats: {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    augment(args.root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
