#!/usr/bin/env python3
"""Summarize adaptation loss from complete, contiguous diagnostic log windows.

This read-only helper uses only the standard library. ``train.loss`` is already
an average over ``window_updates`` trainer updates, not a per-update observation.
The trailing mean therefore includes the shortest suffix of *whole* log windows
covering at least the requested update count. It never invents a partial-window
mean. For example, a 500-update request may cover 525 actual updates.

Only ``directory/training_diagnostics.jsonl`` is read. Pass the main adaptation
output directory, not a preparation directory; no directories are searched or
combined. Steps, including ``first_target_step``, are local to that supplied log.
The caller can add preparation updates when reporting cumulative compute.

``stable_target_met`` means the trailing weighted mean is at most ``target``,
with sufficient contiguous coverage and finite losses throughout that window.
It is a training-loss measurement, not a claim about validation or robot success,
and it never stops training. ``first_target_step`` is the first such observation
at a log boundary in the *current contiguous segment*. Gaps, conflicting overlap,
rewinds and malformed records reset that segment and its first-target result.
Adjacent identical loss-window records are ignored rather than counted twice.
"""

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import NamedTuple

SCHEMA = "piper-revision-loss-v1"


class _Window(NamedTuple):
    start: int  # Exclusive: the updates covered are start + 1 through step.
    step: int
    updates: int
    logged_loss: float | None
    usable_loss: float | None
    stage: object


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def _parse_window(record):
    if not isinstance(record, dict):
        raise ValueError("diagnostic record must be an object")
    step, updates = record.get("step"), record.get("window_updates")
    if type(step) is not int or type(updates) is not int or not 0 < updates <= step:
        raise ValueError("step and window_updates must be positive integers, with updates <= step")
    train = record.get("train")
    loss = _finite_number(train.get("loss")) if isinstance(train, dict) else None
    if loss is not None and loss < 0:
        loss = None  # The monitored flow objective is an MSE, so negative values are invalid.
    usable_loss = loss
    diagnostics = record.get("diagnostics_rank0")
    policy_loss = diagnostics.get("loss") if isinstance(diagnostics, dict) else None
    if isinstance(policy_loss, dict) and "nonfinite" in policy_loss:
        count = policy_loss["nonfinite"]
        # Even a finite aggregate cannot certify a window with reported nonfinite losses.
        if type(count) is not int or count != 0:
            usable_loss = None
    return _Window(step - updates, step, updates, loss, usable_loss, record.get("training_stage"))


def _weighted_mean(windows, updates):
    if not windows or any(window.usable_loss is None for window in windows):
        return None
    # Normalize before summing so a large but finite recorded loss cannot overflow
    # merely because its window contains many updates.
    scale = max(abs(window.usable_loss) for window in windows)
    if scale == 0:
        return 0.0
    value = math.fsum(
        (window.usable_loss / scale) * (window.updates / updates) for window in windows
    ) * scale
    return value if math.isfinite(value) else None


def summarize_loss(directory: Path, target=1e-4, window_updates=500) -> dict:
    """Return a strict-JSON-compatible snapshot; missing logs are a normal result.

    ``window_updates`` is the requested coverage; ``actual_window_updates`` is
    the full-window coverage actually used. ``coverage_complete`` only checks
    update coverage; ``window_all_finite`` additionally checks the loss data.
    A truncated final JSON line is ignored, retaining the latest complete record.
    Invalid complete records break continuity so missing data cannot be hidden.

    No optimizer, model, torch import, recursive search, file write or network
    access is performed. A restarted log cannot inherit a previous segment's
    first-target achievement. An ordinary resume with contiguous steps can.
    """
    target = _finite_number(target)
    if target is None or target <= 0:
        raise ValueError("target must be a finite positive number")
    if type(window_updates) is not int or window_updates <= 0:
        raise ValueError("window_updates must be a positive integer")

    path = Path(directory) / "training_diagnostics.jsonl"
    result = {
        "schema": SCHEMA,
        "diagnostics_path": str(path),
        "status": "empty_log",
        "target": target,
        "window_updates": window_updates,
        "last_logged_step": None,
        "latest_logged_loss": None,
        "latest_window_updates": None,
        "trailing_loss": None,
        "actual_window_updates": 0,
        "trailing_start_step": None,
        "trailing_end_step": None,
        "trailing_log_windows": 0,
        "coverage_complete": False,
        "window_all_finite": False,
        "stable_target_met": False,
        "first_target_step": None,
        "first_target_trailing_loss": None,
        "first_target_actual_window_updates": None,
        "first_target_scope": "current_contiguous_segment",
        "current_segment_start_step": None,
        "current_segment_updates": 0,
        "segments": 0,
        "accepted_log_windows": 0,
        "duplicate_records": 0,
        "restart_or_overlap_records": 0,
        "gap_records": 0,
        "stage_changes": 0,
        "malformed_records": 0,
        "invalid_loss_windows": 0,
        "ignored_incomplete_last_line": False,
        "warnings": [],
    }
    windows = deque()
    previous = None
    covered = 0

    def reset_segment():
        nonlocal previous, covered
        windows.clear()
        previous, covered = None, 0
        result.update({
            "first_target_step": None,
            "first_target_trailing_loss": None,
            "first_target_actual_window_updates": None,
            "current_segment_start_step": None,
            "current_segment_updates": 0,
        })

    try:
        stream = path.open("rb")
    except FileNotFoundError:
        result["status"] = "missing_log"
        result["warnings"].append("No training_diagnostics.jsonl exists in the supplied directory.")
        return result
    except OSError as error:
        result["status"] = "read_error"
        result["warnings"].append(f"Cannot read diagnostics: {error}")
        return result

    with stream:
        for raw in stream:
            if not raw.strip():
                continue
            try:
                record = json.loads(raw.decode("utf-8"))
            except ValueError:
                if not raw.endswith(b"\n"):
                    result["ignored_incomplete_last_line"] = True
                    break
                result["malformed_records"] += 1
                reset_segment()
                continue
            try:
                window = _parse_window(record)
            except ValueError:
                result["malformed_records"] += 1
                reset_segment()
                continue

            if window == previous:
                result["duplicate_records"] += 1
                continue
            if previous is not None:
                reason = None
                if window.step <= previous.step or window.start < previous.step:
                    reason = "restart_or_overlap_records"
                elif window.start > previous.step:
                    reason = "gap_records"
                elif window.stage != previous.stage:
                    reason = "stage_changes"
                if reason:
                    result[reason] += 1
                    reset_segment()
            if previous is None:
                result["segments"] += 1
                result["current_segment_start_step"] = window.start + 1

            previous = window
            result["accepted_log_windows"] += 1
            result["current_segment_updates"] += window.updates
            result["last_logged_step"] = window.step
            result["latest_logged_loss"] = window.logged_loss
            result["latest_window_updates"] = window.updates
            if window.usable_loss is None:
                result["invalid_loss_windows"] += 1
            windows.append(window)
            covered += window.updates
            while covered - windows[0].updates >= window_updates:
                covered -= windows.popleft().updates
            mean = _weighted_mean(windows, covered)
            if (
                result["first_target_step"] is None
                and covered >= window_updates
                and mean is not None
                and mean <= target
            ):
                result["first_target_step"] = window.step
                result["first_target_trailing_loss"] = mean
                result["first_target_actual_window_updates"] = covered

    result["status"] = "ok" if result["accepted_log_windows"] else "empty_log"
    if result["malformed_records"] and not result["accepted_log_windows"]:
        result["status"] = "no_valid_windows"
    mean = _weighted_mean(windows, covered)
    result.update({
        "trailing_loss": mean,
        "actual_window_updates": covered,
        "trailing_start_step": windows[0].start + 1 if windows else None,
        "trailing_end_step": windows[-1].step if windows else None,
        "trailing_log_windows": len(windows),
        "coverage_complete": covered >= window_updates,
        "window_all_finite": mean is not None,
        "stable_target_met": covered >= window_updates and mean is not None and mean <= target,
    })
    warnings = result["warnings"]
    if result["ignored_incomplete_last_line"]:
        warnings.append("Ignored an incomplete final line; statistics end at the last complete log record.")
    if result["duplicate_records"]:
        warnings.append(f"Ignored {result['duplicate_records']} adjacent identical loss-window records.")
    for key, description in (
        ("restart_or_overlap_records", "rewinds or overlapping/conflicting log windows"),
        ("gap_records", "gaps between recorded update windows"),
        ("stage_changes", "training-stage changes"),
        ("malformed_records", "malformed complete records"),
    ):
        if result[key]:
            warnings.append(f"Reset contiguous coverage after {result[key]} {description}.")
    if result["invalid_loss_windows"]:
        warnings.append(
            f"Found {result['invalid_loss_windows']} windows with missing, nonfinite or negative loss data; "
            "any trailing window containing them cannot meet the target."
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Main adaptation output directory")
    parser.add_argument("--target", type=float, default=1e-4)
    parser.add_argument("--window-updates", type=int, default=500)
    args = parser.parse_args(argv)
    try:
        result = summarize_loss(args.directory, args.target, args.window_updates)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
