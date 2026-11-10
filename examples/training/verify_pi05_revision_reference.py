#!/usr/bin/env python
"""Read-only parity check against a saved revision train_config.json (stdlib only).

Path equality proves which local assets were requested, not that historical
weights or dataset contents have never changed. Keep original assets immutable.
"""

import argparse
import json
from pathlib import Path


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def local_directory(value: str, label: str) -> Path:
    if not value or not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute local directory, got {value!r}")
    path = Path(value).resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")
    return path


def reference_recipe(reference: Path) -> tuple[dict, Path, int]:
    cfg = read_config(reference)
    policy = cfg.get("policy") or {}
    if policy.get("type") != "pi05" or policy.get("training_stage", "flow") != "flow":
        raise ValueError("Reference must be a PI05 formal-flow train_config.json (not a Stage-1-only run)")
    original_base = local_directory(policy.get("pretrained_path"), "Reference pretrained_path")
    total = int(cfg["steps"])
    transfer_config = original_base / "train_config.json"
    transfer = read_config(transfer_config) if transfer_config.is_file() else None
    if transfer is not None and (transfer.get("policy") or {}).get("training_stage") == "next_action":
        # Integrated training changes the final config's base to its own primed
        # transfer. Recover the original source; never initialize LoRA from SAP.
        if (
            transfer.get("dataset") != cfg.get("dataset")
            or transfer.get("batch_size") != cfg.get("batch_size")
        ):
            raise ValueError(
                "Reference preparation and flow did not use the same dataset/batch configuration"
            )
        total += int(transfer["steps"])
        original_base = local_directory(transfer["policy"].get("pretrained_path"), "Original Stage-1 base")
        print(f"reference-check: recovered original initialization via {transfer_config}")
    elif any(part.endswith("_next_action_pretrain") for part in original_base.parts):
        raise ValueError("Cannot recover original initialization: keep the Stage-1 train_config.json")
    else:
        # Also support an original (pre-launch) config, before its path was
        # changed by integrated training. Completed checkpoints take the branch above.
        total += int(policy.get("next_action_pretrain_steps", 0))
    return cfg, original_base, total


def verify(args: argparse.Namespace) -> None:
    reference = args.reference.resolve(strict=True)
    cfg, original_base, total = reference_recipe(reference)
    policy = cfg["policy"]
    dataset = cfg["dataset"]
    state_path = reference.parent.parent / "training_state" / "training_step.json"
    state = read_config(state_path) if state_path.is_file() else {}
    recorded_processes = state.get("num_processes")
    if recorded_processes is not None:
        reference_processes = int(recorded_processes)
        if args.reference_num_processes and args.reference_num_processes != reference_processes:
            raise ValueError("REFERENCE_NUM_PROCESSES conflicts with saved training_state")
    elif args.reference_num_processes:
        reference_processes = args.reference_num_processes
        print("reference-check: reference world size supplied explicitly, not verified from a checkpoint")
    else:
        raise ValueError(
            "Reference world size is not recorded. Set REFERENCE_NUM_PROCESSES to the verified original "
            "process count; do not infer it from the number of GPUs currently free."
        )
    if min(args.batch_size, args.num_processes, reference_processes) < 1:
        raise ValueError("Batch sizes and process counts must be positive")
    if state.get("batch_size") is not None and state["batch_size"] != cfg["batch_size"]:
        raise ValueError("Reference saved training_state and config disagree on batch size")

    comparisons = {
        "original_base": (original_base, local_directory(args.pretrained, "LoRA base")),
        "dataset.root": (
            local_directory(dataset.get("root"), "Reference dataset"),
            local_directory(args.dataset, "LoRA dataset"),
        ),
        "dataset.repo_id": (dataset.get("repo_id"), args.repo_id),
        "tokenizer": (
            local_directory(policy.get("tokenizer_name"), "Reference tokenizer"),
            local_directory(args.tokenizer, "LoRA tokenizer"),
        ),
        "total_updates": (total, args.total_updates),
        "effective_batch": (
            int(cfg["batch_size"]) * reference_processes,
            args.batch_size * args.num_processes,
        ),
        "dtype": (policy.get("dtype", "float32"), args.dtype),
        "dataset.video_backend": (dataset.get("video_backend"), args.video_backend),
        "dataset.episodes": (dataset.get("episodes"), None),
        "dataset.eval_split": (dataset.get("eval_split", 0.0), 0.0),
        "dataset.streaming": (dataset.get("streaming", False), False),
        "dataset.image_transforms.enable": (
            (dataset.get("image_transforms") or {}).get("enable", False),
            False,
        ),
        "chunk_size": (policy.get("chunk_size", 50), 50),
        "n_action_steps": (policy.get("n_action_steps", 50), 50),
        "num_inference_steps": (policy.get("num_inference_steps", 10), 10),
        "sample_weighting": (cfg.get("sample_weighting"), None),
        "rename_map": (cfg.get("rename_map", {}), {}),
    }
    failures = []
    for name, (reference_value, lora_value) in comparisons.items():
        if reference_value != lora_value:
            failures.append(f"{name}: reference={reference_value!r}, requested LoRA={lora_value!r}")
    if failures:
        raise ValueError("Reference protocol mismatch:\n  " + "\n  ".join(failures))
    print(
        f"reference-check: PASS {len(comparisons)} recorded-setting checks; "
        f"base={original_base}, updates={total}, effective_batch={args.batch_size * args.num_processes}"
    )
    print(
        "reference-check: not a historical content-hash, AMP, exact-sample-order or equal-GPU-hours guarantee"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--num-processes", required=True, type=int)
    parser.add_argument("--reference-num-processes", default=0, type=int)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--video-backend", required=True)
    parser.add_argument("--total-updates", required=True, type=int)
    args = parser.parse_args()
    try:
        verify(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"reference-check: ERROR: {exc}\n")


if __name__ == "__main__":
    main()
