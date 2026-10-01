#!/usr/bin/env python3
"""Read-only Piper checkpoint/runtime audit. Never loads policy weights or robot drivers."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = (
    "src/lerobot/policies/pi05/modeling_pi05.py",
    "src/lerobot/policies/pi05/configuration_pi05.py",
    "src/lerobot/policies/pi05/processor_pi05.py",
    "src/lerobot/policies/pi_gemma.py",
    "src/lerobot/processor/normalize_processor.py",
    "deploy_piper_wyn.py",
    "deploy_piper_vlaa.py",
    "piper_deploy_guard.py",
    "src/lerobot/robots/piper/piper.py",
    "src/lerobot/robots/piper/configuration_piper.py",
)


def fingerprint(path):
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_notices(version):
    notices = [
        "Import/key-loading success does not verify training/deployment forward equivalence. "
        "Match the checkpoint's training environment before comparing offline actions."
    ]
    if version == "5.3.0":
        notices.append(
            "Transformers 5.3.0 has different PaliGemma image and Gemma text embedding scales "
            "from 5.5.4. For checkpoints trained with 5.5.4, use transformers==5.5.4 and "
            "rerun the probe and offline action evaluation. This does not establish the "
            "training version of an unknown checkpoint."
        )
    return notices


def transformers_sources():
    """Inspect installed source without importing Transformers or constructing models."""
    try:
        distribution = importlib.metadata.distribution("transformers")
    except importlib.metadata.PackageNotFoundError:
        return {}
    results = {}
    selectors = {
        "paligemma": {("PaliGemmaModel", "get_image_features")},
        "gemma": {("GemmaModel", "__init__"), ("GemmaTextScaledWordEmbedding", "forward")},
    }
    for model, methods in selectors.items():
        path = Path(distribution.locate_file(f"transformers/models/{model}/modeling_{model}.py"))
        if not path.is_file():
            results[model] = {"source_missing": True}
            continue
        source = path.read_text(encoding="utf-8")
        selected = {}
        for cls in ast.parse(source).body:
            if not isinstance(cls, ast.ClassDef):
                continue
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and (cls.name, node.name) in methods:
                    selected[f"{cls.name}.{node.name}"] = ast.get_source_segment(source, node)
        results[model] = {"sha256": fingerprint(path), "methods": selected}
    return results


def probe_embeddings():
    """CPU-only probe of HF primitives used by PiGemma; not full PI05 inference.

    A 16-wide, one-layer Gemma measures the embedding module's scale. A stub
    vision tower/projector returning ones isolates get_image_features' scale.
    These objects use no checkpoint, tokenizer, downloads, camera or CAN.
    """
    import inspect
    from types import SimpleNamespace

    import torch
    from transformers import GemmaConfig
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from transformers.models.gemma.modeling_gemma import GemmaModel
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaModel

    width = 16
    cfg = GemmaConfig(
        vocab_size=8,
        hidden_size=width,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    with torch.device("cpu"), torch.random.fork_rng(devices=[]), torch.inference_mode():
        model = GemmaModel(cfg).eval()
        embedding = model.get_input_embeddings()
        embedding.weight.fill_(1.0)
        text_value = embedding(torch.tensor([[1]]))
        stub = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=width)),
            vision_tower=lambda pixels, **kwargs: BaseModelOutputWithPooling(last_hidden_state=pixels),
            multi_modal_projector=lambda pixels: pixels,
        )
        image_value = inspect.unwrap(PaliGemmaModel.get_image_features)(
            stub, torch.ones(1, 2, width)
        ).pooler_output
        return {
            "scope": "HF primitives on CPU; not full policy, weights, or task success",
            "hidden_size": width,
            "image_scale_from_unit_projector_output": image_value.mean().item(),
            "text_scale_from_unit_embedding_weights": text_value.mean().item(),
            "embedding_class": type(embedding).__name__,
        }


def checkpoint_report(path):
    config = read_json(path / "config.json")
    if config is None:
        raise ValueError(f"Missing config.json: {path}")
    training = read_json(path / "train_config.json")
    dataset = training.get("dataset", {}) if training is not None else {}
    episodes = dataset.get("episodes")
    scope = (
        "unknown: train_config.json missing"
        if training is None
        else "all episodes before eval_split"
        if episodes is None
        else "explicit episode subset"
    )
    files = {"config.json": fingerprint(path / "config.json")}
    processors = {}
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        data = read_json(path / name)
        files[name] = fingerprint(path / name)
        processors[name] = data
        for step in (data or {}).get("steps", []):
            if state_file := step.get("state_file"):
                state_path = (path / state_file).resolve()
                if not state_path.is_relative_to(path.resolve()):
                    raise ValueError(f"Processor state_file escapes checkpoint: {state_file}")
                files[state_file] = fingerprint(state_path)
    return {
        "path": str(path),
        "policy": config,
        "training": {
            "train_config_present": training is not None,
            "dataset_repo_id": dataset.get("repo_id"),
            "dataset_root": dataset.get("root"),
            "episodes": episodes,
            "episode_scope": scope,
            "eval_split": dataset.get("eval_split"),
            "configured_flow_steps": (training or {}).get("steps"),
            "per_process_batch_size": (training or {}).get("batch_size"),
        },
        "saved_step": read_json(path.parent / "training_state/training_step.json"),
        "processors": processors,
        "small_file_sha256": files,
        "weights_verified": False,
    }


def compare_reports(current, other):
    """Differences are diagnostic evidence, never an automatic compatibility approval."""
    differences = []
    for section in ("versions", "repo_sources"):
        left, right = other.get(section, {}), current.get(section, {})
        for key in sorted(left.keys() | right.keys()):
            if left.get(key) != right.get(key):
                differences.append(f"{section}/{key}")
    for section in ("transformers_sources", "embedding_probe"):
        if current.get(section) != other.get(section):
            differences.append(section)
    left = other.get("checkpoint", {}).get("small_file_sha256", {})
    right = current.get("checkpoint", {}).get("small_file_sha256", {})
    for key in sorted(left.keys() | right.keys()):
        if left.get(key) != right.get(key):
            differences.append(f"checkpoint/{key}")
    return {
        "different_fields": differences,
        "interpretation": "No differences does not verify weights, camera geometry, control, or task success.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_path", type=Path, required=True)
    parser.add_argument("--dataset_info", type=Path)
    parser.add_argument(
        "--probe_embeddings", action="store_true", help="Optional tiny CPU Transformers probe"
    )
    parser.add_argument("--compare", type=Path, help="Report from the other machine")
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON file; existing files are not replaced"
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Output exists; choose a new name: {args.output}")
    report = {
        "schema_version": 1,
        "runtime_scope": "Current environment only; not proof of historical training environment",
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "versions": {
            name: installed_version(name)
            for name in ("torch", "transformers", "numpy", "safetensors", "peft", "accelerate")
        },
        "repo_sources": {name: fingerprint(ROOT / name) for name in SOURCE_FILES},
        "transformers_sources": transformers_sources(),
        "checkpoint": checkpoint_report(args.policy_path.expanduser().resolve()),
    }
    report["runtime_notices"] = runtime_notices(report["versions"]["transformers"])
    if args.dataset_info:
        info = read_json(args.dataset_info.expanduser())
        if info is None:
            parser.error(f"Dataset info missing: {args.dataset_info}")
        report["dataset_info"] = {k: info.get(k) for k in ("fps", "total_episodes", "robot_type", "features")}
    if args.probe_embeddings:
        try:
            report["embedding_probe"] = probe_embeddings()
        except Exception as exc:
            # Keep metadata useful when old/mismatched dependencies cannot import.
            report["embedding_probe"] = {"error": f"{type(exc).__name__}: {exc}"}
    if args.compare:
        previous = read_json(args.compare.expanduser())
        if previous is None:
            parser.error(f"Comparison report missing: {args.compare}")
        report["comparison"] = compare_reports(report, previous)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print("Transformers:", report["versions"]["transformers"])
    for notice in report["runtime_notices"]:
        print("RUNTIME_NOTICE:", notice)
    print("Training selection:", json.dumps(report["checkpoint"]["training"], ensure_ascii=False))
    if "embedding_probe" in report:
        print("Embedding probe:", json.dumps(report["embedding_probe"], ensure_ascii=False))
    if "comparison" in report:
        print("Comparison:", json.dumps(report["comparison"], ensure_ascii=False))
    print(f"AUDIT_SAVED: {args.output}; no policy weights or robot hardware loaded")


if __name__ == "__main__":
    main()
