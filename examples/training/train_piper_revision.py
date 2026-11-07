#!/usr/bin/env python
"""Matched-budget Piper controls, sequential training, provenance and compact exports.

Planning/status use only the standard library. This entry point never connects to a robot.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import select
import shlex
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_pi05_real import TASKS, check_dataset  # noqa: E402

GIB = 1024**3
CORE = ("priming", "direct_cpc", "flow_warmup", "priming_no_cpc")
LR_SWEEP = ("fixed_lr_0p5", "fixed_lr_0p1", "fixed_lr_0p03", "fixed_lr_0p01")
GROUPS = {
    "core": CORE,
    "extended": ("direct_no_cpc", "no_bridge"),
    "cpc_sweep": LR_SWEEP,
    "all": (*CORE, "direct_no_cpc", "no_bridge", *LR_SWEEP),
}
SCHEMA = "primingvla-piper-revision-v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing a linked output file: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def lock_json_contract(path, value):
    """Keep later ablations on the exact data/protocol accepted by earlier runs."""
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f"Existing experiment contract differs: {path}; use a new RUN_GROUP")
    else:
        write_json(path, value)


def verify_completed(record, storage, storage_sha):
    """Check small archives against their sealed manifests, hashing the shared base once."""
    identity = read_json(storage / "identity.json")
    if storage_sha != identity["model_sha256"]:
        raise ValueError("Shared reconstruction base is missing/corrupt; completed runs cannot be skipped")
    checkpoints = record.get("checkpoints", [])
    if not checkpoints:
        raise ValueError("Completed record has no checkpoint bundles")
    for item in checkpoints:
        bundle = Path(item["bundle"])
        manifest_path = bundle / "manifest.json"
        if sha256_file(manifest_path) != item["bundle_manifest_sha256"]:
            raise ValueError(f"Changed checkpoint bundle manifest: {bundle}")
        manifest = read_json(manifest_path)
        if manifest["base_model_sha256"] != storage_sha or manifest["source_model_sha256"] != item["model_sha256"]:
            raise ValueError(f"Checkpoint model identity mismatch: {bundle}")
        for relative, expected in manifest["files"].items():
            path = bundle / relative
            if path.is_symlink() or not path.resolve().is_relative_to(bundle.resolve()) or not path.is_file():
                raise ValueError(f"Missing or unsafe checkpoint bundle file: {path}")
            if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
                raise ValueError(f"Corrupt checkpoint bundle file: {path}")


def csv_values(value, allowed=None):
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items) or len(set(items)) != len(items):
        raise ValueError(f"Expected distinct comma-separated values, got {value!r}")
    if allowed is not None and any(item not in allowed for item in items):
        raise ValueError(f"Choose from {list(allowed)}, got {items}")
    return items


def env_int(env, key, default, minimum=1):
    value = int(env.get(key, default))
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value


def env_bool(env, key, default):
    value = env.get(key, str(default).lower()).lower()
    if value not in ("true", "false"):
        raise ValueError(f"{key} must be true or false")
    return value == "true"


def recipe(method, total=4000):
    """The bridge belongs to the 1000-update preparation stage, not an extra 250."""
    if total <= 1000:
        raise ValueError("TOTAL_UPDATES must exceed the 1000 preparation updates")
    if method not in GROUPS["all"]:
        raise ValueError(f"Unknown method: {method}")
    direct = method in ("direct_cpc", "direct_no_cpc")
    prep = 0 if direct else 1000
    bridge = 0 if direct or method == "no_bridge" else 250
    return {
        "method": method,
        "preparation_updates": prep,
        "action_only_updates": prep - bridge,
        "bridge_updates": bridge,
        "flow_updates": total - prep,
        "cumulative_updates": total,
        "cpc": method in ("priming", "direct_cpc", "flow_warmup", "no_bridge"),
        "action_prompt_lr_scale": float(method.removeprefix("fixed_lr_").replace("p", "."))
        if method.startswith("fixed_lr_") else 1.0,
        "stage1_objective": "unconditioned_full_mask_flow" if method == "flow_warmup" else "masked_action_inpainting",
        "masked_steps": 50 if method == "flow_warmup" else 40,
        "full_mask_probability": 1.0 if method == "flow_warmup" else 0.0,
    }


def milestones(env, total):
    value = json.loads(env.get("CUMULATIVE_CHECKPOINTS", json.dumps([step for step in (3000, total) if step <= total])))
    if not isinstance(value, list) or any(type(step) is not int or not 1000 < step <= total for step in value):
        raise ValueError("CUMULATIVE_CHECKPOINTS must be a JSON integer list in (1000, TOTAL_UPDATES]")
    return sorted(set([*value, total]))


def paths_from_env(env):
    work = Path(env.get("WORK_ROOT", "/root/autodl-tmp/piper-revision")).expanduser().resolve()
    group = env.get("RUN_GROUP", "review-v1")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", group):
        raise ValueError("RUN_GROUP must be a single filename-safe label")
    return {
        "work": work,
        "output": Path(env.get("OUTPUT_ROOT", str(work / "runs" / group))).expanduser().resolve(),
        "dataset": Path(env.get("DATASET_BASE", str(work / "datasets/May-pick-and-place"))).expanduser().resolve(),
        "base": Path(env.get("PRETRAINED_PATH", str(work / "models/pi05_base"))).expanduser().resolve(),
        "tokenizer": Path(env.get("TOKENIZER_PATH", str(work / "models/google/paligemma-3b-pt-224"))).expanduser().resolve(),
    }


def check_original_base(base):
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (base / name).is_file():
            raise ValueError(f"Missing {base / name}; set PRETRAINED_PATH to the original local base")
    config = read_json(base / "config.json")
    if any(config.get(key) for key in ("train_action_head", "train_action_projections", "train_action_expert_last_n_layers", "piper_gripper_history_steps")):
        raise ValueError("Use an original base checkpoint, not an adapted Piper head/history model")
    with (base / "model.safetensors").open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Base model.safetensors has an incomplete header")
        size = struct.unpack("<Q", prefix)[0]
        if not 2 <= size <= min(64 * 1024**2, (base / "model.safetensors").stat().st_size - 8):
            raise ValueError("Invalid base safetensors header length")
        header = json.loads(stream.read(size))
    if any(key.endswith(("vlm_prompt_tokens.weight", "prompt_tokens.weight")) for key in header):
        raise ValueError("Training base already contains prompt tensors; use the shared original pi05 base")
    return config


def dataset_identity(dataset):
    info = read_json(dataset / "meta/info.json")
    names = [f"joint_{index}.pos" for index in range(1, 7)] + ["gripper.pos"]
    for key in ("observation.state", "action"):
        feature = info["features"][key]
        if feature.get("shape") != [7] or sorted(feature.get("names", [])) != sorted(names):
            raise ValueError(f"{dataset}: {key} requires the seven named Piper coordinates")
    cameras = [key for key, val in info["features"].items() if val.get("dtype") in ("video", "image")]
    if len(cameras) != 2:
        raise ValueError(f"This protocol requires the same two cameras in every method; got {cameras}")
    if int(info.get("total_tasks", 1)) != 1:
        raise ValueError("Each Piper task directory must contain exactly one task for the documented split")
    files = sorted(path for path in dataset.rglob("*") if path.is_file())
    metadata = {str(path.relative_to(dataset)): sha256_file(path) for path in files if path.relative_to(dataset).parts[0] == "meta"}
    inventory = [{"path": str(path.relative_to(dataset)), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in files if path.relative_to(dataset).parts[0] in ("data", "videos")]
    return {
        "root": str(dataset), "episodes": info["total_episodes"], "frames": info["total_frames"],
        "fps": info["fps"], "cameras": cameras, "features": info["features"],
        "metadata_sha256": metadata, "data_video_inventory_sha256": fingerprint(inventory),
        "inventory_note": "Data/video identity uses relative path, byte size and mtime; metadata is content-hashed.",
    }


def build_command(job, common, paths):
    r = job["recipe"]
    values = {
        "dataset.repo_id": "may-pick-and-place/" + TASKS[job["task_id"]],
        "dataset.root": str(paths["dataset"] / TASKS[job["task_id"]]),
        "dataset.video_backend": "pyav", "dataset.use_imagenet_stats": False,
        "dataset.image_transforms.enable": False, "dataset.eval_split": common["eval_split"],
        "policy.type": "pi05", "policy.training_stage": "flow",
        "policy.pretrained_path": str(paths["base"]), "policy.tokenizer_name": str(paths["tokenizer"]),
        "policy.device": "cuda", "policy.dtype": "float32", "policy.compile_model": False,
        "policy.gradient_checkpointing": True, "policy.use_peft": False,
        "policy.num_vlm_prompt_tokens": 16, "policy.num_prompt_tokens": 16,
        "policy.train_action_projections": False, "policy.train_action_expert_last_n_layers": 0,
        "policy.train_action_head": False, "policy.piper_train_normalization": True,
        "policy.piper_gripper_history_steps": 0, "policy.use_relative_actions": False,
        "policy.chunk_size": 50, "policy.n_action_steps": common["n_action_steps"],
        "policy.num_inference_steps": 10, "policy.n_obs_steps": 1,
        "policy.next_action_pretrain_steps": r["preparation_updates"],
        "policy.next_action_bridge_steps": r["bridge_updates"],
        "policy.next_action_masked_steps": r["masked_steps"],
        "policy.next_action_full_mask_probability": r["full_mask_probability"],
        "policy.ntk_save_stage_snapshots": False, "policy.cabo_enabled": r["cpc"],
        "policy.cabo_prompt_update_ratio": 2.0, "policy.action_prompt_lr_scale": r["action_prompt_lr_scale"],
        "policy.optimizer_lr": 2.5e-5, "policy.optimizer_betas": [0.9, 0.95],
        "policy.optimizer_eps": 1e-8, "policy.optimizer_weight_decay": 0.01,
        "policy.optimizer_grad_clip_norm": 0.0, "policy.scheduler_warmup_steps": 1000,
        "policy.scheduler_decay_steps": 30000, "policy.scheduler_decay_lr": 1e-5,
        "policy.push_to_hub": False, "policy.normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"},
        "output_dir": job["output_dir"], "job_name": job["run_name"],
        "steps": r["flow_updates"], "batch_size": common["batch_size"], "seed": job["train_seed"],
        "num_workers": common["num_workers"], "cudnn_deterministic": True,
        "save_checkpoint": True, "save_steps": [step - r["preparation_updates"] for step in common["cumulative_checkpoints"]],
        "env_eval_freq": 0, "log_freq": 50, "eval_steps": 0,
        "piper_eval.freq": common["action_eval_freq"], "piper_eval.samples": common["action_eval_samples"],
        "piper_eval.seed": 20261107, "piper_eval.execution_steps": common["n_action_steps"],
        "piper_eval.update_freq": 50, "piper_eval.sampling": "episode_stratified",
        "piper_eval.select_best_saved": False, "wandb.enable": False,
    }
    def encode(value):
        return json.dumps(value, separators=(",", ":")) if not isinstance(value, str) else value
    return [sys.executable, "-u", "-m", "lerobot.scripts.lerobot_train", *[f"--{key}={encode(value)}" for key, value in values.items()]]


def existing_parent(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def space_check(paths, common, prep, *, enforce=True):
    # Conservative upper bound: a BF16 input may be saved as FP32, plus one packaging copy.
    full_estimate = 2 * (paths["base"] / "model.safetensors").stat().st_size + 64 * 1024**2
    copies = len(common["cumulative_checkpoints"]) + int(prep > 0) + 1
    required = copies * full_estimate + common["reserve_gib"] * GIB
    usage = shutil.disk_usage(existing_parent(paths["output"]))
    print(f"Disk: free={usage.free / GIB:.1f} GiB / capacity={usage.total / GIB:.1f} GiB; conservative next-run peak+reserve={required / GIB:.1f} GiB", flush=True)
    if enforce and usage.free < required:
        raise ValueError("Insufficient measured free space for the next run; no checkpoint was removed")
    return {"free_bytes": usage.free, "estimated_required_bytes": required}


def last_diagnostic(directory):
    path = Path(directory) / "training_diagnostics.jsonl"
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 262144))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            pass
    return None


def progress(job, elapsed, queue_index, queue_size):
    out = Path(job["output_dir"])
    record = last_diagnostic(out)
    prep = job["recipe"]["preparation_updates"]
    if record is not None:
        done, stage = prep + record["step"], "adaptation"
    else:
        record = last_diagnostic(out.with_name(out.name + "_next_action_pretrain"))
        done, stage = (record["step"], "preparation") if record else (0, "startup")
    total = job["recipe"]["cumulative_updates"]
    # Use the current stage's recent timings; future stage speed may differ.
    timings = record.get("train", {}) if record else {}
    step_s = timings.get("update_s", 0) + timings.get("dataloading_s", 0)
    remaining = f"~{max(0, total - done) * step_s / 60:.1f} min (current-stage rate)" if step_s > 0 else "warming up"
    print(f"\n[QUEUE {queue_index}/{queue_size}] {job['run_name']} | {stage} | cumulative {done}/{total} ({100 * done / total:.1f}%) | elapsed {elapsed / 60:.1f} min | ETA {remaining}", flush=True)


def run_training(command, env, log, job, common, index, count):
    started = time.monotonic()
    with Path(log).open("xb") as output:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            last_report = started
            while True:
                ready, _, _ = select.select([process.stdout], [], [], 1.0)
                if ready:
                    data = os.read(process.stdout.fileno(), 65536)
                    if data:
                        output.write(data)
                        output.flush()
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                    elif process.poll() is not None:
                        break
                if time.monotonic() - last_report >= 30:
                    progress(job, time.monotonic() - started, index, count)
                    last_report = time.monotonic()
                    if shutil.disk_usage(Path(job["output_dir"]).parent).free < 5 * GIB:
                        raise RuntimeError("Free disk fell below 5 GiB; stopping this run and preserving its files")
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"Training exited {return_code}; retained log: {log}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    return time.monotonic() - started


def check_training_gpu(env):
    """Probe CUDA in a short-lived child so the queue holds no GPU context itself."""
    code = (
        "import json,torch; "
        "assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'Exactly one visible CUDA device is required'; "
        "x=torch.ones(1,device='cuda'); (x+1).sum().item(); "
        "p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'name':p.name,'total_memory_bytes':p.total_memory,'torch':torch.__version__,'cuda':torch.version.cuda}))"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"CUDA preflight failed in the active Python environment:\n{result.stderr}")
    report = json.loads(result.stdout.splitlines()[-1])
    print(f"CUDA preflight: {report}", flush=True)
    return report


def audit_run(job):
    output = Path(job["output_dir"])
    parameters = read_json(output / "trainable_parameters.json")
    if parameters["trainable_total"] != 49152 or parameters["categories"].get("vlm_prompt") != 32768 or parameters["categories"].get("action_prompt") != 16384 or any(parameters["categories"].get(key, 0) for key in ("action_projection", "action_expert_blocks", "action_expert_norm", "action_time_mlp", "other")):
        raise ValueError("Actual trainable parameters differ from the 49,152-parameter prompt-only protocol")
    final_step = job["recipe"]["flow_updates"]
    config = read_json(output / "checkpoints" / f"{final_step:06d}" / "pretrained_model/config.json")
    expected = {
        "training_stage": "flow", "num_vlm_prompt_tokens": 16, "num_prompt_tokens": 16,
        "train_action_head": False, "train_action_projections": False,
        "train_action_expert_last_n_layers": 0, "piper_gripper_history_steps": 0,
        "piper_train_normalization": True, "use_relative_actions": False,
        "cabo_enabled": job["recipe"]["cpc"], "action_prompt_lr_scale": job["recipe"]["action_prompt_lr_scale"],
        "chunk_size": 50,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved policy config differs from the declared reviewer control")
    split = read_json(output / "dataset_split.json")
    if not split["train_episodes"] or not split["eval_episodes"] or set(split["train_episodes"]) & set(split["eval_episodes"]):
        raise ValueError("Invalid or overlapping actual training/validation episode split")
    prep = output.with_name(output.name + "_next_action_pretrain")
    if job["recipe"]["preparation_updates"]:
        prep_split = read_json(prep / "dataset_split.json")
        if split != prep_split:
            raise ValueError("Preparation and adaptation used different datasets/splits")
    skipped = 0.0
    peak = 0.0
    for directory in (output, prep) if job["recipe"]["preparation_updates"] else (output,):
        path = directory / "training_diagnostics.jsonl"
        if not path.is_file():
            raise ValueError(f"Missing training audit: {path}")
        for line in path.read_text().splitlines():
            record = json.loads(line)
            stats = record.get("diagnostics_rank0", {}).get("optimizer_step/skipped", {})
            skipped += (stats.get("mean") or 0) * stats.get("count", 0)
            peak = max(peak, record.get("diagnostics_rank0", {}).get("resources/step_peak_allocated_gpu_gib", {}).get("max") or 0)
    if skipped:
        raise ValueError(f"Observed {skipped:g} skipped optimizer steps; cannot report the matched update budget")
    return {
        "parameters": parameters, "split": split,
        "normalization": read_json(output / "piper_normalization.json"),
        "peak_allocated_gpu_gib": peak,
        "gpu_peak_scope": "maximum allocated in training steps; excludes model initialization and offline evaluation",
        "skipped_optimizer_steps": skipped,
    }


def compact_run(job, paths, common, audit):
    sys.path.insert(0, str(ROOT / "examples/piper"))
    from revision_checkpoint import pack_checkpoint

    output = Path(job["output_dir"])
    prep = job["recipe"]["preparation_updates"]
    sources = []
    for total in common["cumulative_checkpoints"]:
        step = total - prep
        model = output / "checkpoints" / f"{step:06d}" / "pretrained_model"
        state = read_json(model.parent / "training_state/training_step.json")
        if state.get("step") != step:
            raise ValueError(f"Wrong checkpoint counter: {model}")
        sources.append((total, step, model, False))
    if prep:
        stage = output.with_name(output.name + "_next_action_pretrain")
        sources.append((prep, prep, stage / "checkpoints" / f"{prep:06d}" / "pretrained_model", True))
    for _, _, model, _ in sources:
        if model.is_symlink() or (model / "model.safetensors").is_symlink() or not (model / "model.safetensors").is_file():
            raise ValueError(f"Missing/unexpected linked fresh checkpoint: {model}")
        shutil.copy2(paths["dataset"] / TASKS[job["task_id"]] / "meta/info.json", model / "dataset_info.json")
    storage = paths["output"] / "storage_base"
    if not storage.exists():
        temporary = storage.with_name("storage_base.tmp")
        temporary.mkdir()
        try:
            try:
                os.link(sources[0][2] / "model.safetensors", temporary / "model.safetensors")
            except OSError:
                shutil.copy2(sources[0][2] / "model.safetensors", temporary / "model.safetensors")
            write_json(temporary / "identity.json", {
                "role": "reconstruction only; NEVER a training initialization",
                "training_base": str(paths["base"]), "training_base_sha256": common["base_sha256"],
                "model_sha256": sha256_file(temporary / "model.safetensors"),
                "source_run": job["run_name"], "trainable_parameters": audit["parameters"]["trainable_total"],
            })
            temporary.rename(storage)
        except BaseException:
            # Preserve any completed weight copy for diagnosis; never remove another run's data.
            raise
    storage_identity = read_json(storage / "identity.json")
    if storage_identity["training_base_sha256"] != common["base_sha256"]:
        raise ValueError("Shared storage base belongs to a different training initialization")
    if sha256_file(storage / "model.safetensors") != storage_identity["model_sha256"]:
        raise ValueError("Shared storage base changed; preserving this run's full checkpoints")
    bundles = []
    for total, step, model, preparation in sources:
        label = f"preparation-{total:06d}" if preparation else f"total-{total:06d}"
        bundle = output / "bundles" / label
        manifest = pack_checkpoint(model, storage, bundle)
        if not manifest.get("reconstruction_verified") or manifest["base_model_sha256"] != storage_identity["model_sha256"]:
            raise ValueError("Unverified checkpoint archive; preserving all full checkpoints")
        bundles.append({
            "cumulative_updates": total, "flow_updates": 0 if preparation else step,
            "preparation": preparation, "bundle": str(bundle),
            "model_sha256": sha256_file(model / "model.safetensors"),
            "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "source_checkpoint": str(model), "storage_manifest": manifest,
        })
    # Every source has been streamed back and checksum-verified by pack_checkpoint first.
    # Delete ONLY the large model files made by this just-completed, owned run. Training
    # state, configs, diagnostics and exact reconstruction bundles are retained.
    if not common["keep_full"]:
        for _, _, model, _ in sources:
            (model / "model.safetensors").unlink()
            write_json(model / "COMPACT_CHECKPOINT.json", {"restore_with": "examples/piper/revision_checkpoint.py materialize", "storage_base": str(storage), "bundle": next(item["bundle"] for item in bundles if item["source_checkpoint"] == str(model))})
    return bundles


def show_status(output):
    records = sorted((output / "records").glob("*.json")) if (output / "records").is_dir() else []
    print("run\tstatus\tcumulative_updates\ttraining_hours\tmanifest")
    for path in records:
        record = read_json(path)
        print(f"{record['run_name']}\t{record['status']}\t{record['recipe']['cumulative_updates']}\t{record.get('training_wall_seconds', 0) / 3600:.2f}\t{path}")
    if not records:
        print(f"No recorded runs under {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="See examples/piper/README_REVISION.md. Paths: WORK_ROOT, DATASET_BASE, PRETRAINED_PATH, TOKENIZER_PATH. DRY_RUN=true validates and prints only.")
    parser.add_argument("group", nargs="?", default="core", help="core / extended / cpc_sweep / all, or comma-separated method names")
    parser.add_argument("seeds", nargs="?", default="0", help="Independent training seeds, e.g. 0 or 0,1,2")
    parser.add_argument("--tasks", default=os.environ.get("TASKS", "1,2,3,4"))
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    env = os.environ.copy()
    paths = paths_from_env(env)
    if args.status:
        show_status(paths["output"])
        return 0
    methods = list(GROUPS[args.group]) if args.group in GROUPS else csv_values(args.group, GROUPS["all"])
    tasks = csv_values(args.tasks, TASKS)
    seeds = [int(seed) for seed in csv_values(args.seeds)]
    if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be distinct non-negative integers")
    dry = env_bool(env, "DRY_RUN", False)
    total = env_int(env, "TOTAL_UPDATES", 4000, 1001)
    gpu = env.get("GPU_ID", env.get("GPU_IDS", "0"))
    if not re.fullmatch(r"\d+", gpu) or int(env.get("NUM_PROCESSES", "1")) != 1:
        raise ValueError("This single-A800 launcher requires one numeric GPU_ID and NUM_PROCESSES=1")
    if int(env.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Do not run this sequential launcher inside a distributed worker")
    if "PRETRAINED_PATH" not in env:
        raise ValueError("Set PRETRAINED_PATH explicitly to this server's original pi05_base or pi05_libero_base")
    check_original_base(paths["base"])
    if not (paths["tokenizer"] / "tokenizer_config.json").is_file() or not any((paths["tokenizer"] / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise ValueError(f"Incomplete local tokenizer: {paths['tokenizer']}")
    split = float(env.get("EVAL_SPLIT", "0.1"))
    if not math.isfinite(split) or not 0 < split < 0.5:
        raise ValueError("EVAL_SPLIT must be in (0, 0.5); validation episodes are required")
    common = {
        "schema": SCHEMA, "cumulative_updates": total, "cumulative_checkpoints": milestones(env, total),
        "base_path": str(paths["base"]), "base_config_sha256": sha256_file(paths["base"] / "config.json"),
        "base_sha256": "not_hashed_in_dry_run" if dry else sha256_file(paths["base"] / "model.safetensors"),
        "tokenizer_path": str(paths["tokenizer"]),
        "tokenizer_files": {path.name: sha256_file(path) for path in sorted(paths["tokenizer"].iterdir()) if path.is_file() and path.suffix in (".json", ".model")},
        "batch_size": env_int(env, "BATCH_SIZE", 8), "eval_split": split,
        "n_action_steps": env_int(env, "N_ACTION_STEPS", 8), "num_workers": env_int(env, "NUM_WORKERS", 4, 0),
        "action_eval_freq": env_int(env, "ACTION_EVAL_FREQ", 500), "action_eval_samples": env_int(env, "ACTION_EVAL_SAMPLES", 32),
        "reserve_gib": env_int(env, "RESERVE_GIB", 30), "keep_full": env_bool(env, "KEEP_FULL_CHECKPOINTS", False),
        "inputs": "two RGB views, language, discretized current state; no history20",
        "stage_transition": "SAP-to-bridge retains AdamW/scheduler; preparation-to-adaptation resets both",
    }
    if common["n_action_steps"] > 50:
        raise ValueError("N_ACTION_STEPS cannot exceed chunk size 50")
    code = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    common["code_commit"] = code
    common["source_files_sha256"] = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in [Path(__file__), *sorted((ROOT / "src/lerobot/policies/pi05").glob("*.py")),
                     ROOT / "src/lerobot/scripts/lerobot_train.py", ROOT / "src/lerobot/datasets/factory.py",
                     ROOT / "src/lerobot/utils/piper_action_representation.py", ROOT / "src/lerobot/optim/cabo.py"]
    }
    identities = {}
    for task in tasks:
        dataset = paths["dataset"] / TASKS[task]
        check_dataset(dataset, "QUANTILES")
        identities[task] = dataset_identity(dataset)
    jobs = []
    # Task-first interleaving reveals deployment/embodiment problems early on one task.
    for seed in seeds:
        for task in tasks:
            for method in methods:
                run = f"task{task}-{method}-seed{seed}-total{total}"
                job = {"schema": SCHEMA, "run_name": run, "task_id": task, "method": method, "train_seed": seed, "recipe": recipe(method, total), "dataset": identities[task], "output_dir": str(paths["output"] / run)}
                job["signature"] = fingerprint({"common": common, "job": job})
                jobs.append(job)
    print(f"Piper reviewer protocol: {len(jobs)} runs, ONE GPU {gpu}, sequential; batch={common['batch_size']}; 16+16 prompts=49,152 parameters", flush=True)
    print(f"Output: {paths['output']}\nInputs: {common['inputs']}\nCumulative checkpoint budgets: {common['cumulative_checkpoints']}", flush=True)
    space_check(paths, common, max(job["recipe"]["preparation_updates"] for job in jobs), enforce=False)
    if dry:
        for job in jobs:
            print(f"\n{job['run_name']}: {json.dumps(job['recipe'])}")
            print(shlex.join(["env", f"CUDA_VISIBLE_DEVICES={gpu}", *build_command(job, common, paths)]))
        print("DRY_RUN complete: no model loaded, no training or robot execution, no output files created.")
        return 0
    paths["output"].mkdir(parents=True, exist_ok=True)
    queue_lock = (paths["output"] / ".revision.lock").open("a+")
    try:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        queue_lock.close()
        raise ValueError("Another revision queue is already using this output group") from exc
    protocol_path = paths["output"] / "protocol.json"
    lock_json_contract(protocol_path, common)
    for task, identity in identities.items():
        lock_json_contract(paths["output"] / "task_protocols" / f"task{task}.json", identity)
    # Check ALL planned destinations before starting a potentially long queue.
    pending = []
    storage = paths["output"] / "storage_base"
    storage_sha = sha256_file(storage / "model.safetensors") if storage.exists() else None
    for job in jobs:
        record_path = paths["output"] / "records" / (job["run_name"] + ".json")
        if record_path.exists():
            saved = read_json(record_path)
            if saved.get("signature") != job["signature"]:
                raise ValueError(f"Run configuration changed: {record_path}; use a new RUN_GROUP")
            if saved.get("status") == "completed":
                if storage_sha is None:
                    raise ValueError("Completed runs require their shared storage_base; it is missing")
                verify_completed(saved, storage, storage_sha)
                print(f"SKIP completed: {job['run_name']}")
                continue
            raise ValueError(f"An unfinished run is preserved at {record_path}; inspect its log/checkpoint before resuming, or select a fresh RUN_GROUP")
        for path in (Path(job["output_dir"]), Path(job["output_dir"] + "_next_action_pretrain")):
            if path.exists():
                raise ValueError(f"Unowned existing output; refusing to overwrite: {path}")
        pending.append(job)
    if not pending:
        show_status(paths["output"])
        queue_lock.close()
        return 0
    env.update({"CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled", "TQDM_MININTERVAL": "5", "PYTHONPATH": str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")})
    for key, path in {"TMPDIR": paths["work"] / "tmp", "HF_HOME": paths["work"] / "cache/huggingface", "TORCHINDUCTOR_CACHE_DIR": paths["work"] / "cache/torchinductor", "TRITON_CACHE_DIR": paths["work"] / "cache/triton"}.items():
        env[key] = str(path)
        path.mkdir(parents=True, exist_ok=True)
    hardware = check_training_gpu(env)
    for index, job in enumerate(pending, 1):
        job["hardware"] = hardware
        job["disk_before"] = space_check(paths, common, job["recipe"]["preparation_updates"])
        job["status"] = "running"
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        job["command"] = build_command(job, common, paths)
        job["common"] = common
        record_path = paths["output"] / "records" / (job["run_name"] + ".json")
        log = paths["output"] / "logs" / (job["run_name"] + ".log")
        log.parent.mkdir(exist_ok=True)
        write_json(record_path, job)
        print(f"\n[QUEUE {index}/{len(pending)}] {job['run_name']}\nLog: {log}", flush=True)
        try:
            job["training_wall_seconds"] = run_training(job["command"], env, log, job, common, index, len(pending))
            audit = audit_run(job)
            lock_json_contract(paths["output"] / "task_protocols" / f"task{job['task_id']}-actual.json", {"split": audit["split"], "normalization": audit["normalization"]})
            job["audit"] = audit
            job["checkpoints"] = compact_run(job, paths, common, audit)
            job["status"] = "completed"
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
            write_json(Path(job["output_dir"]) / "revision_manifest.json", job)
            write_json(record_path, job)
            print(f"DONE {job['run_name']} | training {job['training_wall_seconds'] / 3600:.2f} h | peak allocated {audit['peak_allocated_gpu_gib']:.2f} GiB | compact checkpoints verified", flush=True)
        except BaseException as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
            write_json(record_path, job)
            raise
    show_status(paths["output"])
    queue_lock.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Piper revision: {error}", file=sys.stderr)
        raise SystemExit(1) from error
