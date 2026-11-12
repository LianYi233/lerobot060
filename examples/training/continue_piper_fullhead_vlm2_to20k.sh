#!/usr/bin/env bash
# Resume the completed task2 fullhead_vlm2 +10k PI05 run through TOTAL step 20k.
# Preserves model, optimizer moments, RNG, data-sampler offset, and trained processors.
# Uses a smooth LR restart; source checkpoint is never modified.
set -euo pipefail

usage() {
cat <<'EOF'
Usage:
  GPU_IDS=0 DRY_RUN=true bash examples/training/continue_piper_fullhead_vlm2_to20k.sh
  GPU_IDS=0 DRY_RUN=false bash examples/training/continue_piper_fullhead_vlm2_to20k.sh
Defaults:
  SOURCE_RUN=/root/autodl-tmp/piper-partial-ft-work/chkpt/2601-lerobot/piper/task2-fullhead-vlm2-plus10k/pi05-may-2-remove_the_cuboid_from_blue_plate-dual_prompt_only-seed0
  SOURCE_STEP=10000, TARGET_STEPS=20000
  LR_RESTART_WARMUP=300, LR_RESTART_PEAK_RATIO=0.5
  SAVE_STEPS='[12000,15000,20000]'
  RUN_GROUP=task2-fullhead-vlm2-resume10k-to20k
Override SOURCE_RUN, WORK_ROOT, RUN_GROUP, TARGET_STEPS, SAVE_STEPS if needed.
This is an optimizer-state resume (NOT a fresh 20k run). Input is the FULL
checkpoint directory including training_state/, NOT just pretrained_model/.
No robot is connected or commanded by this script.
EOF
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if (( $# )); then echo "Unexpected arguments: $*" >&2; usage >&2; exit 2; fi

export WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/piper-partial-ft-work}"
RUN_NAME=pi05-may-2-remove_the_cuboid_from_blue_plate-dual_prompt_only-seed0
SOURCE_RUN="${SOURCE_RUN:-$WORK_ROOT/chkpt/2601-lerobot/piper/task2-fullhead-vlm2-plus10k/$RUN_NAME}"
SOURCE_STEP="${SOURCE_STEP:-10000}"
TARGET_STEPS="${TARGET_STEPS:-20000}"
LR_RESTART_WARMUP="${LR_RESTART_WARMUP:-300}"
LR_RESTART_PEAK_RATIO="${LR_RESTART_PEAK_RATIO:-0.5}"
SAVE_STEPS="${SAVE_STEPS:-[12000,15000,20000]}"
RUN_GROUP="${RUN_GROUP:-task2-fullhead-vlm2-resume10k-to20k}"
GPU_IDS="${GPU_IDS:-0}"
DRY_RUN="${DRY_RUN:-true}"
if [[ ! "$GPU_IDS" =~ ^[0-9]+$ ]]; then echo "Set one numeric GPU_IDS" >&2; exit 2; fi
case "$DRY_RUN" in true|false) ;; *) echo "DRY_RUN must be true/false" >&2; exit 2;; esac

# Use a fresh independent output directory; never overwrite the completed 10k run.
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/chkpt/2601-lerobot/piper/$RUN_GROUP}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
LOG_ROOT="${LOG_ROOT:-$WORK_ROOT/logs/piper/$RUN_GROUP}"
LOG_FILE="$LOG_ROOT/$RUN_NAME.log"
CHECKPOINT="$SOURCE_RUN/checkpoints/$(printf '%06d' "$SOURCE_STEP")"
TRAIN_CONFIG="$CHECKPOINT/pretrained_model/train_config.json"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CHECKPOINT SOURCE_STEP TARGET_STEPS LR_RESTART_WARMUP LR_RESTART_PEAK_RATIO SAVE_STEPS OUTPUT_DIR

python - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["CHECKPOINT"]).resolve()
out = Path(os.environ["OUTPUT_DIR"]).resolve()
if out.exists():
    raise SystemExit(f"Refusing to overwrite existing destination: {out}")
if src in out.parents or out in src.parents:
    raise SystemExit("Resume destination must be separate from source checkpoint")

required = (
    "pretrained_model/config.json",
    "pretrained_model/model.safetensors",
    "pretrained_model/train_config.json",
    "pretrained_model/policy_preprocessor.json",
    "pretrained_model/policy_postprocessor.json",
    "training_state/training_step.json",
    "training_state/optimizer_state.safetensors",
    "training_state/optimizer_param_groups.json",
    "training_state/scheduler_state.json",
    "training_state/rng_state.safetensors",
)
missing = [str(src / name) for name in required if not (src / name).is_file()]
if missing:
    raise SystemExit(f"Resume requires complete checkpoint; missing: {missing}")
cfg = json.loads((src / "pretrained_model/train_config.json").read_text())
policy = cfg["policy"]
train_step = json.loads((src / "training_state/training_step.json").read_text())
source_step = int(os.environ["SOURCE_STEP"])
target = int(os.environ["TARGET_STEPS"])
warmup = int(os.environ["LR_RESTART_WARMUP"])
ratio = float(os.environ["LR_RESTART_PEAK_RATIO"])
save_steps = json.loads(os.environ["SAVE_STEPS"])
assert cfg["steps"] == source_step, (cfg["steps"], source_step)
assert train_step["step"] == source_step, (train_step["step"], source_step)
assert source_step < target, (source_step, target)
assert 0 <= warmup < target-source_step
assert 0 < ratio <= 1
assert isinstance(save_steps, list) and save_steps
assert all(type(step) is int and source_step < step <= target for step in save_steps)
assert save_steps == sorted(set(save_steps)) and save_steps[-1] == target
assert policy["partial_ft_enabled"] is True
assert policy["train_action_head"] is True
assert policy["train_action_expert_last_n_layers"] == 0
assert policy["train_action_projections"] is True
assert policy["train_vlm_last_n_layers"] == 2
assert policy["cabo_enabled"] is False
assert policy["piper_train_normalization"] is True
assert policy["use_relative_actions"] is False
assert cfg["dataset"]["eval_split"] == 0.1
assert train_step.get("num_processes") == 1
assert train_step.get("batch_size") == cfg["batch_size"]
assert Path(cfg["dataset"]["root"]).is_dir(), cfg["dataset"]["root"]
assert cfg["scheduler"]["type"] == "cosine_decay_with_warmup"
print(f"Resume validated: source_step={source_step}, target_step={target}, batch={cfg['batch_size']}")
print(f"Source: {src}")
print(f"Output: {out}")
print(f"Save at: {save_steps}; LR restart warmup={warmup}, peak_ratio={ratio}")
PY

TRAIN=(
    accelerate launch --num_machines=1 --num_processes=1 --mixed_precision=bf16
    --dynamo_backend=no --module lerobot.scripts.lerobot_train
    "--config_path=$TRAIN_CONFIG"
    --resume=true
    "--output_dir=$OUTPUT_DIR"
    "--steps=$TARGET_STEPS"
    "--save_steps=$SAVE_STEPS"
    "--scheduler.restart_at_step=$SOURCE_STEP"
    "--scheduler.restart_warmup_steps=$LR_RESTART_WARMUP"
    "--scheduler.restart_peak_ratio=$LR_RESTART_PEAK_RATIO"
)
if [[ "$DRY_RUN" == true ]]; then
    printf '%q ' env "CUDA_VISIBLE_DEVICES=$GPU_IDS" "${TRAIN[@]}"
    printf '\n'
    exit 0
fi
command -v accelerate >/dev/null || { echo "accelerate not installed" >&2; exit 1; }
python - <<'PY'
import importlib.util
if importlib.util.find_spec("lerobot.scripts.lerobot_train") is None:
    raise SystemExit("Unable to import lerobot.scripts.lerobot_train; check PYTHONPATH")
PY
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TMPDIR="${TMPDIR:-$WORK_ROOT/tmp}"
mkdir -p "$TMPDIR"
env CUDA_VISIBLE_DEVICES="$GPU_IDS" "${TRAIN[@]}" 2>&1 | tee "$LOG_FILE"
