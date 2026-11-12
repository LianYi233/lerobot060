#!/usr/bin/env bash
# Controlled Piper task-2/task-3 partial-FT profiles, each from its OWN 50k model.
# One GPU per invocation. This script does not drive a robot.
set -euo pipefail

usage() {
cat <<'EOF'
Usage: DRY_RUN=true GPU_IDS=0 bash examples/training/train_piper_partial_ft.sh fullhead|fullhead_vlm2|fullhead_vlm4
       DRY_RUN=false GPU_IDS=0 bash examples/training/train_piper_partial_ft.sh fullhead|fullhead_vlm2|fullhead_vlm4
Also supported (original partial experiments): action4|dual2|dual4
Required: DATASET_BASE, PRETRAINED_PATH (materialized task-specific 50k), TOKENIZER_PATH.
 Optional: PIPER_TASK=2 (default) or PIPER_TASK=3; task 3 REQUIRES a matching
 source train_config.json and/or dataset_info.json to prevent mixed-task runs.
Optional: WORK_ROOT=/data1/wyn/piper-work, FLOW_STEPS=10000, SAVE_STEPS='[1000,3000,5000,10000]',
          WARMUP_STEPS=500, BATCH_SIZE=1, RUN_GROUP=..., EVAL_SPLIT=0.1,
          ACTION_EVAL_FREQ=500, NUM_WORKERS=2.
The original action4/dual2/dual4 profiles train the last 4 expert blocks;
fullhead/fullhead_vlm2/fullhead_vlm4 train ALL expert blocks, final expert norm,
action projections and time MLPs. *_vlm2/4 also train VLM's final 2/4 language
layers. Vision tower, embeddings and unused expert LM head stay frozen.
All are new model-weight initialization runs from 50k: NO SAP/Bridge or optimizer resume.
This does NOT perform three-card data parallelism.
EOF
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
PROFILE="${1:-}"
FULL_ACTION_HEAD=false
EXPERT_LAST=4
case "$PROFILE" in
    action4) VLM_LAST=0 ;;
    dual2) VLM_LAST=2 ;;
    dual4) VLM_LAST=4 ;;
    fullhead) VLM_LAST=0; FULL_ACTION_HEAD=true; EXPERT_LAST=0 ;;
    fullhead_vlm2) VLM_LAST=2; FULL_ACTION_HEAD=true; EXPERT_LAST=0 ;;
    fullhead_vlm4) VLM_LAST=4; FULL_ACTION_HEAD=true; EXPERT_LAST=0 ;;
    *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then echo "Unexpected positional arguments: ${*:2}" >&2; exit 2; fi

PIPER_TASK="${PIPER_TASK:-2}"
case "$PIPER_TASK" in
  2) TASK_FOLDER=2-remove_the_cuboid_from_blue_plate ;;
  3) TASK_FOLDER=3-move_the_tennis_from_yellow_plate_to_blue_plate ;;
  *) echo "PIPER_TASK must be 2 or 3; got: $PIPER_TASK" >&2; exit 2 ;;
esac
export PIPER_TASK TASK_FOLDER

: "${DATASET_BASE:?Export DATASET_BASE pointing to the local parent of the four task folders}"
: "${TOKENIZER_PATH:?Export TOKENIZER_PATH pointing to the local PaliGemma tokenizer}"
: "${PRETRAINED_PATH:?Export PRETRAINED_PATH pointing to the materialized task-specific 50k policy directory}"
: "${GPU_IDS:?Export one physical GPU ID, e.g. GPU_IDS=1}"
[[ "$GPU_IDS" =~ ^[0-9]+$ ]] || { echo "GPU_IDS must be ONE numeric GPU, not a list" >&2; exit 2; }

export NUM_PROCESSES=1
export WORK_ROOT="${WORK_ROOT:-/data1/wyn/piper-work}"
# AutoDL checkpoints and datasets reside under /root/autodl-tmp and /root/autodl-fs.
# The legacy local preset otherwise remaps /root/autodl-tmp paths to /data1.
if [[ "$WORK_ROOT" == /root/autodl-tmp || "$WORK_ROOT" == /root/autodl-tmp/* ]]; then
  export PIPER_RUNTIME_LAYOUT=autodl
fi
export BATCH_SIZE="${BATCH_SIZE:-1}"
export DTYPE="${DTYPE:-bfloat16}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export COMPILE_MODEL=false
export GRADIENT_CHECKPOINTING=true
export FLOW_STEPS="${FLOW_STEPS:-10000}"
export WARMUP_STEPS="${WARMUP_STEPS:-500}"
export SAVE_STEPS="${SAVE_STEPS:-[1000,3000,5000,10000]}"
export EVAL_SPLIT="${EVAL_SPLIT:-0.1}"
export ACTION_EVAL_FREQ="${ACTION_EVAL_FREQ:-500}"
export ACTION_EVAL_SAMPLES="${ACTION_EVAL_SAMPLES:-32}"
export ACTION_EVAL_SAMPLING=episode_stratified
export ACTION_SELECT_BEST_SAVED=false
export NUM_WORKERS="${NUM_WORKERS:-2}"
export WANDB_ENABLE=false
export NTK_SAVE_STAGE_SNAPSHOTS=false
export KEEP_PRETRAIN_CHECKPOINT=false
export VLM_PROMPT_TOKENS=16 ACTION_PROMPT_TOKENS=16
export NORMALIZATION_MODE=QUANTILES
export N_ACTION_STEPS=8 CHUNK_SIZE=50
export RUN_GROUP="${RUN_GROUP:-partial-task${PIPER_TASK}-${PROFILE}-seed0-${FLOW_STEPS}}"
export DRY_RUN="${DRY_RUN:-true}"

if [[ ! "$FLOW_STEPS" =~ ^[1-9][0-9]*$ || ! "$WARMUP_STEPS" =~ ^[0-9]+$ ]] || (( WARMUP_STEPS >= FLOW_STEPS )); then
    echo "Set 0 <= WARMUP_STEPS < FLOW_STEPS" >&2; exit 2
fi
if [[ "$DTYPE" != "bfloat16" || "$MIXED_PRECISION" != "bf16" ]]; then
    echo "This partial FT preset requires DTYPE=bfloat16 and MIXED_PRECISION=bf16" >&2; exit 2
fi
if [[ "$EVAL_SPLIT" != "0.1" ]]; then
    echo "Keep EVAL_SPLIT=0.1 to match the original 50k task$PIPER_TASK train/validation episode split" >&2; exit 2
fi
for file in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
    [[ -f "$PRETRAINED_PATH/$file" ]] || { echo "Missing 50k policy file: $PRETRAINED_PATH/$file" >&2; exit 1; }
done
[[ -f "$DATASET_BASE/$TASK_FOLDER/meta/info.json" ]] || {
    echo "Missing task $PIPER_TASK LeRobot dataset: $DATASET_BASE/$TASK_FOLDER" >&2; exit 1;
}
# The source must be the task-matched, TRAINED prompt-enabled 50k model,
# never pi05_base, a compact storage bundle, or a checkpoint from another task.
PRETRAINED_PATH="$PRETRAINED_PATH" DATASET_BASE="$DATASET_BASE" python - <<'PY'
import json
import os
from pathlib import Path
from safetensors import safe_open
p = Path(os.environ["PRETRAINED_PATH"])
task = int(os.environ["PIPER_TASK"])
folder = os.environ["TASK_FOLDER"]
dataset_info = Path(os.environ["DATASET_BASE"]) / folder / "meta/info.json"
cfg = json.loads((p / "config.json").read_text())
for key, value in (
    ("num_vlm_prompt_tokens", 16), ("num_prompt_tokens", 16),
    ("piper_train_normalization", True), ("use_relative_actions", False),
    ("training_stage", "flow"),
):
    if cfg.get(key) != value:
        raise SystemExit(f"Wrong source policy {key}={cfg.get(key)!r}, expected {value!r}")

# Cross-check task identity from original training provenance or saved dataset
# metadata. The task3 path must provide positive evidence; otherwise fail closed.
identity_sources = []
source_train_cfg = p / "train_config.json"
if source_train_cfg.is_file():
    train_cfg = json.loads(source_train_cfg.read_text())
    data = train_cfg.get("dataset", {})
    repo_id = str(data.get("repo_id", ""))
    data_root = str(data.get("root", ""))
    if repo_id:
        if repo_id.rsplit("/", 1)[-1] != folder:
            raise SystemExit(f"Wrong pretrained task in train_config repo_id: {repo_id}")
        identity_sources.append("train_config.repo_id")
    if data_root:
        if Path(data_root).name != folder:
            raise SystemExit(f"Wrong pretrained task in train_config dataset.root: {data_root}")
        identity_sources.append("train_config.dataset.root")
source_dataset_info = p / "dataset_info.json"
if source_dataset_info.is_file():
    if json.loads(source_dataset_info.read_text()) != json.loads(dataset_info.read_text()):
        raise SystemExit(f"Pretrained dataset_info.json differs from selected task {task}")
    identity_sources.append("dataset_info.json")
if task == 3 and not identity_sources:
    raise SystemExit(
        "Task 3 requires original train_config.json or dataset_info.json "
        "inside the materialized policy to verify task identity; refusing to mix tasks"
    )

with safe_open(str(p / "model.safetensors"), framework="pt", device="cpu") as f:
    for name, shape in (
        ("model.vlm_prompt_tokens.weight", [16, 2048]),
        ("model.prompt_tokens.weight", [16, 1024]),
    ):
        if name not in f.keys() or list(f.get_slice(name).get_shape()) != shape:
            raise SystemExit(f"Missing/mismatched learned prompt tensor: {name}, expected {shape}")
print(f"Source policy validated for task {task} (learned 16+16 prompts): {p}; identity={identity_sources}")
PY

echo "Piper task=$PIPER_TASK profile=$PROFILE GPU=$GPU_IDS VLM_last=$VLM_LAST full_action_head=$FULL_ACTION_HEAD expert_last=$EXPERT_LAST"
echo "Initialization: $PRETRAINED_PATH; flow updates=$FLOW_STEPS, no SAP/Bridge"
echo "Output group: $RUN_GROUP; LR base=2.5e-5, projections/expert=1e-5, VLM=1e-6"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_piper_autodl.sh" "$PIPER_TASK" dual_prompt_only 0 \
  --policy.partial_ft_enabled=true \
  "--policy.train_vlm_last_n_layers=$VLM_LAST" \
  --policy.train_action_projections=true \
  "--policy.train_action_expert_last_n_layers=$EXPERT_LAST" \
  "--policy.train_action_head=$FULL_ACTION_HEAD" \
  --policy.action_projection_lr_scale=0.4 \
  --policy.action_expert_lr_scale=0.4 \
  --policy.vlm_lr_scale=0.04 \
  --policy.cabo_enabled=false \
  --policy.piper_train_normalization=true \
  --policy.use_relative_actions=false \
  --policy.optimizer_lr=0.000025 \
  --policy.optimizer_grad_clip_norm=1.0 \
  "--policy.scheduler_warmup_steps=$WARMUP_STEPS" \
  "--policy.scheduler_decay_steps=$FLOW_STEPS" \
  --policy.scheduler_decay_lr=0.000001 \
  --dataset.image_transforms.enable=false \
  --log_freq=50 \
  --eval_steps=0
