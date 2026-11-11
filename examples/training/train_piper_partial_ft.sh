#!/usr/bin/env bash
# Three controlled Piper task-2 partial-FT profiles from the SAME completed 50k model.
# One GPU per invocation. This script does not drive a robot.
set -euo pipefail

usage() {
cat <<'EOF'
Usage: DRY_RUN=true GPU_IDS=1 bash examples/training/train_piper_partial_ft.sh action4|dual2|dual4
       DRY_RUN=false GPU_IDS=1 bash examples/training/train_piper_partial_ft.sh action4|dual2|dual4
Required: DATASET_BASE, PRETRAINED_PATH (materialized task2 50k), TOKENIZER_PATH.
Optional: WORK_ROOT=/data1/wyn/piper-work, FLOW_STEPS=10000, SAVE_STEPS='[1000,3000,5000,10000]',
          WARMUP_STEPS=500, BATCH_SIZE=1, RUN_GROUP=..., EVAL_SPLIT=0.1,
          ACTION_EVAL_FREQ=500, NUM_WORKERS=2.
The three profiles use identical prompt banks, action projections and final 4 action
expert layers. dual2/dual4 additionally train the last 2/4 VLM language layers.
All are new model-weight initialization runs from 50k: NO SAP/Bridge or optimizer resume.
This does NOT perform three-card data parallelism.
EOF
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
PROFILE="${1:-}"
case "$PROFILE" in
    action4) VLM_LAST=0 ;;
    dual2) VLM_LAST=2 ;;
    dual4) VLM_LAST=4 ;;
    *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then echo "Unexpected positional arguments: ${*:2}" >&2; exit 2; fi

: "${DATASET_BASE:?Export DATASET_BASE pointing to the local parent of the four task folders}"
: "${TOKENIZER_PATH:?Export TOKENIZER_PATH pointing to the local PaliGemma tokenizer}"
: "${PRETRAINED_PATH:?Export PRETRAINED_PATH pointing to the materialized task2 50k policy directory}"
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
export RUN_GROUP="${RUN_GROUP:-partial-task2-${PROFILE}-seed0-${FLOW_STEPS}}"
export DRY_RUN="${DRY_RUN:-true}"

if [[ ! "$FLOW_STEPS" =~ ^[1-9][0-9]*$ || ! "$WARMUP_STEPS" =~ ^[0-9]+$ ]] || (( WARMUP_STEPS >= FLOW_STEPS )); then
    echo "Set 0 <= WARMUP_STEPS < FLOW_STEPS" >&2; exit 2
fi
if [[ "$DTYPE" != "bfloat16" || "$MIXED_PRECISION" != "bf16" ]]; then
    echo "This partial FT preset requires DTYPE=bfloat16 and MIXED_PRECISION=bf16" >&2; exit 2
fi
if [[ "$EVAL_SPLIT" != "0.1" ]]; then
    echo "Keep EVAL_SPLIT=0.1 to match the original 50k task2 train/validation episode split" >&2; exit 2
fi
for file in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
    [[ -f "$PRETRAINED_PATH/$file" ]] || { echo "Missing 50k policy file: $PRETRAINED_PATH/$file" >&2; exit 1; }
done
[[ -f "$DATASET_BASE/2-remove_the_cuboid_from_blue_plate/meta/info.json" ]] || {
    echo "Missing task 2 LeRobot dataset under DATASET_BASE=$DATASET_BASE" >&2; exit 1;
}
# The source must be the TRAINED prompt-enabled task2 model, not pi05_base or a storage bundle.
PRETRAINED_PATH="$PRETRAINED_PATH" python - <<'PY'
import json
import os
from pathlib import Path
from safetensors import safe_open
p = Path(os.environ["PRETRAINED_PATH"])
cfg = json.loads((p / "config.json").read_text())
for key, value in (
    ("num_vlm_prompt_tokens", 16), ("num_prompt_tokens", 16),
    ("piper_train_normalization", True), ("use_relative_actions", False),
    ("training_stage", "flow"),
):
    if cfg.get(key) != value:
        raise SystemExit(f"Wrong source policy {key}={cfg.get(key)!r}, expected {value!r}")
with safe_open(str(p / "model.safetensors"), framework="pt", device="cpu") as f:
    for name, shape in (
        ("model.vlm_prompt_tokens.weight", [16, 2048]),
        ("model.prompt_tokens.weight", [16, 1024]),
    ):
        if name not in f.keys() or list(f.get_slice(name).get_shape()) != shape:
            raise SystemExit(f"Missing/mismatched learned prompt tensor: {name}, expected {shape}")
print(f"Source policy validated (learned 16+16 prompts): {p}")
PY

echo "Piper partial FT profile=$PROFILE GPU=$GPU_IDS VLM_last=$VLM_LAST expert_last=4"
echo "Initialization: $PRETRAINED_PATH; flow updates=$FLOW_STEPS, no SAP/Bridge"
echo "Output group: $RUN_GROUP; LR base=2.5e-5, projections/expert=1e-5, VLM=1e-6"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_piper_autodl.sh" 2 dual_prompt_only 0 \
  --policy.partial_ft_enabled=true \
  "--policy.train_vlm_last_n_layers=$VLM_LAST" \
  --policy.train_action_projections=true \
  --policy.train_action_expert_last_n_layers=4 \
  --policy.train_action_head=false \
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
