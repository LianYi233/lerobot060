#!/usr/bin/env bash
# Full continuous-action head adaptation with a frozen VLM and optional causal history.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_piper_action_head.sh [TASK] [TOKENS_PER_BANK] [SEED]

Defaults: task1, 128+128 prompts, seed0, history20, 12000 direct flow updates.
Train both prompt banks AND the full continuous-action head: all expert blocks,
final adaptive norm, action input/output maps, and time MLPs. VLM/vision stay frozen.
The unused expert language-output head and legacy inpainting embedding stay frozen.
This is NOT prompt-only tuning and NOT the old last18 recipe (norm/time were frozen).

PRETRAINED_PATH is required: use the SAME original pi05_base as the history20 control.
Do not use a trained Piper checkpoint for the initial capacity comparison.
PROMPT_LR=0.0001, PROMPT_FINAL_LR=0.00001; ACTION_EXPERT_LR_SCALE=0.1 scales
ALL action-head parameters (including projections/time/norm), giving 1e-5 -> 1e-6.
Both groups use warmup/cosine; all three LR variables can be overridden.
GPU_IDS=0,1; NUM_PROCESSES inferred; BATCH_SIZE=32 PER GPU (override as needed).
More trainable weights add gradients, optimizer states and DDP communication.
GRIPPER_HISTORY_STEPS=20 (default) or 0; TOKENS_PER_BANK=16/32/64/128/512.
Absolute actions, chunk16/execute8, all episodes with 10% holdout, FP32, no compile.
No priming, bridge, CABO, PEFT or image augmentation. Full checkpoints at 6000/9000/12000.
W&B, action_eval and local diagnostics remain enabled. Keep DATASET_BASE/TOKENIZER_PATH
pointing to your actual local files. WORK_ROOT defaults to /data1/wyn/piper-work.
RUN_GROUP/OUTPUT_ROOT/LOG_ROOT are parents; a full-action-head-tN-bB-historyH tag is added.
Use a fresh parent when changing LR/base/seed/process count; no resume or extra CLI args.
DRY_RUN=true prints the command without GPU training. MAIN_PROCESS_PORT defaults to 30328.
EOF
  exit 0
fi
TASK="${1:-1}"
TOKENS="${2:-128}"
SEED="${3:-0}"
if (( $# > 3 )) || [[ ! "${TASK}" =~ ^([1-4]|all)$ ]] || [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "Use TASK (1..4 or all), TOKENS, non-negative SEED; no extra CLI arguments" >&2
  exit 2
fi
case "${TOKENS}" in
  16|32|64|128|512) ;;
  *) echo "TOKENS must be 16, 32, 64, 128 or 512 PER BANK" >&2; exit 2 ;;
esac
: "${PRETRAINED_PATH:?Set PRETRAINED_PATH to the original local base checkpoint}"
export PRETRAINED_PATH
export GRIPPER_HISTORY_STEPS="${GRIPPER_HISTORY_STEPS:-20}"
if [[ "${GRIPPER_HISTORY_STEPS}" != 0 && "${GRIPPER_HISTORY_STEPS}" != 20 ]]; then
  echo "GRIPPER_HISTORY_STEPS must be 0 or 20" >&2
  exit 2
fi
export BATCH_SIZE="${BATCH_SIZE:-32}"
if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
export PROMPT_LR="${PROMPT_LR:-0.0001}" PROMPT_FINAL_LR="${PROMPT_FINAL_LR:-0.00001}"
export ACTION_EXPERT_LR_SCALE="${ACTION_EXPERT_LR_SCALE:-0.1}"
"${PYTHON:-python}" - <<'PY'
import math
import os

try:
    peak, floor, scale = (float(os.environ[key]) for key in
                          ("PROMPT_LR", "PROMPT_FINAL_LR", "ACTION_EXPERT_LR_SCALE"))
    if not all(math.isfinite(v) and v > 0 for v in (peak, floor, scale)) or floor > peak or scale > 1:
        raise ValueError("require 0 < final <= peak and 0 < scale <= 1")
except ValueError as exc:
    raise SystemExit(f"Invalid full-head learning rates: {exc}") from exc
print(f"Prompt LR {peak:g} -> {floor:g}; FULL action head LR {peak * scale:g} -> {floor * scale:g}")
PY
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/piper_runtime_paths.sh"
PIPER_HEAD_GROUP="${RUN_GROUP:-piper-task${TASK}-full-head-$(date +%Y%m%d-%H%M%S)}"
PIPER_HEAD_OUTPUT="${OUTPUT_ROOT:-${WORK_ROOT}/chkpt/2601-lerobot/piper/${PIPER_HEAD_GROUP}}"
PIPER_HEAD_LOG="${LOG_ROOT:-${WORK_ROOT}/logs/piper/${PIPER_HEAD_GROUP}}"
PIPER_HEAD_TAG="full-action-head-t${TOKENS}-b${BATCH_SIZE}-history${GRIPPER_HISTORY_STEPS}"
export RUN_GROUP="${PIPER_HEAD_GROUP}-${PIPER_HEAD_TAG}"
export OUTPUT_ROOT="${PIPER_HEAD_OUTPUT}/${PIPER_HEAD_TAG}" LOG_ROOT="${PIPER_HEAD_LOG}/${PIPER_HEAD_TAG}"
export JOB_NAME="${RUN_GROUP}-task${TASK}-seed${SEED}"
unset WANDB_RUN_ID WANDB_RESUME
export GPU_IDS="${GPU_IDS:-0,1}"
IFS=',' read -r -a PIPER_HEAD_GPUS <<< "${GPU_IDS}"
export NUM_PROCESSES="${#PIPER_HEAD_GPUS[@]}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-30328}"
export FLOW_STEPS="${FLOW_STEPS:-12000}" SAVE_STEPS="${SAVE_STEPS:-[6000,9000,12000]}"
export VLM_PROMPT_TOKENS="${TOKENS}" ACTION_PROMPT_TOKENS="${TOKENS}"
export CHUNK_SIZE=16 N_ACTION_STEPS=8 MASKED_STEPS=12 FIT_EPISODES=all EVAL_SPLIT=0.1
export DTYPE=float32 MIXED_PRECISION=no COMPILE_MODEL=false GRADIENT_CHECKPOINTING=true
export DATA_AUDIT="${DATA_AUDIT:-true}"
export ACTION_EVAL_FREQ=500 ACTION_EVAL_SAMPLES=128 ACTION_UPDATE_FREQ=50
export ACTION_EVAL_SEED="${SEED}" ACTION_EVAL_SAMPLING=episode_stratified ACTION_SELECT_BEST_SAVED=true
echo "Trainable: both ${TOKENS}-token prompt banks and FULL continuous-action head; frozen VLM/vision"
echo "Base: ${PRETRAINED_PATH}; gripper history=${GRIPPER_HISTORY_STEPS}; actual parameter counts printed by trainer"
echo "Batch: per_GPU=${BATCH_SIZE}, processes=${NUM_PROCESSES}, global=$((BATCH_SIZE * NUM_PROCESSES))"
exec bash "${SCRIPT_DIR}/train_piper_autodl.sh" "${TASK}" dual_prompt_only "${SEED}" \
  --policy.train_action_head=true \
  --policy.train_action_projections=true \
  --policy.train_action_expert_last_n_layers=0 \
  "--policy.action_expert_lr_scale=${ACTION_EXPERT_LR_SCALE}" \
  --policy.use_peft=false \
  "--policy.optimizer_lr=${PROMPT_LR}" \
  "--policy.scheduler_decay_lr=${PROMPT_FINAL_LR}" \
  --policy.piper_train_normalization=true \
  "--policy.piper_gripper_history_steps=${GRIPPER_HISTORY_STEPS}" \
  --policy.use_relative_actions=false \
  --dataset.image_transforms.enable=false --log_freq=50
