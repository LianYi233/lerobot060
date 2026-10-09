#!/usr/bin/env bash
# Strict two-bank prompt tuning: no trainable expert blocks or action projections.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_piper_prompt_only.sh [TASK] [16|32|64] [SEED]

Defaults: task 1, 32 tokens PER BANK, seed 0. TASK accepts 1/2/3/4/all.
32 means VLM 32x2048 + action 32x1024 = 98,304 trainable parameters.
64 means VLM 64x2048 + action 64x1024 = 196,608 trainable parameters.
16 is a matched 49,152-parameter control. Counts assume gemma_2b/gemma_300m.
Only vlm_prompt_tokens.weight and prompt_tokens.weight train. Both backbones, action
input/output projections, time MLPs and final norm stay frozen; PEFT wrappers are off.
The training parameter report must show action_projection=0, action_expert_blocks=0, other=0.

Set PRETRAINED_PATH to the SAME original base checkpoint in every group.
Do not initialize from a Piper last4/last18 model, which already has adapted backbone weights.
Default: 12000 direct flow updates, save [6000,9000,12000], no priming/bridge/CABO,
absolute actions, chunk16/execute8, all episodes minus 10% holdout, FP32, no compile.
Prompt LR: 5e-5 -> 5e-6 with warmup/cosine. Override PROMPT_LR / PROMPT_FINAL_LR together.
GPU_IDS defaults to 0,1; NUM_PROCESSES inferred from the list; BATCH_SIZE defaults to 64 PER GPU.
BATCH_SIZE=128 is supported; memory must be measured with the chosen token count.
More prompt tokens still require gradients through frozen backbones and may increase memory.

Old EXPERT_LAST_N_LAYERS, token-count and optimizer-LR exports are replaced deliberately.
DATASET_BASE, TOKENIZER_PATH, WORK_ROOT, FLOW_STEPS/SAVE_STEPS and W&B settings work as usual.
RUN_GROUP is a parent label. OUTPUT_ROOT / LOG_ROOT are PARENT directories; each gets
prompt-only-tTOKENS-bBATCH (per-GPU batch). Different tokens/batches do not overwrite each other.
Changing LR, base or process count requires a new parent run directory.
MAIN_PROCESS_PORT defaults to 30000 + TOKENS; override for concurrent batches/seeds of one token count.
DRY_RUN=true prints the command and exits successfully without loading weights or using GPUs.
No extra CLI arguments. See examples/piper/README_PROMPT_ONLY.md.
EOF
  exit 0
fi
TASK="${1:-1}"
TOKENS="${2:-32}"
SEED="${3:-0}"
if (( $# > 3 )) || [[ ! "${TASK}" =~ ^([1-4]|all)$ ]]; then
  echo "Use TASK (1..4 or all), TOKENS (16/32/64), SEED; no extra CLI arguments" >&2
  exit 2
fi
case "${TOKENS}" in
  16|32|64) ;;
  *) echo "TOKENS must be 16, 32 or 64 PER BANK" >&2; exit 2 ;;
esac
if [[ -z "${PRETRAINED_PATH:-}" ]]; then
  echo "Set PRETRAINED_PATH explicitly to the local original base checkpoint" >&2
  exit 2
fi
export PRETRAINED_PATH
export BATCH_SIZE="${BATCH_SIZE:-64}"
if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
export PROMPT_LR="${PROMPT_LR:-0.00005}"
export PROMPT_FINAL_LR="${PROMPT_FINAL_LR:-0.000005}"
"${PYTHON:-python}" - <<'PY'
import math
import os

try:
    peak = float(os.environ["PROMPT_LR"])
    floor = float(os.environ["PROMPT_FINAL_LR"])
    if not all(math.isfinite(value) and value > 0 for value in (peak, floor)) or floor > peak:
        raise ValueError("require finite positive LR and PROMPT_FINAL_LR <= PROMPT_LR")
except ValueError as exc:
    raise SystemExit(f"Invalid prompt learning rates: {exc}") from exc
print(f"Both prompt banks: peak LR={peak:g}, final LR={floor:g}; warmup + cosine")
PY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
PIPER_PROMPT_GROUP="${RUN_GROUP:-piper-task${TASK}-prompt-only-$(date +%Y%m%d-%H%M%S)}"
PIPER_PROMPT_OUTPUT="${OUTPUT_ROOT:-${WORK_ROOT}/chkpt/2601-lerobot/piper/${PIPER_PROMPT_GROUP}}"
PIPER_PROMPT_LOG="${LOG_ROOT:-${WORK_ROOT}/logs/piper/${PIPER_PROMPT_GROUP}}"
PIPER_PROMPT_TAG="prompt-only-t${TOKENS}-b${BATCH_SIZE}"
export RUN_GROUP="${PIPER_PROMPT_GROUP}-${PIPER_PROMPT_TAG}"
export OUTPUT_ROOT="${PIPER_PROMPT_OUTPUT}/${PIPER_PROMPT_TAG}"
export LOG_ROOT="${PIPER_PROMPT_LOG}/${PIPER_PROMPT_TAG}"
export JOB_NAME="${RUN_GROUP}-task${TASK}-seed${SEED}"
unset WANDB_RUN_ID WANDB_RESUME

export GPU_IDS="${GPU_IDS:-0,1}"
IFS=',' read -r -a PIPER_PROMPT_GPUS <<< "${GPU_IDS}"
export NUM_PROCESSES="${#PIPER_PROMPT_GPUS[@]}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((30000 + TOKENS))}"
export FLOW_STEPS="${FLOW_STEPS:-12000}"
export SAVE_STEPS="${SAVE_STEPS:-[6000,9000,12000]}"
export VLM_PROMPT_TOKENS="${TOKENS}" ACTION_PROMPT_TOKENS="${TOKENS}"
export CHUNK_SIZE=16 N_ACTION_STEPS=8 MASKED_STEPS=12 FIT_EPISODES=all EVAL_SPLIT=0.1
export DTYPE=float32 MIXED_PRECISION=no COMPILE_MODEL=false GRADIENT_CHECKPOINTING=true
export DATA_AUDIT="${DATA_AUDIT:-true}"
export ACTION_EVAL_FREQ=500 ACTION_EVAL_SAMPLES=128 ACTION_UPDATE_FREQ=50
export ACTION_EVAL_SEED="${SEED}" ACTION_EVAL_SAMPLING=episode_stratified ACTION_SELECT_BEST_SAVED=true

echo "Piper prompt-only: VLM=${TOKENS}x2048, action=${TOKENS}x1024; expected trainable=$((TOKENS * 3072))"
echo "Trainable ONLY: vlm_prompt_tokens.weight, prompt_tokens.weight; expert layers=0, action projections=false"
echo "Batch: per_GPU=${BATCH_SIZE}, processes=${NUM_PROCESSES}, global=$((BATCH_SIZE * NUM_PROCESSES))"
exec bash "${SCRIPT_DIR}/train_piper_autodl.sh" "${TASK}" dual_prompt_only "${SEED}" \
  --policy.train_action_projections=false \
  --policy.train_action_expert_last_n_layers=0 \
  --policy.use_peft=false \
  "--policy.optimizer_lr=${PROMPT_LR}" \
  "--policy.scheduler_decay_lr=${PROMPT_FINAL_LR}" \
  --policy.piper_train_normalization=true \
  --policy.use_relative_actions=false \
  --dataset.image_transforms.enable=false --log_freq=50
