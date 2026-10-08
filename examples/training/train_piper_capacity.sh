#!/usr/bin/env bash
# Matched action-expert capacity experiments using an explicitly selected base.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_piper_capacity.sh [TASK] [last2|last4|last8|last18] [SEED]

Defaults: task 1, last8, seed 0. TASK accepts 1/2/3/4/all.
Set PRETRAINED_PATH explicitly to the SAME initial base for each capacity comparison.
No implicit LIBERO/base/DROID fallback. Do not use a trained Piper checkpoint for only one group.

Train prompts + action input/output projections + final N action-expert blocks:
last2: 47,313,952; last4: 94,512,160; last8: 188,908,576; last18: 424,899,616 parameters.
Counts assume gemma_2b/gemma_300m, 16+16 prompts, max_action_dim=32; see actual training report.
last18 includes all 18 transformer blocks, NOT the frozen final norm or time MLPs.
The VLM and vision encoder remain frozen in every profile.

Fixed comparison: absolute actions, chunk 16 / execute 8, all episodes minus 10% holdout,
16+16 prompts, FP32, no compile, gradient checkpointing, no priming/bridge/CABO.
LR resets to prompts/projections 1e-4 -> 1e-5, expert blocks 1e-5 -> 1e-6.
Old LR/EXPERT_LAST_N_LAYERS/FIT_EPISODES/CHUNK_SIZE exports are replaced deliberately.
Default 12000 flow updates, saves [6000,9000,12000]; override FLOW_STEPS and SAVE_STEPS together.
GPU_IDS defaults to 0,1; NUM_PROCESSES is inferred from that list; BATCH_SIZE defaults to 16 per GPU.
MAIN_PROCESS_PORT defaults to 29500 + N; choose different ports for concurrent runs of one profile.

DATASET_BASE, TOKENIZER_PATH, WORK_ROOT, BATCH_SIZE and W&B settings work as usual.
RUN_GROUP is a parent label. OUTPUT_ROOT / LOG_ROOT are PARENT directories here:
each gets a capacity-PROFILE subdirectory, preventing cross-profile output collisions.
Re-running the same profile in the same directory still refuses to overwrite it.
W&B run IDs are cleared; action errors and parameter-update diagnostics remain enabled.
DRY_RUN=true validates paths/metadata and prints the command; no model/GPU/robot is used.
No extra CLI args; use train_piper_direct.sh for custom recipes.
See examples/piper/README_CAPACITY.md for commands and interpretation of loss.
EOF
  exit 0
fi
TASK="${1:-1}"
PROFILE="${2:-last8}"
SEED="${3:-0}"
if (( $# > 3 )) || [[ ! "${TASK}" =~ ^([1-4]|all)$ ]]; then
  echo "Use TASK (1..4 or all), last2|last4|last8|last18, SEED; no extra CLI arguments" >&2
  exit 2
fi
case "${PROFILE}" in
  last2) export EXPERT_LAST_N_LAYERS=2 ;;
  last4) export EXPERT_LAST_N_LAYERS=4 ;;
  last8) export EXPERT_LAST_N_LAYERS=8 ;;
  last18) export EXPERT_LAST_N_LAYERS=18 ;;
  *) echo "Unknown capacity profile: ${PROFILE}; use last2, last4, last8 or last18" >&2; exit 2 ;;
esac
if [[ -z "${PRETRAINED_PATH:-}" ]]; then
  echo "Set PRETRAINED_PATH explicitly to the local initial base checkpoint for this comparison" >&2
  exit 2
fi
export PRETRAINED_PATH
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
PIPER_CAPACITY_GROUP="${RUN_GROUP:-piper-task${TASK}-capacity-$(date +%Y%m%d-%H%M%S)}"
PIPER_CAPACITY_OUTPUT="${OUTPUT_ROOT:-${WORK_ROOT}/chkpt/2601-lerobot/piper/${PIPER_CAPACITY_GROUP}}"
PIPER_CAPACITY_LOG="${LOG_ROOT:-${WORK_ROOT}/logs/piper/${PIPER_CAPACITY_GROUP}}"
export RUN_GROUP="${PIPER_CAPACITY_GROUP}-capacity-${PROFILE}"
export OUTPUT_ROOT="${PIPER_CAPACITY_OUTPUT}/capacity-${PROFILE}"
export LOG_ROOT="${PIPER_CAPACITY_LOG}/capacity-${PROFILE}"
export JOB_NAME="${RUN_GROUP}-task${TASK}-seed${SEED}"
unset WANDB_RUN_ID WANDB_RESUME

export GPU_IDS="${GPU_IDS:-0,1}"
IFS=',' read -r -a PIPER_CAPACITY_GPUS <<< "${GPU_IDS}"
export NUM_PROCESSES="${#PIPER_CAPACITY_GPUS[@]}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((29500 + EXPERT_LAST_N_LAYERS))}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export FLOW_STEPS="${FLOW_STEPS:-12000}"
export SAVE_STEPS="${SAVE_STEPS:-[6000,9000,12000]}"
export CHUNK_SIZE=16 N_ACTION_STEPS=8 MASKED_STEPS=12 FIT_EPISODES=all EVAL_SPLIT=0.1
export DTYPE=float32 MIXED_PRECISION=no COMPILE_MODEL=false GRADIENT_CHECKPOINTING=true
export OPTIMIZER_LR=0.0001 SCHEDULER_DECAY_LR=0.00001 ACTION_EXPERT_LR_SCALE=0.1
export ACTION_EVAL_FREQ=500 ACTION_EVAL_SAMPLES=128 ACTION_UPDATE_FREQ=50
export ACTION_EVAL_SEED="${SEED}" ACTION_EVAL_SAMPLING=episode_stratified ACTION_SELECT_BEST_SAVED=true

echo "Piper capacity: ${PROFILE}, initial weights=${PRETRAINED_PATH}, GPUs=${GPU_IDS}"
echo "Expected trainable parameters (default architecture): $((115744 + EXPERT_LAST_N_LAYERS * 23599104))"
echo "Frozen: VLM/vision, expert final norm, time MLPs; loss < 0.03 is not a success guarantee."
exec bash "${SCRIPT_DIR}/train_piper_direct.sh" "${TASK}" absolute "${SEED}"
