#!/usr/bin/env bash
# Action-expert capacity and LR/batch experiments using an explicitly selected base.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_piper_capacity.sh [TASK] [last2|last4|last8|last18] [SEED] [RECIPE]

Defaults: task 1, last8, seed 0, reference. TASK accepts 1/2/3/4/all.
Set PRETRAINED_PATH explicitly to the SAME initial base for each capacity comparison.
No implicit LIBERO/base/DROID fallback. Do not use a trained Piper checkpoint for only one group.

Train prompts + action input/output projections + final N action-expert blocks:
last2: 47,313,952; last4: 94,512,160; last8: 188,908,576; last18: 424,899,616 parameters.
Counts assume gemma_2b/gemma_300m, 16+16 prompts, max_action_dim=32; see actual training report.
last18 includes all 18 transformer blocks, NOT the frozen final norm or time MLPs.
The VLM and vision encoder remain frozen in every profile.

Fixed comparison: absolute actions, chunk 16 / execute 8, all episodes minus 10% holdout,
16+16 prompts, FP32, no compile, gradient checkpointing, no priming/bridge/CABO.
RECIPE            prompts/maps LR   expert LR       default batch/GPU   port base
reference         1e-4 -> 1e-5      1e-5 -> 1e-6    16                  29500
low_lr            5e-5 -> 5e-6      5e-6 -> 5e-7    16                  29600
big_batch         1e-4 -> 1e-5      1e-5 -> 1e-6    32                  29700
low_lr_big_batch   5e-5 -> 5e-6      5e-6 -> 5e-7    32                  29800
Both LR groups retain warmup + cosine; no LR scaling is applied automatically for larger batches.
The selected recipe replaces old LR/EXPERT_LAST_N_LAYERS/FIT_EPISODES/CHUNK_SIZE exports.
BATCH_SIZE overrides the recipe default, including an inherited export. Check the printed actual batch.
Default 12000 flow updates, saves [6000,9000,12000]; override FLOW_STEPS and SAVE_STEPS together.
GPU_IDS defaults to 0,1; NUM_PROCESSES is inferred from that list (no gradient accumulation added).
MAIN_PROCESS_PORT defaults to recipe port base + N; override for concurrent runs of the same profile/recipe.
At equal update counts, doubling global batch doubles training sample slots; this is NOT equal compute.

DATASET_BASE, TOKENIZER_PATH, WORK_ROOT, BATCH_SIZE and W&B settings work as usual.
RUN_GROUP is a parent label. OUTPUT_ROOT / LOG_ROOT are PARENT directories here:
reference keeps the original capacity-PROFILE subdirectory; other recipes append -RECIPE.
Profiles/recipes have separate directories and W&B names. Re-running an existing run refuses overwrite.
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
RECIPE="${4:-reference}"
if (( $# > 4 )) || [[ ! "${TASK}" =~ ^([1-4]|all)$ ]]; then
  echo "Use TASK (1..4 or all), last2|last4|last8|last18, SEED, RECIPE; no extra CLI arguments" >&2
  exit 2
fi
case "${PROFILE}" in
  last2) export EXPERT_LAST_N_LAYERS=2 ;;
  last4) export EXPERT_LAST_N_LAYERS=4 ;;
  last8) export EXPERT_LAST_N_LAYERS=8 ;;
  last18) export EXPERT_LAST_N_LAYERS=18 ;;
  *) echo "Unknown capacity profile: ${PROFILE}; use last2, last4, last8 or last18" >&2; exit 2 ;;
esac
case "${RECIPE}" in
  reference|big_batch)
    export OPTIMIZER_LR=0.0001 SCHEDULER_DECAY_LR=0.00001
    ;;
  low_lr|low_lr_big_batch)
    export OPTIMIZER_LR=0.00005 SCHEDULER_DECAY_LR=0.000005
    ;;
  *) echo "Unknown capacity recipe: ${RECIPE}; use reference, low_lr, big_batch or low_lr_big_batch" >&2; exit 2 ;;
esac
export ACTION_EXPERT_LR_SCALE=0.1
case "${RECIPE}" in
  reference) PIPER_RECIPE_BATCH=16; PIPER_RECIPE_PORT_BASE=29500 ;;
  low_lr) PIPER_RECIPE_BATCH=16; PIPER_RECIPE_PORT_BASE=29600 ;;
  big_batch) PIPER_RECIPE_BATCH=32; PIPER_RECIPE_PORT_BASE=29700 ;;
  low_lr_big_batch) PIPER_RECIPE_BATCH=32; PIPER_RECIPE_PORT_BASE=29800 ;;
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
PIPER_CAPACITY_TAG="capacity-${PROFILE}"
if [[ "${RECIPE}" != reference ]]; then
  PIPER_CAPACITY_TAG+="-${RECIPE}"
fi
export RUN_GROUP="${PIPER_CAPACITY_GROUP}-${PIPER_CAPACITY_TAG}"
export OUTPUT_ROOT="${PIPER_CAPACITY_OUTPUT}/${PIPER_CAPACITY_TAG}"
export LOG_ROOT="${PIPER_CAPACITY_LOG}/${PIPER_CAPACITY_TAG}"
export JOB_NAME="${RUN_GROUP}-task${TASK}-seed${SEED}"
unset WANDB_RUN_ID WANDB_RESUME

export GPU_IDS="${GPU_IDS:-0,1}"
IFS=',' read -r -a PIPER_CAPACITY_GPUS <<< "${GPU_IDS}"
export NUM_PROCESSES="${#PIPER_CAPACITY_GPUS[@]}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((PIPER_RECIPE_PORT_BASE + EXPERT_LAST_N_LAYERS))}"
export BATCH_SIZE="${BATCH_SIZE:-${PIPER_RECIPE_BATCH}}"
if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
export FLOW_STEPS="${FLOW_STEPS:-12000}"
export SAVE_STEPS="${SAVE_STEPS:-[6000,9000,12000]}"
export CHUNK_SIZE=16 N_ACTION_STEPS=8 MASKED_STEPS=12 FIT_EPISODES=all EVAL_SPLIT=0.1
export DTYPE=float32 MIXED_PRECISION=no COMPILE_MODEL=false GRADIENT_CHECKPOINTING=true
export ACTION_EVAL_FREQ=500 ACTION_EVAL_SAMPLES=128 ACTION_UPDATE_FREQ=50
export ACTION_EVAL_SEED="${SEED}" ACTION_EVAL_SAMPLING=episode_stratified ACTION_SELECT_BEST_SAVED=true

echo "Piper capacity: ${PROFILE}, recipe=${RECIPE}, initial weights=${PRETRAINED_PATH}, GPUs=${GPU_IDS}"
echo "Batch: per_GPU=${BATCH_SIZE}, processes=${NUM_PROCESSES}, global=$((BATCH_SIZE * NUM_PROCESSES)); recipe default per_GPU=${PIPER_RECIPE_BATCH}"
echo "Expected trainable parameters (default architecture): $((115744 + EXPERT_LAST_N_LAYERS * 23599104))"
echo "Frozen: VLM/vision, expert final norm, time MLPs; evaluate physical action errors as well as training loss."
exec bash "${SCRIPT_DIR}/train_piper_direct.sh" "${TASK}" absolute "${SEED}"
