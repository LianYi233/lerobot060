#!/usr/bin/env bash
# Matched two-GPU DROID runs: halve the entire LR schedule, or keep the original.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_piper_droid_lr.sh [TASK] [low|reference] [SEED]

Defaults: task 1, low, seed 0. TASK accepts 1/2/3/4.
low:       GPUs 0,1; port 29501; prompts/maps LR 5e-5 -> 5e-6; expert 5e-6 -> 5e-7.
reference: GPUs 2,3; port 29502; prompts/maps LR 1e-4 -> 1e-5; expert 1e-5 -> 1e-6.
Same DROID weights, absolute actions, last2, 12000 flow updates, warmup/cosine schedule.
Each process defaults to batch 16 (global batch 32). Predict 16 / execute 8.
All episodes with 10% held out; same seed and fixed action-evaluation samples.
Both runs start from the DROID initialization, NOT a previous Piper checkpoint.

Set WORK_ROOT, PI05_DROID_PATH, DATASET_BASE and TOKENIZER_PATH as usual.
RUN_GROUP is the parent comparison name; -lr-low/-lr-reference is appended.
OUTPUT_ROOT and LOG_ROOT are PARENT directories here: each gets an lr-PROFILE subdirectory.
GPU_IDS, NUM_PROCESSES, MAIN_PROCESS_PORT, JOB_NAME and LR variables from old terminal
exports are replaced. To change GPU pairs/ports use LR_LOW_GPU_IDS / LR_REFERENCE_GPU_IDS
and LR_LOW_PORT / LR_REFERENCE_PORT; keep the two groups and ports distinct.
Other training environment overrides still apply: keep them identical in both terminals.
No extra CLI args: use train_piper_droid.sh for a custom recipe.
DRY_RUN=true prints the full effective command without loading weights or using GPUs.
See examples/piper/README_DROID_LR.md for two-terminal commands and comparison metrics.
EOF
  exit 0
fi
TASK="${1:-1}"
LR_PROFILE="${2:-low}"
SEED="${3:-0}"
if (( $# > 3 )) || [[ ! "${TASK}" =~ ^[1-4]$ ]]; then
  echo "Use TASK (1..4), low|reference, SEED; no extra CLI arguments" >&2
  exit 2
fi
case "${LR_PROFILE}" in
  low)
    export GPU_IDS="${LR_LOW_GPU_IDS:-0,1}"
    export MAIN_PROCESS_PORT="${LR_LOW_PORT:-29501}"
    export OPTIMIZER_LR=0.00005 SCHEDULER_DECAY_LR=0.000005
    ;;
  reference)
    export GPU_IDS="${LR_REFERENCE_GPU_IDS:-2,3}"
    export MAIN_PROCESS_PORT="${LR_REFERENCE_PORT:-29502}"
    export OPTIMIZER_LR=0.0001 SCHEDULER_DECAY_LR=0.00001
    ;;
  *) echo "Unknown LR profile: ${LR_PROFILE}; use low or reference" >&2; exit 2 ;;
esac
export ACTION_EXPERT_LR_SCALE=0.1
export NUM_PROCESSES=2
export BATCH_SIZE="${BATCH_SIZE:-16}"
export WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
COMPARISON_GROUP="${RUN_GROUP:-piper-task${TASK}-droid-lr-$(date +%Y%m%d-%H%M%S)}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/chkpt/2601-lerobot/piper/${COMPARISON_GROUP}}/lr-${LR_PROFILE}"
export LOG_ROOT="${LOG_ROOT:-${WORK_ROOT}/logs/piper/${COMPARISON_GROUP}}/lr-${LR_PROFILE}"
export RUN_GROUP="${COMPARISON_GROUP}-lr-${LR_PROFILE}"
export JOB_NAME="${RUN_GROUP}-task${TASK}-seed${SEED}"
# An inherited W&B identity could merge two experiments. Each fresh run gets its own ID.
unset WANDB_RUN_ID WANDB_RESUME
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "DROID LR comparison: ${LR_PROFILE}; GPUs=${GPU_IDS}; port=${MAIN_PROCESS_PORT}; job=${JOB_NAME}"
exec bash "${SCRIPT_DIR}/train_piper_droid.sh" "${TASK}" last2 "${SEED}"
