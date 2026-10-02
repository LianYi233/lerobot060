#!/usr/bin/env bash
# Revised Piper training after a recorded-action fit audit. Existing presets stay unchanged.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_piper_retrain.sh TASK [expert_last2|projections] [SEED]

Default: expert_last2; all episodes before a 10% episode validation split;
750 action priming + 250 bridge + 3000 formal flow updates; FP32; 50 predicted / 8 executed.
Formal flow: prompts/maps peak LR 1e-4; last 2 expert blocks peak LR 1e-5.
Priming/bridge keep the existing base peak LR 2.5e-5, expert blocks 2.5e-6.
VLM, other expert blocks, time maps and final expert norm remain frozen. CABO is off.
Default model: 47,313,952 trainable parameters; this is outside the 0.05M prompt budget.
projections is a matched control with the expert backbone frozen.

Raw parquet action/state audit runs before GPU model allocation. It reports equality
and adjacent-frame errors; it never shifts labels or changes normalization statistics.
ACTION_EVAL_SAMPLES=128 per split, ACTION_EVAL_FREQ=500, episode-stratified sampling.
SAVE_STEPS='[1000,2000,3000]'; best_joint and best_gripper link to the best SAVED models
using validation first-8-action error. Loss alone does not choose a model.

Use DATASET_BASE, PRETRAINED_PATH, TOKENIZER_PATH, WORK_ROOT, OUTPUT_ROOT, LOG_ROOT,
RUN_GROUP, GPU_IDS, NUM_PROCESSES and BATCH_SIZE as with train_piper_fit.sh.
DRY_RUN=true only validates metadata/paths and prints commands (raw audit not executed).
For the separate short-horizon experiment: CHUNK_SIZE=16 MASKED_STEPS=12 N_ACTION_STEPS=8.
Use fresh output directories for every profile and horizon. This starts from PRETRAINED_PATH,
not the previous 12000-step fitted checkpoint. Both training and deployment need this branch.
EOF
  exit 0
fi
TASK="${1:?Specify task 1,2,3,4 or all}"
PROFILE="${2:-expert_last2}"
SEED="${3:-0}"
if (( $# >= 3 )); then shift 3; else shift "$#"; fi
case "${PROFILE}" in
  expert_last2|projections) ;;
  *) echo "Use expert_last2 or projections for this matched retraining experiment" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_GROUP="${RUN_GROUP:-piper-retrain-task${TASK}-${PROFILE}-$(date +%Y%m%d-%H%M%S)}"
export FLOW_STEPS="${FLOW_STEPS:-3000}"
export SAVE_STEPS="${SAVE_STEPS:-[1000,2000,3000]}"
export FIT_EPISODES="${FIT_EPISODES:-all}"
export EVAL_SPLIT="${EVAL_SPLIT:-0.1}"
export DATA_AUDIT="${DATA_AUDIT:-true}"
export ACTION_EVAL_SAMPLES="${ACTION_EVAL_SAMPLES:-128}"
export ACTION_EVAL_FREQ="${ACTION_EVAL_FREQ:-500}"
export ACTION_EVAL_SAMPLING="${ACTION_EVAL_SAMPLING:-episode_stratified}"
export ACTION_SELECT_BEST_SAVED="${ACTION_SELECT_BEST_SAVED:-true}"
exec bash "${SCRIPT_DIR}/train_piper_fit.sh" "${TASK}" "${PROFILE}" "${SEED}" "$@"
