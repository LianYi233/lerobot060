#!/usr/bin/env bash
# Matched Piper experiment initialized from LeRobot's PI05 DROID/Franka weights.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_piper_droid.sh [TASK] [last2|last4] [SEED] [extra args...]

Defaults: task 1, last2, seed 0. TASK also accepts 2/3/4/all.
Download lerobot/pi05_droid with examples/piper/download_pi05_droid.py first.
PI05_DROID_PATH defaults to ${WORK_ROOT:-/root/autodl-tmp}/models/pi05_droid.
This entry point replaces inherited PI05_BASE_PATH / PRETRAINED_PATH with PI05_DROID_PATH.

Same training recipe as train_piper_base.sh: absolute Piper actions, predict 16 / execute 8,
12000 flow updates, no priming/bridge/CABO, 10% held-out episodes, FP32, no compile.
last2 trains prompts + projections + final 2 expert blocks (47,313,952 parameters).
last4 trains the final 4 expert blocks instead (94,512,160 parameters).
All data, GPU, batch, budget and logging overrides from train_piper_base.sh still apply.

Only initial model weights come from DROID. Piper feature names and train-split statistics
are rebuilt; the DROID robot's action mapping, processor statistics and horizon 15 are not used.
Use a fresh RUN_GROUP / OUTPUT_ROOT / LOG_ROOT. No robot hardware is accessed.
See examples/piper/README_DROID_BASE.md for download, training and output paths.
EOF
  exit 0
fi
TASK="${1:-1}"
PROFILE="${2:-last2}"
SEED="${3:-0}"
if (( $# >= 3 )); then shift 3; else shift "$#"; fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
export PI05_DROID_PATH="${PI05_DROID_PATH:-${WORK_ROOT}/models/pi05_droid}"
if [[ -n "${PI05_BASE_PATH:-}" && "${PI05_BASE_PATH}" != "${PI05_DROID_PATH}" ]]; then
  echo "DROID experiment: replacing PI05_BASE_PATH=${PI05_BASE_PATH} with ${PI05_DROID_PATH}"
fi
export PI05_BASE_PATH="${PI05_DROID_PATH}"
export CHUNK_SIZE="${CHUNK_SIZE:-16}"
export N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
export RUN_GROUP="${RUN_GROUP:-piper-task${TASK}-pi05-droid-${PROFILE}-h${CHUNK_SIZE}-$(date +%Y%m%d-%H%M%S)}"
echo "Piper DROID initialization: ${PI05_DROID_PATH}"
exec bash "${SCRIPT_DIR}/train_piper_base.sh" "${TASK}" "${PROFILE}" "${SEED}" "$@"
