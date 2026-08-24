#!/usr/bin/env bash
# Full official LIBERO-plus eval for the public π0.5 LIBERO checkpoint.
#
# Same protocol as eval_flow_mask_checkpoint.sh full-plus:
#   ~10,030 perturbation tasks × 1 trial, per-suite + Camera/Robot/Language/
#   Light/Background/Noise/Layout columns.
#
# Defaults (wyn workstation):
#   policy: /media/wyn/data/10-EmbodiedAI/models/lerobot/pi05_libero_finetuned_v044
#   output: /media/wyn/data/10-EmbodiedAI/2601-lerobot/eval/pi05_libero_finetuned_v044_baseline
#
# Usage:
#   GPU_ID=0 SKIP_GIT=1 bash examples/libero_plus/eval_pi05_baseline.sh

set -euo pipefail

export MODE="${MODE:-full-plus}"
export CHKPT_ROOT="${CHKPT_ROOT:-/media/wyn/data/10-EmbodiedAI/models/lerobot/pi05_libero_finetuned_v044}"
export OUTPUT_DIR="${OUTPUT_DIR:-/media/wyn/data/10-EmbodiedAI/2601-lerobot/eval/pi05_libero_finetuned_v044_baseline}"
export SKIP_GIT="${SKIP_GIT:-1}"
export GPU_ID="${GPU_ID:-0}"
export CONTROL_MODE="${CONTROL_MODE:-relative}"
export N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
export N_EPISODES="${N_EPISODES:-1}"
export TASK="${TASK:-libero_spatial,libero_object,libero_goal,libero_10}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/eval_flow_mask_checkpoint.sh" "${MODE}"
