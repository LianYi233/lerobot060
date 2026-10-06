#!/usr/bin/env bash
# Compare the generic LeRobot PI05 base with the previous LIBERO initialization.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_piper_base.sh TASK [last2|last4] [SEED] [extra args...]

TASK: 1, 2, 3, 4 or all (four independent policies, trained sequentially).
last2 (default): prompts + action projections + final 2 expert blocks (47,313,952 params).
last4: the same recipe with final 4 expert blocks (94,512,160 params).
Counts assume gemma_2b / gemma_300m, 16+16 prompts, max_action_dim=32.
The actual trainable parameter report is printed by lerobot-train.

Download lerobot/pi05_base in LeRobot safetensors format first. Set PI05_BASE_PATH;
default: ${WORK_ROOT:-/root/autodl-tmp}/models/pi05_base.
This entry point deliberately replaces PRETRAINED_PATH with PI05_BASE_PATH so an old
export pointing to pi05_libero_base cannot silently select the previous checkpoint.
A directory name does not verify weight provenance; use the documented Hub download.

Defaults: absolute actions, chunk 16 / execute 8, all episodes with 10% held out,
12000 direct flow updates, priming=0, bridge=0, CABO off, FP32, no compile.
Save 6000/9000/12000. Refit normalization on the Piper training split.
LR: prompts/projections 1e-4; expert blocks 1e-5. W&B and action diagnostics on.
Use last2 first to compare bases; use last4 separately to compare trainable capacity.

DATASET_BASE, TOKENIZER_PATH, GPU_IDS, NUM_PROCESSES, BATCH_SIZE, RUN_GROUP,
OUTPUT_ROOT, LOG_ROOT and DRY_RUN work as in train_piper_direct.sh.
CHUNK_SIZE / N_ACTION_STEPS and FLOW_STEPS / SAVE_STEPS can override the defaults.
Choose fresh output/log roots for each run. No robot hardware is accessed.
See examples/piper/README_BASE_MODEL.md for download, training and comparison commands.
EOF
  exit 0
fi
TASK="${1:?Specify task 1,2,3,4 or all}"
PROFILE="${2:-last2}"
SEED="${3:-0}"
if (( $# >= 3 )); then shift 3; else shift "$#"; fi
case "${PROFILE}" in
  last2) export EXPERT_LAST_N_LAYERS=2 ;;
  last4) export EXPERT_LAST_N_LAYERS=4 ;;
  *) echo "Unknown base profile: ${PROFILE}; use last2 or last4" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
export PI05_BASE_PATH="${PI05_BASE_PATH:-${WORK_ROOT}/models/pi05_base}"
if [[ -n "${PRETRAINED_PATH:-}" && "${PRETRAINED_PATH}" != "${PI05_BASE_PATH}" ]]; then
  echo "Base experiment: replacing PRETRAINED_PATH=${PRETRAINED_PATH} with PI05_BASE_PATH=${PI05_BASE_PATH}"
fi
export PRETRAINED_PATH="${PI05_BASE_PATH}"
export CHUNK_SIZE="${CHUNK_SIZE:-16}"
export N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
export RUN_GROUP="${RUN_GROUP:-piper-task${TASK}-pi05-base-${PROFILE}-h${CHUNK_SIZE}-$(date +%Y%m%d-%H%M%S)}"
echo "Piper base: profile=${PROFILE}, weights=${PRETRAINED_PATH}, chunk=${CHUNK_SIZE}, execute=${N_ACTION_STEPS}"
exec bash "${SCRIPT_DIR}/train_piper_direct.sh" "${TASK}" absolute "${SEED}" "$@"
