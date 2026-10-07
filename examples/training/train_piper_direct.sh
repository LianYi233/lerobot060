#!/usr/bin/env bash
# Matched Piper experiments: direct flow, absolute vs relative joints; no priming/bridge.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: bash examples/training/train_piper_direct.sh TASK [absolute|relative] [SEED] [extra args...]

12000 TOTAL updates, all observation-conditioned flow; priming=0, bridge=0, CABO off.
Both profiles train prompts, action projections and last two expert blocks by default (47,313,952 params).
EXPERT_LAST_N_LAYERS=0..18 changes the number of trainable final expert blocks.
absolute: predict absolute joint positions and gripper (control).
relative: predict joint offsets from the current observation; gripper stays absolute.
Both refit state/action normalization on the training split, never the held-out episodes.
Labels on disk are untouched. Inference restores absolute actions using saved processors.

Default: all episodes, 10% episode holdout, chunk 50 / execute 8, FP32, no compile.
Peak LR: prompts/maps 1e-4; expert blocks 1e-5. W&B and action diagnostics enabled.
Override OPTIMIZER_LR, SCHEDULER_DECAY_LR and ACTION_EXPERT_LR_SCALE (defaults 1e-4,1e-5,0.1).
Both parameter groups follow the same warmup/cosine multiplier; the expert ratio stays fixed.
Evaluate 128 fixed samples per split every 500 steps. Save only 6000,9000,12000;
best_joint / best_gripper are links to the best SAVED checkpoints, not extra copies.

Paths/GPU/batch: same environment variables as train_piper_autodl.sh.
Use fresh RUN_GROUP, OUTPUT_ROOT and LOG_ROOT for each profile; use the same base weights.
For another budget set FLOW_STEPS and SAVE_STEPS together.
DRY_RUN=true validates paths/metadata and prints commands; it does not train or read raw rows.
The underlying run name contains dual_prompt_only (stage recipe); expert blocks ARE unfrozen.
EOF
  exit 0
fi
TASK="${1:?Specify task 1,2,3,4 or all}"
PROFILE="${2:-absolute}"
SEED="${3:-0}"
if (( $# >= 3 )); then shift 3; else shift "$#"; fi
case "${PROFILE}" in
  absolute) RELATIVE=false ;;
  relative) RELATIVE=true ;;
  *) echo "Use absolute or relative" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_GROUP="${RUN_GROUP:-piper-direct-task${TASK}-${PROFILE}-$(date +%Y%m%d-%H%M%S)}"
export FLOW_STEPS="${FLOW_STEPS:-12000}"
export SAVE_STEPS="${SAVE_STEPS:-[6000,9000,12000]}"
export COMPILE_MODEL="${COMPILE_MODEL:-false}"
export EVAL_SPLIT="${EVAL_SPLIT:-0.1}"
export DATA_AUDIT="${DATA_AUDIT:-true}"
export ACTION_EVAL_SAMPLES="${ACTION_EVAL_SAMPLES:-128}"
export ACTION_EVAL_FREQ="${ACTION_EVAL_FREQ:-500}"
export ACTION_EVAL_SAMPLING="${ACTION_EVAL_SAMPLING:-episode_stratified}"
export ACTION_SELECT_BEST_SAVED="${ACTION_SELECT_BEST_SAVED:-true}"
export EXPERT_LAST_N_LAYERS="${EXPERT_LAST_N_LAYERS:-2}"
export OPTIMIZER_LR="${OPTIMIZER_LR:-0.0001}"
export SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-0.00001}"
export ACTION_EXPERT_LR_SCALE="${ACTION_EXPERT_LR_SCALE:-0.1}"
"${PYTHON:-python}" - <<'PY'
import math
import os

try:
    peak, floor, scale = (float(os.environ[key]) for key in (
        "OPTIMIZER_LR", "SCHEDULER_DECAY_LR", "ACTION_EXPERT_LR_SCALE"
    ))
    if not all(math.isfinite(x) and x > 0 for x in (peak, floor, scale)) or floor > peak:
        raise ValueError("require finite positive values and SCHEDULER_DECAY_LR <= OPTIMIZER_LR")
except ValueError as exc:
    raise SystemExit(f"Invalid Piper learning rates: {exc}") from exc
print(f"Piper LR: prompts/maps peak={peak:g}, final={floor:g}; "
      f"expert peak={peak * scale:g}, final={floor * scale:g}; warmup + cosine")
PY
if [[ ! "${EXPERT_LAST_N_LAYERS}" =~ ^([0-9]|1[0-8])$ ]]; then
  echo "EXPERT_LAST_N_LAYERS must be an integer from 0 to 18" >&2
  exit 2
fi
FIT_ARGS=()
if [[ "${FIT_EPISODES:-all}" != all ]]; then
  FIT_ARGS+=("--dataset.episodes=${FIT_EPISODES}")
fi
echo "Piper direct: profile=${PROFILE}, total flow updates=${FLOW_STEPS}, priming=0, bridge=0"
echo "Trainable: both prompt banks, action projections, last ${EXPERT_LAST_N_LAYERS} expert blocks"
exec bash "${SCRIPT_DIR}/train_piper_autodl.sh" "${TASK}" dual_prompt_only "${SEED}" \
  --policy.train_action_projections=true \
  "--policy.train_action_expert_last_n_layers=${EXPERT_LAST_N_LAYERS}" \
  "--policy.action_expert_lr_scale=${ACTION_EXPERT_LR_SCALE}" \
  "--policy.optimizer_lr=${OPTIMIZER_LR}" \
  "--policy.scheduler_decay_lr=${SCHEDULER_DECAY_LR}" \
  --policy.piper_train_normalization=true \
  "--policy.use_relative_actions=${RELATIVE}" \
  --dataset.image_transforms.enable=false --log_freq=50 "${FIT_ARGS[@]}" "$@"
