#!/usr/bin/env bash
set -euo pipefail

# PI0.5 / BridgeVLA fine-tuning on the official LeRobot-format LIBERO-Plus
# training dataset: https://huggingface.co/datasets/lerobot/libero_plus
#
# Default recipe in this branch is matched to the known convergent reference:
#   effective global batch = 256
#   optimizer steps        = 8000
#
# On two GPUs (CUDA devices 2,3), LeRobot's --batch_size is PER PROCESS/GPU.
# Therefore:
#   16 samples/GPU x 2 GPUs x 8 gradient-accumulation micro-batches = 256.
#
# Importantly, STEPS, PRIOR_STEPS, BRIDGE_STEPS, scheduler progress, CAMB/CABO
# control, and checkpoint cadence are all expressed in optimizer-step units.
# Gradient accumulation is implemented by lerobot_train_accum.py so these
# semantics do not change.

GPU_IDS="${GPU_IDS:-2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
DATASET_REPO="${DATASET_REPO:-lerobot/libero_plus}"
DATASET_ROOT="${DATASET_ROOT:-/data1/datasets/libero_plus_lerobot}"
PRETRAINED="${PRETRAINED:-/data/models/lerobot/pi05_libero_base}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/wyn/chkpt/2601-lerobot/libero-plus-pi05-fmm823-bridge-8k}"
JOB_NAME="${JOB_NAME:-pi05-libero-plus-fmm823-bridge-8k}"
POLICY_REPO_ID="${POLICY_REPO_ID:-yiliawu_repo_id}"

# These are optimizer updates, not micro-steps.
STEPS="${STEPS:-8000}"
PRIOR_STEPS="${PRIOR_STEPS:-1000}"
BRIDGE_STEPS="${BRIDGE_STEPS:-250}"

# Per-GPU batch. With two processes and accumulation=8, the default effective
# global batch is 16 * 2 * 8 = 256.
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"

NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-500}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
CABO_ENABLED="${CABO_ENABLED:-true}"
AUTO_AUGMENT_QUANTILES="${AUTO_AUGMENT_QUANTILES:-true}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export LEROBOT_GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}"

if [[ -n "${DATASET_ROOT}" && ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: DATASET_ROOT does not exist: ${DATASET_ROOT}" >&2
  echo "Download lerobot/libero_plus there first, or override DATASET_ROOT." >&2
  exit 2
fi

# PI0.5 uses QUANTILES normalization for observation.state and action. The
# public LIBERO-Plus LeRobot metadata may not contain q01/q99. Add those
# statistics locally before launching DDP; this reads only parquet state/action
# columns and never decodes videos or contacts the Hub.
if [[ "${AUTO_AUGMENT_QUANTILES}" == "true" && -n "${DATASET_ROOT}" ]]; then
  echo "Checking PI0.5 quantile statistics in ${DATASET_ROOT}/meta/stats.json ..."
  python -m lerobot.scripts.augment_libero_plus_quantiles \
    --root "${DATASET_ROOT}"
fi

EFFECTIVE_BATCH=$((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))

DATASET_ARGS=(
  --dataset.repo_id="${DATASET_REPO}"
  --dataset.video_backend="${VIDEO_BACKEND}"
)
if [[ -n "${DATASET_ROOT}" ]]; then
  DATASET_ARGS+=(--dataset.root="${DATASET_ROOT}")
fi

echo "=== PI0.5 LIBERO-Plus training ==="
echo "GPUs:                    ${GPU_IDS}"
echo "Processes:               ${NUM_PROCESSES}"
echo "Dataset repo:            ${DATASET_REPO}"
echo "Dataset root:            ${DATASET_ROOT:-<HF cache>}"
echo "Pretrained:              ${PRETRAINED}"
echo "Output:                  ${OUTPUT_DIR}"
echo "Optimizer steps:         ${STEPS}"
echo "Batch / GPU:             ${BATCH_SIZE}"
echo "Gradient accumulation:   ${GRAD_ACCUM_STEPS}"
echo "Effective global batch:  ${BATCH_SIZE} x ${NUM_PROCESSES} x ${GRAD_ACCUM_STEPS} = ${EFFECTIVE_BATCH}"
echo "Action Prior / Bridge:   ${PRIOR_STEPS} / ${BRIDGE_STEPS} optimizer steps"
echo "CAMB/CABO enabled:       ${CABO_ENABLED}"
echo "Checkpoint frequency:    every ${SAVE_FREQ} optimizer steps"

accelerate launch \
  --multi_gpu \
  --num_processes="${NUM_PROCESSES}" \
  --mixed_precision=bf16 \
  --module lerobot.scripts.lerobot_train_accum \
  "${DATASET_ARGS[@]}" \
  --policy.type=pi05 \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --policy.pretrained_path="${PRETRAINED}" \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.next_action_pretrain_steps="${PRIOR_STEPS}" \
  --policy.next_action_bridge_steps="${BRIDGE_STEPS}" \
  --policy.cabo_enabled="${CABO_ENABLED}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}" \
  --env_eval_freq=0 \
  --policy.device=cuda \
  --wandb.enable=true
