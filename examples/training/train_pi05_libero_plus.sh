#!/usr/bin/env bash
set -euo pipefail

# PI0.5 / BridgeVLA fine-tuning on the official LeRobot-format LIBERO-Plus
# training dataset: https://huggingface.co/datasets/lerobot/libero_plus
#
# The dataset contains observation.images.front / observation.images.wrist,
# observation.state (8D), and action (7D). Because this recipe uses
# --policy.type=pi05 together with --policy.pretrained_path, PI0.5 derives its
# input feature names from the current dataset; therefore no --rename_map is
# needed during training.
#
# Typical usage:
#   bash examples/training/train_pi05_libero_plus.sh
#
# Local dataset mirror:
#   DATASET_ROOT=/data1/datasets/libero_plus \
#     bash examples/training/train_pi05_libero_plus.sh
#
# Longer run after a 6k pilot:
#   STEPS=12000 bash examples/training/train_pi05_libero_plus.sh

GPU_IDS="${GPU_IDS:-0,1,2}"
NUM_PROCESSES="${NUM_PROCESSES:-3}"
DATASET_REPO="${DATASET_REPO:-lerobot/libero_plus}"
DATASET_ROOT="${DATASET_ROOT:-}"
PRETRAINED="${PRETRAINED:-/data/models/lerobot/pi05_libero_base}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/wyn/chkpt/2601-lerobot/libero-plus-pi05-fmm823-bridge-6k}"
JOB_NAME="${JOB_NAME:-pi05-libero-plus-fmm823-bridge-6k}"
POLICY_REPO_ID="${POLICY_REPO_ID:-yiliawu_repo_id}"
STEPS="${STEPS:-6000}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-500}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

# Preserve the successful BridgeVLA schedule used on LIBERO unless explicitly
# overridden. The first 1000 updates prepare the action pathway; the final 250
# of those use observation-conditioned flow under a frozen VLM.
PRIOR_STEPS="${PRIOR_STEPS:-1000}"
BRIDGE_STEPS="${BRIDGE_STEPS:-250}"
CABO_ENABLED="${CABO_ENABLED:-true}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

DATASET_ARGS=(
  --dataset.repo_id="${DATASET_REPO}"
  --dataset.video_backend="${VIDEO_BACKEND}"
)
if [[ -n "${DATASET_ROOT}" ]]; then
  DATASET_ARGS+=(--dataset.root="${DATASET_ROOT}")
fi

echo "=== PI0.5 LIBERO-Plus training ==="
echo "GPUs:             ${GPU_IDS}"
echo "Processes:        ${NUM_PROCESSES}"
echo "Dataset repo:     ${DATASET_REPO}"
echo "Dataset root:     ${DATASET_ROOT:-<HF cache>}"
echo "Pretrained:       ${PRETRAINED}"
echo "Output:           ${OUTPUT_DIR}"
echo "Steps:            ${STEPS}"
echo "Batch size/GPU:   $((BATCH_SIZE / NUM_PROCESSES)) (global ${BATCH_SIZE})"
echo "Prior/Bridge:     ${PRIOR_STEPS}/${BRIDGE_STEPS}"
echo "CABO:             ${CABO_ENABLED}"

accelerate launch \
  --multi_gpu \
  --num_processes="${NUM_PROCESSES}" \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
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
