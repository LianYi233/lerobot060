#!/usr/bin/env bash
set -euo pipefail

# Evaluate a PI0.5 checkpoint trained on lerobot/libero_plus.
#
# Important: the training dataset uses observation.images.front/wrist, while
# the LIBERO(-Plus) environment processor emits observation.images.image/image2.
# The rename_map below maps environment keys to the feature names baked into a
# checkpoint trained with --policy.pretrained_path on lerobot/libero_plus.
#
# Usage:
#   POLICY_PATH=/path/to/pretrained_model \
#     bash examples/training/eval_pi05_libero_plus.sh

POLICY_PATH="${POLICY_PATH:?Set POLICY_PATH to the checkpoint pretrained_model directory}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/wyn/eval/libero-plus-pi05}"
N_EPISODES="${N_EPISODES:-1}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
TASKS="${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

lerobot-eval \
  --policy.path="${POLICY_PATH}" \
  --policy.n_action_steps="${N_ACTION_STEPS}" \
  --env.type=libero \
  --env.is_libero_plus=true \
  --env.task="${TASKS}" \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes="${N_EPISODES}" \
  --rename_map='{"observation.images.image":"observation.images.front","observation.images.image2":"observation.images.wrist"}' \
  --output_dir="${OUTPUT_DIR}"
