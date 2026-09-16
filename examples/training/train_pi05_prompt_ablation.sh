#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash train_pi05_prompt_ablation.sh VARIANT [SEED] [extra lerobot-train args...]

Variants:
  full_reference    750 inpainting + 250 bridge + 6000 flow, dual prompts, CABO
  no_bridge         1000 inpainting + 0 bridge + 6000 flow, dual prompts, CABO
  direct_dual       0 pretraining + 7000 flow, dual prompts, CABO
  dual_prompt_only  0 pretraining + 7000 flow, dual prompts, CABO off
  vlm_only          0 pretraining + 7000 flow, VLM prompt only, CABO off
  action_only       0 pretraining + 7000 flow, action prompt only, CABO off
  no_cabo           750 inpainting + 250 bridge + 6000 flow, dual prompts, CABO off

The direct variants use 7000 flow updates so that their total update budget matches
the 1000 + 6000 updates used by the staged variants.
EOF
}

VARIANT="${1:-}"
if [[ -z "${VARIANT}" ]]; then
  usage
  exit 2
fi
SEED="${2:-0}"
if (( $# >= 2 )); then
  shift 2
else
  shift "$#"
fi
EXTRA_ARGS=("$@")

VLM_PROMPT_TOKENS=16
ACTION_PROMPT_TOKENS=16
CABO_ENABLED=true
PRETRAIN_STEPS=1000
BRIDGE_STEPS=250
FLOW_STEPS_OVERRIDE="${FLOW_STEPS:-}"
FLOW_STEPS_DEFAULT=6000

case "${VARIANT}" in
  full_reference)
    ;;
  no_bridge)
    BRIDGE_STEPS=0
    ;;
  direct_dual)
    PRETRAIN_STEPS=0
    BRIDGE_STEPS=0
    FLOW_STEPS_DEFAULT=7000
    ;;
  dual_prompt_only)
    PRETRAIN_STEPS=0
    BRIDGE_STEPS=0
    FLOW_STEPS_DEFAULT=7000
    CABO_ENABLED=false
    ;;
  vlm_only)
    PRETRAIN_STEPS=0
    BRIDGE_STEPS=0
    FLOW_STEPS_DEFAULT=7000
    ACTION_PROMPT_TOKENS=0
    CABO_ENABLED=false
    ;;
  action_only)
    PRETRAIN_STEPS=0
    BRIDGE_STEPS=0
    FLOW_STEPS_DEFAULT=7000
    VLM_PROMPT_TOKENS=0
    CABO_ENABLED=false
    ;;
  no_cabo)
    CABO_ENABLED=false
    ;;
  *)
    echo "Unknown variant: ${VARIANT}" >&2
    usage >&2
    exit 2
    ;;
esac

FLOW_STEPS="${FLOW_STEPS_OVERRIDE:-${FLOW_STEPS_DEFAULT}}"

DATASET_REPO_ID="${DATASET_REPO_ID:-libero}"
DATASET_ROOT="${DATASET_ROOT:-/data/datasets/libero}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/data/models/lerobot/pi05_libero_base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/wyn/chkpt/2601-lerobot/prompt-ablation}"
LOG_ROOT="${LOG_ROOT:-/data1/wyn/logs/prompt-ablation}"
GPU_IDS="${GPU_IDS:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAVE_FREQ="${SAVE_FREQ:-3000}"
CABO_RATIO="${CABO_RATIO:-2.0}"
MASKED_STEPS="${MASKED_STEPS:-40}"
DTYPE="${DTYPE:-float32}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

RUN_NAME="pi05-${VARIANT}-seed${SEED}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_FILE="${LOG_ROOT}/${RUN_NAME}.log"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${PRETRAINED_PATH}" ]]; then
  echo "Pretrained model does not exist: ${PRETRAINED_PATH}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Output already exists; refusing to overwrite: ${OUTPUT_DIR}" >&2
  exit 1
fi
if ! command -v accelerate >/dev/null 2>&1; then
  echo "accelerate is not available in the active environment" >&2
  exit 1
fi
if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train is not available; install this checkout with: pip install -e ." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
export TMPDIR="${TMPDIR:-/data1/wyn/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/data1/wyn/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data1/wyn/cache/triton}"
export HF_HOME="${HF_HOME:-/data1/wyn/cache/huggingface}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${HF_HOME}"

TRAIN_ARGS=(
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.video_backend="${VIDEO_BACKEND}"
  --policy.type=pi05
  --policy.training_stage=flow
  --policy.pretrained_path="${PRETRAINED_PATH}"
  --policy.device=cuda
  --policy.dtype="${DTYPE}"
  --policy.compile_model="${COMPILE_MODEL}"
  --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}"
  --policy.num_vlm_prompt_tokens="${VLM_PROMPT_TOKENS}"
  --policy.num_prompt_tokens="${ACTION_PROMPT_TOKENS}"
  --policy.next_action_pretrain_steps="${PRETRAIN_STEPS}"
  --policy.next_action_bridge_steps="${BRIDGE_STEPS}"
  --policy.next_action_masked_steps="${MASKED_STEPS}"
  --policy.cabo_enabled="${CABO_ENABLED}"
  --policy.cabo_prompt_update_ratio="${CABO_RATIO}"
  --policy.push_to_hub=false
  --output_dir="${OUTPUT_DIR}"
  --job_name="${RUN_NAME}"
  --steps="${FLOW_STEPS}"
  --batch_size="${BATCH_SIZE}"
  --seed="${SEED}"
  --save_checkpoint=true
  --save_freq="${SAVE_FREQ}"
)

LAUNCH_ARGS=(
  launch
  --num_processes="${NUM_PROCESSES}"
  --mixed_precision="${MIXED_PRECISION}"
)
if (( NUM_PROCESSES > 1 )); then
  LAUNCH_ARGS+=(--multi_gpu)
fi

echo "variant=${VARIANT} seed=${SEED} GPUs=${GPU_IDS} processes=${NUM_PROCESSES}"
echo "prompts=${VLM_PROMPT_TOKENS}+${ACTION_PROMPT_TOKENS} pretrain=${PRETRAIN_STEPS} bridge=${BRIDGE_STEPS} flow=${FLOW_STEPS} CABO=${CABO_ENABLED}"
echo "output=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  accelerate "${LAUNCH_ARGS[@]}" "$(command -v lerobot-train)" \
  "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
