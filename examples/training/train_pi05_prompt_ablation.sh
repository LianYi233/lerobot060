#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash train_pi05_prompt_ablation.sh VARIANT [SEED] [extra lerobot-train args...]

Variants:
  full_reference    750 inpainting + 250 bridge + 3000 flow, dual prompts, CABO
  no_sap_matched    750 joint flow (CABO off) + 250 bridge + 3000 flow (CABO on)
  no_bridge         1000 inpainting + 0 bridge + 3000 flow, dual prompts, CABO
  direct_dual       0 pretraining + 3000 flow, dual prompts, CABO
  dual_prompt_only  0 pretraining + 3000 flow, dual prompts, CABO off
  vlm_only          0 pretraining + 3000 flow, VLM prompt only, CABO off
  action_only       0 pretraining + 3000 flow, action prompt only, CABO off
  no_cabo           750 inpainting + 250 bridge + 3000 flow, dual prompts, CABO off

All variants use 3000 flow updates by default. Set FLOW_STEPS to override this
for a one-off run, except for no_sap_matched, whose schedule is fixed.
no_sap_matched starts from the reference's saved step-0 weights, keeps its 1000-step
first-phase LR schedule and collects the four NTK checkpoints. Set STAGED_RUN to
the reference run directory, or PRETRAINED_PATH to its exact step-0 pretrained_model.
full_reference automatically saves exact initialization and priming-end snapshots,
and saves flow checkpoints every 1000 updates (including flow 2000 = total 3000).
Set NTK_SAVE_STAGE_SNAPSHOTS=false to disable extra stage snapshots, or SAVE_FREQ
to override the flow checkpoint interval. Other variants keep their existing defaults.
Set FLOW_STEPS=4000 SAVE_STEPS='[0,750,1000,4000]' for exact flow-update checkpoints;
0 saves the actual initialized model and processors before its first update.
Set RUN_NAME to isolate tasks, or DRY_RUN=true to print the command without training.
EOF
}

VARIANT="${1:-}"
if [[ "${VARIANT}" == --help || "${VARIANT}" == -h ]]; then
  usage
  exit 0
fi
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
FLOW_STEPS_DEFAULT=3000
SAVE_FREQ_DEFAULT=3000
NTK_SNAPSHOTS_DEFAULT=false

case "${VARIANT}" in
  full_reference)
    SAVE_FREQ_DEFAULT=1000
    NTK_SNAPSHOTS_DEFAULT=true
    ;;
  no_sap_matched)
    NTK_SNAPSHOTS_DEFAULT=true
    if [[ "${FLOW_STEPS_OVERRIDE:-3000}" != 3000 || "${NTK_SAVE_STAGE_SNAPSHOTS:-true}" != true ]]; then
      echo "no_sap_matched requires FLOW_STEPS=3000 and NTK_SAVE_STAGE_SNAPSHOTS=true." >&2
      exit 2
    fi
    # Do not silently allow free-form CLI flags to replace the reference protocol.
    for arg in "${EXTRA_ARGS[@]}"; do
      case "${arg}" in
        --wandb.*=*|--log_freq=*) ;;
        *) echo "Unsupported no_sap_matched override: ${arg}. Use environment variables for paths and hardware." >&2; exit 2 ;;
      esac
    done
    ;;
  no_bridge)
    BRIDGE_STEPS=0
    ;;
  direct_dual)
    PRETRAIN_STEPS=0
    BRIDGE_STEPS=0
    ;;
  dual_prompt_only)
    PRETRAIN_STEPS=0
    BRIDGE_STEPS=0
    CABO_ENABLED=false
    ;;
  vlm_only)
    PRETRAIN_STEPS=0
    BRIDGE_STEPS=0
    ACTION_PROMPT_TOKENS=0
    CABO_ENABLED=false
    ;;
  action_only)
    PRETRAIN_STEPS=0
    BRIDGE_STEPS=0
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
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/datasets/libero}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/root/autodl-tmp/models/google/paligemma-3b-pt-224}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/chkpt/2601-lerobot/prompt-ablation}"
LOG_ROOT="${LOG_ROOT:-/root/autodl-tmp/logs/prompt-ablation}"
GPU_IDS="${GPU_IDS:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAVE_FREQ="${SAVE_FREQ:-${SAVE_FREQ_DEFAULT}}"
SAVE_STEPS="${SAVE_STEPS:-[]}"
NTK_SAVE_STAGE_SNAPSHOTS="${NTK_SAVE_STAGE_SNAPSHOTS:-${NTK_SNAPSHOTS_DEFAULT}}"
CABO_RATIO="${CABO_RATIO:-2.0}"
MASKED_STEPS="${MASKED_STEPS:-40}"
DTYPE="${DTYPE:-float32}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

if [[ "${VARIANT}" == no_sap_matched ]]; then
  STAGED_RUN="${STAGED_RUN:-${OUTPUT_ROOT}/pi05-full-reference-ntk-seed${SEED}}"
  PRETRAINED_PATH="${PRETRAINED_PATH:-${STAGED_RUN}_next_action_pretrain/checkpoints/000000/pretrained_model}"
  RUN_NAME="${RUN_NAME:-pi05-no-sap-matched-ntk-seed${SEED}}"
  if [[ ! -s "${PRETRAINED_PATH}/model.safetensors" || ! -s "${PRETRAINED_PATH}/config.json" ]]; then
    echo "Missing reference step-0 weights or config: ${PRETRAINED_PATH}" >&2
    exit 1
  fi
  # Reject an accidentally inherited path to the base model or a later checkpoint.
  python - "${PRETRAINED_PATH}" "${SEED}" "${NUM_PROCESSES}" "${BATCH_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

model = Path(sys.argv[1])
try:
    state = json.loads((model.parent / "training_state/training_step.json").read_text())
    cfg = json.loads((model / "train_config.json").read_text())
except (OSError, ValueError) as exc:
    raise SystemExit(f"Cannot verify the reference step-0 checkpoint: {exc}")
expected = (0, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
actual = (state.get("step"), cfg.get("seed"), state.get("num_processes"), cfg.get("batch_size"))
if actual != expected:
    raise SystemExit(f"Reference (step, seed, num_processes, batch_size): {actual}; expected {expected}")
PY
else
  PRETRAINED_PATH="${PRETRAINED_PATH:-/root/autodl-tmp/models/pi05_libero_base}"
  RUN_NAME="${RUN_NAME:-pi05-${VARIANT}-seed${SEED}}"
fi
DRY_RUN="${DRY_RUN:-false}"
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
if [[ ! -d "${TOKENIZER_PATH}" ]]; then
  echo "Tokenizer does not exist: ${TOKENIZER_PATH}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Output already exists; refusing to overwrite: ${OUTPUT_DIR}" >&2
  exit 1
fi
if (( PRETRAIN_STEPS > 0 )) && [[ -e "${OUTPUT_DIR}_next_action_pretrain" ]]; then
  echo "Pretraining output already exists; choose a new RUN_NAME: ${OUTPUT_DIR}_next_action_pretrain" >&2
  exit 1
fi
if [[ "${DRY_RUN}" != true ]] && ! command -v accelerate >/dev/null 2>&1; then
  echo "accelerate is not available in the active environment" >&2
  exit 1
fi
if [[ "${DRY_RUN}" != true ]] && ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train is not available; install this checkout with: pip install -e ." >&2
  exit 1
fi

TRAIN_ARGS=(
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.video_backend="${VIDEO_BACKEND}"
  --policy.type=pi05
  --policy.training_stage=flow
  --policy.pretrained_path="${PRETRAINED_PATH}"
  --policy.tokenizer_name="${TOKENIZER_PATH}"
  --policy.device=cuda
  --policy.dtype="${DTYPE}"
  --policy.compile_model="${COMPILE_MODEL}"
  --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}"
  --policy.num_vlm_prompt_tokens="${VLM_PROMPT_TOKENS}"
  --policy.num_prompt_tokens="${ACTION_PROMPT_TOKENS}"
  --policy.next_action_pretrain_steps="${PRETRAIN_STEPS}"
  --policy.next_action_bridge_steps="${BRIDGE_STEPS}"
  --policy.next_action_masked_steps="${MASKED_STEPS}"
  --policy.ntk_save_stage_snapshots="${NTK_SAVE_STAGE_SNAPSHOTS}"
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
  --save_steps="${SAVE_STEPS}"
)
if [[ "${VARIANT}" == no_sap_matched ]]; then
  TRAIN_ARGS+=(--policy.next_action_joint_flow_before_bridge=true)
fi

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
echo "NTK stage snapshots=${NTK_SAVE_STAGE_SNAPSHOTS} flow checkpoint interval=${SAVE_FREQ}"
echo "Additional exact flow checkpoints=${SAVE_STEPS}"
if [[ "${NTK_SAVE_STAGE_SNAPSHOTS}" == true ]] && (( PRETRAIN_STEPS > 0 )); then
  PRETRAIN_OUTPUT_DIR="${OUTPUT_DIR}_next_action_pretrain"
  printf 'NTK before training: %s/checkpoints/%06d/pretrained_model\n' "${PRETRAIN_OUTPUT_DIR}" 0
  if (( BRIDGE_STEPS > 0 )); then
    BOUNDARY_LABEL="after priming"
    if [[ "${VARIANT}" == no_sap_matched ]]; then BOUNDARY_LABEL="before bridge"; fi
    printf 'NTK %s:  %s/checkpoints/%06d/pretrained_model\n' "${BOUNDARY_LABEL}" \
      "${PRETRAIN_OUTPUT_DIR}" "$((PRETRAIN_STEPS - BRIDGE_STEPS))"
  fi
  printf 'NTK after stage 2:  %s/checkpoints/%06d/pretrained_model\n' "${PRETRAIN_OUTPUT_DIR}" "${PRETRAIN_STEPS}"
  printf 'NTK final flow:     %s/checkpoints/%06d/pretrained_model (cumulative updates: %d)\n' \
    "${OUTPUT_DIR}" "${FLOW_STEPS}" "$((PRETRAIN_STEPS + FLOW_STEPS))"
fi

if [[ "${DRY_RUN}" == true ]]; then
  printf '%q ' env "CUDA_VISIBLE_DEVICES=${GPU_IDS}" accelerate "${LAUNCH_ARGS[@]}" \
    lerobot-train "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/root/autodl-tmp/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/autodl-tmp/cache/triton}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${HF_HOME}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  accelerate "${LAUNCH_ARGS[@]}" "$(command -v lerobot-train)" \
  "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
