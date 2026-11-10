#!/usr/bin/env bash
set -euo pipefail

# Rebuttal experiments under exactly 4,000 cumulative optimizer updates.
# Prompt controls retain their existing recipe; lora_r16 is a separate,
# prompt-free external PEFT baseline using the same base/data/flow pipeline.
usage() {
  cat <<'EOF'
Usage: bash examples/training/train_pi05_revision_4000.sh VARIANT [SEED] [extra lerobot-train args...]

Variants (each uses exactly 4,000 cumulative optimizer updates):
  direct_dual       0 SAP + 0 bridge + 4000 flow, dual prompts, CPC on
  dual_prompt_only  0 SAP + 0 bridge + 4000 flow, dual prompts, CPC off
  full_reference    750 SAP + 250 bridge + 3000 flow, dual prompts, CPC on
  full_no_cpc       750 SAP + 250 bridge + 3000 flow, dual prompts, CPC off
  action_flow_warmup 750 full-mask action flow + 250 bridge + 3000 flow, CPC on
  fixed_action_lr_050 750 SAP + 250 bridge + 3000 flow, CPC off, action LR x0.50
  fixed_action_lr_025 750 SAP + 250 bridge + 3000 flow, CPC off, action LR x0.25
  fixed_action_lr_010 750 SAP + 250 bridge + 3000 flow, CPC off, action LR x0.10
  lora_r16           4000 flow, standard LoRA r=alpha=16 on VLM+action expert;
                    no prompts, SAP, bridge or CPC (external PEFT baseline)

Fixed action LR applies only to bridge and formal flow; the first 750 SAP
updates and the VLM prompt LR are unchanged. These are fresh training runs,
not continuations of a CPC-trained 001000 transfer checkpoint.

The formal flow checkpoints are 004000 for direct_dual/dual_prompt_only/lora_r16,
and 003000 for full_reference (both represent 4,000 total updates).
The default checkpoint/log roots are isolated under revision-4000.

Defaults for DATASET_ROOT, PRETRAINED_PATH and TOKENIZER_PATH use the
/root/autodl-fs mount. Override shared environment settings as needed.
DRY_RUN=true prints the command;
the source launcher still checks that the three input directories exist.

For lora_r16, optionally set REFERENCE_TRAIN_CONFIG to an existing matched
reference's pretrained_model/train_config.json. The preflight checks original
base (following a staged transfer back to its Stage-1 config), data, effective
batch, dtype and total updates. It does not hash historical weights/data.
LoRA saves compact adapter checkpoints every 1000 updates by default, including
003000/004000; the standard evaluator selects 004000, not the best checkpoint.
EOF
}

VARIANT="${1:-}"
if [[ "$VARIANT" == --help || "$VARIANT" == -h ]]; then
  usage
  exit 0
fi
SEED="${2:-0}"
if [[ ! "$SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SEED must be a non-negative integer (provide SEED before extra args)" >&2
  exit 2
fi
if (( $# >= 2 )); then
  shift 2
else
  shift "$#"
fi

ACTION_PROMPT_LR_SCALE=1.0
case "$VARIANT" in
  fixed_action_lr_050) ACTION_PROMPT_LR_SCALE=0.5 ;;
  fixed_action_lr_025) ACTION_PROMPT_LR_SCALE=0.25 ;;
  fixed_action_lr_010) ACTION_PROMPT_LR_SCALE=0.1 ;;
esac

case "$VARIANT" in
  direct_dual|dual_prompt_only|lora_r16)
    FLOW_UPDATES=4000
    PRETRAIN_UPDATES=0
    BRIDGE_UPDATES=0
    ;;
  full_reference|full_no_cpc|action_flow_warmup|fixed_action_lr_050|fixed_action_lr_025|fixed_action_lr_010)
    FLOW_UPDATES=3000
    PRETRAIN_UPDATES=1000
    BRIDGE_UPDATES=250
    ;;
  *)
    echo "Unknown revision variant: $VARIANT" >&2
    usage >&2
    exit 2
    ;;
esac

# Extra draccus overrides take precedence over earlier flags. Disallow changing
# protocol-defining settings behind the matched-budget launcher's back.
for arg in "$@"; do
  case "$arg" in
    --steps|--steps=*|--seed|--seed=*|--output_dir|--output_dir=*|\
    --job_name|--job_name=*|--resume|--resume=*|\
    --policy.training_stage|--policy.training_stage=*|\
    --policy.next_action_pretrain_steps|--policy.next_action_pretrain_steps=*|\
    --policy.next_action_bridge_steps|--policy.next_action_bridge_steps=*|\
    --policy.next_action_full_mask_probability|--policy.next_action_full_mask_probability=*|\
    --policy.cabo_enabled|--policy.cabo_enabled=*|\
    --policy.action_prompt_lr_scale|--policy.action_prompt_lr_scale=*|\
    --policy.num_prompt_tokens|--policy.num_prompt_tokens=*|\
    --policy.num_vlm_prompt_tokens|--policy.num_vlm_prompt_tokens=*)
      echo "Protocol-defining CLI override is not allowed: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ "$VARIANT" == lora_r16 ]]; then
  for arg in "$@"; do
    case "$arg" in
      # LR tuning is allowed explicitly, but must use a separate OUTPUT_ROOT
      # and the same validation-only selection rule for every compared method.
      --policy.optimizer_lr|--policy.optimizer_lr=*|--policy.scheduler_decay_lr|--policy.scheduler_decay_lr=*) ;;
      --config_path|--config_path=*|--policy|--policy=*|--policy.*|\
      --peft|--peft=*|--peft.*|--dataset|--dataset=*|--dataset.*|\
      --batch_size|--batch_size=*|--use_policy_training_preset|--use_policy_training_preset=*|\
      --optimizer|--optimizer=*|--optimizer.*|--scheduler|--scheduler=*|--scheduler.*|\
      --sample_weighting|--sample_weighting=*|--sample_weighting.*|--rename_map|--rename_map=*)
        echo "Protocol-defining LoRA CLI override is not allowed: $arg (use the shared environment settings)" >&2
        exit 2
        ;;
    esac
  done
fi

TOTAL_UPDATES=$((PRETRAIN_UPDATES + FLOW_UPDATES))
if (( TOTAL_UPDATES != 4000 )); then
  echo "Internal error: expected 4000 total updates, got $TOTAL_UPDATES" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FLOW_STEPS="$FLOW_UPDATES"
export RUN_NAME="pi05-revision-${VARIANT}-seed${SEED}-total4000"
export DATASET_ROOT="${DATASET_ROOT:-/root/autodl-fs/datasets/libero}"
export PRETRAINED_PATH="${PRETRAINED_PATH:-/root/autodl-fs/ckpt/pi05_libero_base}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/root/autodl-fs/ckpt/google/paligemma-3b-pt-224}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/wyn/chkpt/2601-lerobot/revision-4000}"
export LOG_ROOT="${LOG_ROOT:-/root/autodl-tmp/wyn/logs/revision-4000}"
# Save only the final flow checkpoint by default. Integrated PI0.5 Stage 1
# saves its required transfer checkpoint independently at step 1000, and its
# formal flow stage saves the final checkpoint regardless of save frequency.
# Existing runs remain intact; explicitly set SAVE_FREQ=1000 when intermediate
# checkpoints are needed for an analysis.
if [[ "$VARIANT" == lora_r16 ]]; then
  export SAVE_FREQ="${SAVE_FREQ:-1000}"
else
  export SAVE_FREQ="${SAVE_FREQ:-$FLOW_UPDATES}"
fi
# Stage snapshots are diagnostic only; disable their extra I/O for all groups.
export NTK_SAVE_STAGE_SNAPSHOTS=false

printf 'revision-4000: variant=%s seed=%s SAP=%d bridge=%d flow=%d total=%d CPC=%s\n' \
  "$VARIANT" "$SEED" "$((PRETRAIN_UPDATES - BRIDGE_UPDATES))" "$BRIDGE_UPDATES" \
  "$FLOW_UPDATES" "$TOTAL_UPDATES" \
  "$(if [[ "$VARIANT" == dual_prompt_only || "$VARIANT" == full_no_cpc || "$VARIANT" == fixed_action_lr_* || "$VARIANT" == lora_r16 ]]; then echo false; else echo true; fi)"
printf 'revision-4000: action_prompt_lr_scale=%s (bridge/flow only; SAP unchanged)\n' "$ACTION_PROMPT_LR_SCALE"

if [[ "$VARIANT" == action_flow_warmup ]]; then
  echo "revision-4000: action-only Stage 1 uses full-mask flow matching (no visible action context), not SAP inpainting"
fi

if [[ "$VARIANT" == lora_r16 ]]; then
  echo "revision-4000: external standard LoRA r=16 alpha=16, both transformers; frozen base, no trainable prompts"
  echo "revision-4000: match ORIGINAL initialization, not a primed transfer checkpoint: $PRETRAINED_PATH"
  if [[ -n "${REFERENCE_TRAIN_CONFIG:-}" ]]; then
    "${PYTHON_BIN:-python}" "$SCRIPT_DIR/verify_pi05_revision_reference.py" \
      --reference "$REFERENCE_TRAIN_CONFIG" \
      --pretrained "$PRETRAINED_PATH" --dataset "$DATASET_ROOT" \
      --repo-id "${DATASET_REPO_ID:-libero}" --tokenizer "$TOKENIZER_PATH" \
      --batch-size "${BATCH_SIZE:-32}" --num-processes "${NUM_PROCESSES:-1}" \
      --reference-num-processes "${REFERENCE_NUM_PROCESSES:-0}" \
      --dtype "${DTYPE:-float32}" --video-backend "${VIDEO_BACKEND:-pyav}" --total-updates 4000
  else
    echo "revision-4000: NOTE: no REFERENCE_TRAIN_CONFIG supplied; historical run parity is not verified."
  fi
fi

SOURCE_VARIANT="$VARIANT"
if [[ "$VARIANT" == full_no_cpc || "$VARIANT" == fixed_action_lr_* ]]; then
  SOURCE_VARIANT=no_cabo
fi
exec bash "$SCRIPT_DIR/train_pi05_prompt_ablation.sh" "$SOURCE_VARIANT" "$SEED" \
  --policy.action_prompt_lr_scale="$ACTION_PROMPT_LR_SCALE" \
  --preserve_pi05_flow_save_freq=true "$@"
