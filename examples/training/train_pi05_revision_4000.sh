#!/usr/bin/env bash
set -euo pipefail

# Rebuttal experiment: compare dual-prompt adaptation under exactly 4,000
# cumulative optimizer updates. Keep the original prompt-ablation launcher and
# its optimizer/model/data settings unchanged.
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
  optim_prefix        750 SAP + 250 bridge ONLY (1000 updates); save shared Stage-II checkpoint
  optim_carry_carry   750 SAP + 250 bridge + 3000 flow, CPC on, carry AdamW moments and scheduler
  optim_reset_carry   750 SAP + 250 bridge + 3000 flow, CPC on, reset AdamW moments, carry scheduler
  optim_carry_restart 750 SAP + 250 bridge + 3000 flow, CPC on, carry AdamW moments, restart scheduler
  optim_reset_restart 750 SAP + 250 bridge + 3000 flow, CPC on, reset both (shared-prefix control)

Fixed action LR applies only to bridge and formal flow; the first 750 SAP
updates and the VLM prompt LR are unchanged. These are fresh training runs,
not continuations of a CPC-trained 001000 transfer checkpoint.

Run optim_prefix once, then give each optimizer-transition cell the same
PREFIX_CHECKPOINT_DIR from that exact 1000-step checkpoint. Each runs only
the remaining 3000 adaptation updates, with NUM_PROCESSES=1 on one GPU.
Scheduler carry means remaining at the Stage-II schedule's terminal LR,
not reinterpreting step 1000 in a fresh 3000-step cosine schedule.

The formal flow checkpoints are 004000 for direct_dual/dual_prompt_only,
and 003000 for full_reference (both represent 4,000 total updates).
The default checkpoint/log roots are isolated under revision-4000.

Defaults for DATASET_ROOT, PRETRAINED_PATH and TOKENIZER_PATH use the
/root/autodl-fs mount. Override shared environment settings as needed.
DRY_RUN=true prints the command;
the source launcher still checks that the three input directories exist.
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
STAGE3_ADAMW_MOMENTS=reset
STAGE3_LR_SCHEDULER=restart
case "$VARIANT" in
  fixed_action_lr_050) ACTION_PROMPT_LR_SCALE=0.5 ;;
  fixed_action_lr_025) ACTION_PROMPT_LR_SCALE=0.25 ;;
  fixed_action_lr_010) ACTION_PROMPT_LR_SCALE=0.1 ;;
  optim_carry_carry) STAGE3_ADAMW_MOMENTS=carry; STAGE3_LR_SCHEDULER=carry ;;
  optim_reset_carry) STAGE3_ADAMW_MOMENTS=reset; STAGE3_LR_SCHEDULER=carry ;;
  optim_carry_restart) STAGE3_ADAMW_MOMENTS=carry; STAGE3_LR_SCHEDULER=restart ;;
esac

case "$VARIANT" in
  direct_dual|dual_prompt_only)
    FLOW_UPDATES=4000
    PRETRAIN_UPDATES=0
    BRIDGE_UPDATES=0
    ;;
  full_reference|full_no_cpc|action_flow_warmup|fixed_action_lr_050|fixed_action_lr_025|fixed_action_lr_010|optim_prefix|optim_carry_carry|optim_reset_carry|optim_carry_restart|optim_reset_restart)
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
    --policy.stage3_adamw_moments|--policy.stage3_adamw_moments=*|\
    --policy.stage3_lr_scheduler|--policy.stage3_lr_scheduler=*|\
    --pi05_stage1_only|--pi05_stage1_only=*|\
    --pi05_stage3_transfer_checkpoint|--pi05_stage3_transfer_checkpoint=*|\
    --policy.num_prompt_tokens|--policy.num_prompt_tokens=*|\
    --policy.num_vlm_prompt_tokens|--policy.num_vlm_prompt_tokens=*)
      echo "Protocol-defining CLI override is not allowed: $arg" >&2
      exit 2
      ;;
  esac
done

TOTAL_UPDATES=$((PRETRAIN_UPDATES + FLOW_UPDATES))
if (( TOTAL_UPDATES != 4000 )); then
  echo "Internal error: expected 4000 total updates, got $TOTAL_UPDATES" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FLOW_STEPS="$FLOW_UPDATES"
if [[ "$VARIANT" == optim_prefix ]]; then
  export RUN_NAME="pi05-revision-${VARIANT}-seed${SEED}-prefix1000"
else
  export RUN_NAME="pi05-revision-${VARIANT}-seed${SEED}-total4000"
fi
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
export SAVE_FREQ="${SAVE_FREQ:-$FLOW_UPDATES}"
# Stage snapshots are diagnostic only; disable their extra I/O for all groups.
export NTK_SAVE_STAGE_SNAPSHOTS=false

printf 'revision-4000: variant=%s seed=%s SAP=%d bridge=%d flow=%d total=%d CPC=%s\n' \
  "$VARIANT" "$SEED" "$((PRETRAIN_UPDATES - BRIDGE_UPDATES))" "$BRIDGE_UPDATES" \
  "$FLOW_UPDATES" "$TOTAL_UPDATES" \
  "$(if [[ "$VARIANT" == dual_prompt_only || "$VARIANT" == full_no_cpc || "$VARIANT" == fixed_action_lr_* ]]; then echo false; else echo true; fi)"
printf 'revision-4000: action_prompt_lr_scale=%s (bridge/flow only; SAP unchanged)\n' "$ACTION_PROMPT_LR_SCALE"
printf 'revision-4000: Stage-III transition AdamW=%s scheduler=%s\n' "$STAGE3_ADAMW_MOMENTS" "$STAGE3_LR_SCHEDULER"

TRANSITION_ARGS=()
if [[ "$VARIANT" == optim_prefix ]]; then
  TRANSITION_ARGS+=(--pi05_stage1_only=true)
  printf 'revision-4000: shared prefix checkpoint: %s/%s_next_action_pretrain/checkpoints/001000\n' "$OUTPUT_ROOT" "$RUN_NAME"
elif [[ "$VARIANT" == optim_* ]]; then
  if [[ -z "${PREFIX_CHECKPOINT_DIR:-}" ]]; then
    echo "Optimizer ablations require PREFIX_CHECKPOINT_DIR from a completed optim_prefix run" >&2
    exit 2
  fi
  if [[ ! -d "$PREFIX_CHECKPOINT_DIR/pretrained_model" || ! -d "$PREFIX_CHECKPOINT_DIR/training_state" ]]; then
    echo "Incomplete shared Stage-II checkpoint: $PREFIX_CHECKPOINT_DIR" >&2
    exit 2
  fi
  for required_file in \
    pretrained_model/train_config.json \
    pretrained_model/model.safetensors \
    training_state/optimizer_state.safetensors \
    training_state/optimizer_param_groups.json \
    training_state/scheduler_state.json; do
    if [[ ! -f "$PREFIX_CHECKPOINT_DIR/$required_file" ]]; then
      echo "Missing Stage-II state file: $PREFIX_CHECKPOINT_DIR/$required_file" >&2
      exit 2
    fi
  done
  export PRETRAINED_PATH="$PREFIX_CHECKPOINT_DIR/pretrained_model"
  TRANSITION_ARGS+=(--pi05_stage3_transfer_checkpoint="$PREFIX_CHECKPOINT_DIR")
  printf 'revision-4000: shared prefix source=%s; training only the 3000-update formal flow\n' "$PREFIX_CHECKPOINT_DIR"
fi

if [[ "$VARIANT" == optim_* && "${NUM_PROCESSES:-1}" != 1 ]]; then
  echo "Optimizer-transition P1 requires NUM_PROCESSES=1 (one independent run per GPU)" >&2
  exit 2
fi

if [[ "$VARIANT" == action_flow_warmup ]]; then
  echo "revision-4000: action-only Stage 1 uses full-mask flow matching (no visible action context), not SAP inpainting"
fi

SOURCE_VARIANT="$VARIANT"
if [[ "$VARIANT" == full_no_cpc || "$VARIANT" == fixed_action_lr_* ]]; then
  SOURCE_VARIANT=no_cabo
elif [[ "$VARIANT" == optim_* ]]; then
  SOURCE_VARIANT=full_reference
fi
exec bash "$SCRIPT_DIR/train_pi05_prompt_ablation.sh" "$SOURCE_VARIANT" "$SEED" \
  --policy.action_prompt_lr_scale="$ACTION_PROMPT_LR_SCALE" \
  --policy.stage3_adamw_moments="$STAGE3_ADAMW_MOMENTS" \
  --policy.stage3_lr_scheduler="$STAGE3_LR_SCHEDULER" \
  "${TRANSITION_ARGS[@]}" \
  --preserve_pi05_flow_save_freq=true "$@"
