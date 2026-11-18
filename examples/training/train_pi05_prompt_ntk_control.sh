#!/usr/bin/env bash
# Matched-budget, uninterrupted prompt-learning control for the four-checkpoint NTK analysis.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash train_pi05_prompt_ntk_control.sh [dual_prompt_only|direct_dual] [SEED] [extra lerobot-train args...]

dual_prompt_only (default): train both prompt banks together, CABO/CPC off.
direct_dual:                 same uninterrupted run with CABO/CPC on.

Both controls use frozen backbones, 16+16 prompt tokens, no SAP or Bridge,
and one continuous 4000-update flow optimizer/scheduler. Exact checkpoints,
including initialized prompt weights and processors, are saved at 0/750/1000/4000
under RUN_DIR/checkpoints/{000000,000750,001000,004000}/pretrained_model.

Set DATASET_ROOT, PRETRAINED_PATH, TOKENIZER_PATH, BATCH_SIZE, GPU_IDS and other
environment options supported by train_pi05_prompt_ablation.sh as needed.
Use the same base model, data, seed, batch size and optimizer settings as your
full_reference run. For identical initial prompt weights, PRETRAINED_PATH may
point at its saved *_next_action_pretrain/checkpoints/000000/pretrained_model.
RUN_NAME defaults to pi05-VARIANT-ntk-control-seedSEED; DRY_RUN=true prints commands.

This wrapper fixes the update budget, snapshot schedule, prompt counts and stage
settings. Conflicting extra CLI arguments are rejected, and inherited FLOW_STEPS,
SAVE_FREQ, SAVE_STEPS and NTK_SAVE_STAGE_SNAPSHOTS are replaced with this protocol.
For resumption, invoke lerobot-train --config_path=<saved train_config.json>
--resume=true directly; the saved configuration retains the snapshot schedule.
EOF
}

VARIANT="${1:-dual_prompt_only}"
if [[ "${VARIANT}" == --help || "${VARIANT}" == -h ]]; then
  usage
  exit 0
fi
case "${VARIANT}" in
  dual_prompt_only|direct_dual) ;;
  *) echo "NTK control must be dual_prompt_only or direct_dual, got: ${VARIANT}" >&2; exit 2 ;;
esac
SEED="${2:-0}"
if (( $# >= 2 )); then
  shift 2
else
  shift "$#"
fi

# Reject both --name=value and --name value forms before the base launcher can append them.
# A loaded policy/config or resume could silently replace the controlled training recipe.
# Also reject prefixes: argument parsers may accept unambiguous option abbreviations.
LOCKED_OPTIONS=(
  --config_path --resume --steps --save_checkpoint --save_freq --save_steps --seed
  --policy --policy.path --policy.type --policy.training_stage
  --policy.next_action_pretrain_steps --policy.next_action_bridge_steps
  --policy.num_vlm_prompt_tokens --policy.num_prompt_tokens --policy.ntk_save_stage_snapshots
  --policy.cabo_enabled --policy.use_peft --peft --reward_model
)
for argument in "$@"; do
  key="${argument%%=*}"
  if [[ "${key}" == --* ]]; then
    for locked_option in "${LOCKED_OPTIONS[@]}"; do
      if [[ "${locked_option}" == "${key}"* ]]; then
        echo "${key} is fixed by the NTK control protocol; use the standard launcher for custom experiments." >&2
        exit 2
      fi
    done
  fi
  case "${key}" in
    --policy.next_action_*|--peft.*|--reward_model.*)
      echo "${key} is fixed by the NTK control protocol; use the standard launcher for custom experiments." >&2
      exit 2
      ;;
  esac
done

export FLOW_STEPS=4000
export SAVE_FREQ=4000
export SAVE_STEPS='[0,750,1000,4000]'
export NTK_SAVE_STAGE_SNAPSHOTS=false
export RUN_NAME="${RUN_NAME:-pi05-${VARIANT}-ntk-control-seed${SEED}}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_pi05_prompt_ablation.sh" "${VARIANT}" "${SEED}" "$@"
