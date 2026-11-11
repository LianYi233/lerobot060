# P1: AdamW state x scheduler state (2x2)

This is a controlled PrimingVLA reviewer ablation, not an extra training stage.
The two compared stages use AdamW with the same optimizer hyperparameters;
the intervention is whether its accumulated moments and the LR scheduler's
position survive the Stage-II -> Stage-III boundary.

## Design

One shared seed-3 prefix runs **750 SAP + 250 bridge** updates (the same original
model initialization, dataset, prompts, and CPC settings as `full_reference`).
It saves the Stage-II `001000` checkpoint with model, AdamW, and scheduler state.
Four independent Flow Adaptation runs each load **the same** weights and then
perform 3,000 updates, for 4,000 total updates per cell:

| Variant | AdamW moments | LR scheduler |
| --- | --- | --- |
| `optim_carry_carry` | carry | carry |
| `optim_reset_carry` | reset | carry |
| `optim_carry_restart` | carry | restart |
| `optim_reset_restart` | reset | restart (matched control) |

The Stage-II scheduler's cosine curve completes its 1,000-update horizon at
`scheduler_decay_lr=1e-5`; **carry** means keeping that terminal LR during
the next 3,000 updates. **Restart** builds the Stage-III 3,000-update cosine
curve, with its own scaled warm-up. This is a deliberately literal carry/reset
comparison and should be defined in the paper. Only one process per GPU is
supported. Do not reuse a historical `full_reference` checkpoint as the matched
control unless its exact Stage-II state, source seed and protocol are verified.

## Before launching on AutoDL server A

Run these commands from the repo **after installing/pulling a revision that
actually contains the `optim_*` launcher variants and the state-transfer code**.
The source in this working tree is not automatically present on server A.
Using `PEFT-baseline` or `revision` without these changes will not implement P1.

```bash
cd /root/lerobot060
conda activate lerobot

# Confirm that this is the P1-capable code, before spending GPU time.
rg -n 'optim_prefix|optim_carry_carry' examples/training/train_pi05_revision_4000.sh
rg -n 'PI05_OPTIMIZER_TRANSITION' src/lerobot/scripts/lerobot_train.py

export P1_SEED=3
export P1_CKPT_ROOT=/root/autodl-tmp/wyn/chkpt/2601-lerobot/optimizer-transition-p1
export P1_LOG_ROOT=/root/autodl-tmp/wyn/logs/optimizer-transition-p1
export DATASET_ROOT=/root/autodl-fs/datasets/libero
export PRETRAINED_PATH=/root/autodl-fs/ckpt/pi05_libero_base
export TOKENIZER_PATH=/root/autodl-fs/ckpt/google/paligemma-3b-pt-224
export NUM_PROCESSES=1
export BATCH_SIZE=32

test -d "$DATASET_ROOT" && test -d "$PRETRAINED_PATH" && test -d "$TOKENIZER_PATH"
```

**Shared prefix on GPU0 (only once):**

```bash
GPU_IDS=0 OUTPUT_ROOT="$P1_CKPT_ROOT" LOG_ROOT="$P1_LOG_ROOT" \
  bash examples/training/train_pi05_revision_4000.sh optim_prefix "$P1_SEED"

export PREFIX_CHECKPOINT_DIR="$P1_CKPT_ROOT/pi05-revision-optim_prefix-seed${P1_SEED}-prefix1000_next_action_pretrain/checkpoints/001000"
test -d "$PREFIX_CHECKPOINT_DIR/pretrained_model"
test -f "$PREFIX_CHECKPOINT_DIR/pretrained_model/train_config.json"
test -f "$PREFIX_CHECKPOINT_DIR/pretrained_model/model.safetensors"
test -f "$PREFIX_CHECKPOINT_DIR/training_state/optimizer_state.safetensors"
test -f "$PREFIX_CHECKPOINT_DIR/training_state/optimizer_param_groups.json"
test -f "$PREFIX_CHECKPOINT_DIR/training_state/scheduler_state.json"
```

**Three independent runs, on GPU0/1/2.** Run each command in its own terminal or
`tmux` window, with the exported environment above available in each window.
Optional `DRY_RUN=true` can be prepended for a zero-GPU path check.

```bash
# Terminal 1 / GPU0
GPU_IDS=0 PREFIX_CHECKPOINT_DIR="$PREFIX_CHECKPOINT_DIR" \
  OUTPUT_ROOT="$P1_CKPT_ROOT" LOG_ROOT="$P1_LOG_ROOT" \
  bash examples/training/train_pi05_revision_4000.sh optim_carry_carry "$P1_SEED"

# Terminal 2 / GPU1
GPU_IDS=1 PREFIX_CHECKPOINT_DIR="$PREFIX_CHECKPOINT_DIR" \
  OUTPUT_ROOT="$P1_CKPT_ROOT" LOG_ROOT="$P1_LOG_ROOT" \
  bash examples/training/train_pi05_revision_4000.sh optim_reset_carry "$P1_SEED"

# Terminal 3 / GPU2
GPU_IDS=2 PREFIX_CHECKPOINT_DIR="$PREFIX_CHECKPOINT_DIR" \
  OUTPUT_ROOT="$P1_CKPT_ROOT" LOG_ROOT="$P1_LOG_ROOT" \
  bash examples/training/train_pi05_revision_4000.sh optim_carry_restart "$P1_SEED"
```

**Fourth run, after any GPU becomes free (e.g., GPU0):**

```bash
GPU_IDS=0 PREFIX_CHECKPOINT_DIR="$PREFIX_CHECKPOINT_DIR" \
  OUTPUT_ROOT="$P1_CKPT_ROOT" LOG_ROOT="$P1_LOG_ROOT" \
  bash examples/training/train_pi05_revision_4000.sh optim_reset_restart "$P1_SEED"
```

Each Stage-III run writes its final `checkpoints/003000/pretrained_model`.
Check each log for `PI05_OPTIMIZER_TRANSITION moments=... scheduler=...`, except
the reset/restart cell (which deliberately uses the default fresh optimizer and
scheduler). Never infer successful carry-over from the run name alone.

## Evaluation (once GPU3 is available)

```bash
export CKPT_ROOT="$P1_CKPT_ROOT"
export BASE_OUTPUT=/root/autodl-tmp/wyn/eval/2601-lerobot/optimizer-transition-p1
export GPU_ID=3
export EPISODES_PER_TASK=10

bash run_eval_pi05_revision_4000.sh optim_carry_carry all "$P1_SEED"
bash run_eval_pi05_revision_4000.sh optim_reset_carry all "$P1_SEED"
bash run_eval_pi05_revision_4000.sh optim_carry_restart all "$P1_SEED"
bash run_eval_pi05_revision_4000.sh optim_reset_restart all "$P1_SEED"
```

Use the same evaluation seed and episodes per task for all four cells. Report
Spatial, Object, Goal, Long, Overall and (at least) one second independent
training seed if resources permit. A test-time seed is not a substitute for a
newly trained seed. If other methods were trained from `pi05_base`, do **not**
compare this `pi05_libero_base` 2x2 as an equal-initialization experiment.
