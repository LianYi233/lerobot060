# Piper task 2: controlled partial fine-tuning on three GPUs

The branch `piper-partial-ft` adds *opt-in* trainable VLM upper language layers and
separate AdamW groups, while leaving `piper` and original Prompt-only recipes untouched.

## Experimental objective

Use the **same completed task2 Prompt-only 50k checkpoint** as initialization for
three independent new models (no optimizer-state resume, no repeated SAP/Bridge):

| GPU | Profile | Action trainables | VLM trainables | Peak LR |
|---|---|---|---|---|
| 1 | action4 | input/output projections + last 4 expert blocks | none | prompts 2.5e-5; action 1e-5 |
| 2 | dual2 | same as action4 | last 2 language-model layers | VLM 1e-6 |
| 3 | dual4 | same as action4 | last 4 language-model layers | VLM 1e-6 |

Vision tower and text embeddings remain frozen. Both learned prompt banks stay trainable.
`partial_ft_enabled=True` additionally disables the fast frozen-observation KV path
whenever VLM language layers train, so the selected layers actually receive gradients
through the **recorded observations**, not only through prompt tokens.

These runs **start from the trained 50k weights**. An additional 10k is **not**
equivalent to 10k from `pi05_base`. For publication, add a same-initialization,
same-extra-budget Prompt-only continuation control to separate capacity from extra steps.
Flow velocity MSE (~0.11 before) is not a joint-angle MAE.

## Local machine prerequisites

In the `piper-partial-ft` checkout, use the same Python/LeRobot environment as training.
Before training, materialize or copy the entire original task2 50k policy directory
(`model.safetensors`, `config.json`, `policy_preprocessor.json`,
`policy_postprocessor.json` plus processor state).
The source MUST contain learned VLM/action prompt weights; the new launcher verifies
both and fails on an original generic base model.

```bash
cd /home/wyn/VLAA/lerobot060      # replace if your local checkout is elsewhere
git fetch origin piper-partial-ft
git merge --ff-only origin/piper-partial-ft
git branch --show-current

export WORK_ROOT=/data1/wyn/piper-work
export DATASET_BASE=/path/to/local/reality-dataset-parent
export PRETRAINED_PATH=/path/to/materialized/task2-priming-seed0-total50000
export TOKENIZER_PATH=/path/to/google/paligemma-3b-pt-224

test -f "$PRETRAINED_PATH/model.safetensors"
test -f "$DATASET_BASE/2-remove_the_cuboid_from_blue_plate/meta/info.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
df -h "$WORK_ROOT"
```

### AutoDL A800 instead of the /data1 local server

If the 50k checkpoint is already on AutoDL, do **not** transfer it to the local server.
Use a separate worktree and restore the final model from its verified compact bundle:

```bash
cd /root/autodl-tmp/lerobot060-piper-reality
git fetch origin piper-partial-ft
git worktree add --detach /root/autodl-tmp/lerobot060-piper-partial-ft origin/piper-partial-ft
cd /root/autodl-tmp/lerobot060-piper-partial-ft

export WORK_ROOT=/root/autodl-tmp/piper-partial-ft-work
export DATASET_BASE=/root/autodl-fs/datasets/reality
export TOKENIZER_PATH=/root/autodl-fs/ckpt/google/paligemma-3b-pt-224
export PRETRAINED_PATH=/root/autodl-tmp/piper-revision/materialized/task2-priming-seed0-total50000

# Restore only if model.safetensors is not already present and output directory is absent.
# Check free disk space first; materialization creates a full-sized model copy.
df -h /root/autodl-tmp
if [[ ! -f "$PRETRAINED_PATH/model.safetensors" ]]; then
  if [[ -e "$PRETRAINED_PATH" ]]; then
    echo "Existing incomplete model directory: $PRETRAINED_PATH" >&2
    exit 1
  fi
  python examples/piper/revision_checkpoint.py materialize \
    --bundle /root/autodl-tmp/piper-revision/runs/reality-parallel-v1/task2-priming-seed0-total50000/bundles/total-050000 \
    --base /root/autodl-tmp/piper-revision/runs/reality-parallel-v1/storage_base \
    --output "$PRETRAINED_PATH"
fi

# A800 has one visible card in many AutoDL containers; choose its actual index.
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
GPU_IDS=0 DRY_RUN=true \
  bash examples/training/train_piper_partial_ft.sh action4

# 200-step GPU smoke after DRY_RUN validates; no robot execution.
GPU_IDS=0 FLOW_STEPS=200 WARMUP_STEPS=20 SAVE_STEPS='[100,200]' \
  RUN_GROUP=smoke-action4-autodl DRY_RUN=false \
  bash examples/training/train_piper_partial_ft.sh action4
```

Set `WORK_ROOT` to an actual writable AutoDL work disk, with sufficient space for
full checkpoints. The new launcher detects AutoDL paths and opts out of the
legacy /data1 path migration; other existing Piper launchers keep their original
local-server behavior. Run one profile at a time on the single A800. After smoke,
run `action4`, `dual2`, `dual4` sequentially with distinct `RUN_GROUP` names.

Use the **same dataset identity and deterministic 10% episode split** as the original
50k experiment; new processor stats are refitted from the training split. Copying the
checkpoint alone but changing the dataset or split invalidates direct comparisons.
All new output directories are separate. During real robot deployment, install the
matching PI05 configuration/model code (or carefully port the metadata fields) and
use the saved trained processors, not the base model's processors.

## Smoke checks: GPU1, GPU2, GPU3, each 200 steps

If GPU RAM is unknown, run each first as a **separate** smoke test; do not assume three
physical GPUs' memories add together. One process runs on each card, not 3-card DDP.
Each GPU sees its own `CUDA_VISIBLE_DEVICES` and therefore uses logical cuda:0.

```bash
FLOW_STEPS=200 WARMUP_STEPS=20 SAVE_STEPS='[100,200]' \
  RUN_GROUP=smoke-action4 GPU_IDS=1 DRY_RUN=true \
  bash examples/training/train_piper_partial_ft.sh action4
FLOW_STEPS=200 WARMUP_STEPS=20 SAVE_STEPS='[100,200]' \
  RUN_GROUP=smoke-dual2 GPU_IDS=2 DRY_RUN=true \
  bash examples/training/train_piper_partial_ft.sh dual2
FLOW_STEPS=200 WARMUP_STEPS=20 SAVE_STEPS='[100,200]' \
  RUN_GROUP=smoke-dual4 GPU_IDS=3 DRY_RUN=true \
  bash examples/training/train_piper_partial_ft.sh dual4
```

Repeat each command with `DRY_RUN=false` to train the independent 200-step smoke runs.
Check the resulting `trainable_parameters.json` before the long runs:
`vlm_language_blocks` must be **0 / positive / larger positive** for
`action4 / dual2 / dual4`, respectively. `action_expert_blocks`,
`action_projection` and both prompts must be nonzero in every run.
Unexpected `other` trainables are an error. Check all optimizer groups
for a finite LR and that loss/gradients remain finite; if CUDA OOM occurs,
reduce `BATCH_SIZE` (default=1) or run fewer profiles concurrently.

## 10k training after smoke

Run these only if smoke checks pass. Always give each experiment its own
`RUN_GROUP`; do not reuse an incomplete run or silently resume it.

```bash
# Run from separate terminals (or put each into a separately named nohup log).
FLOW_STEPS=10000 WARMUP_STEPS=500 SAVE_STEPS='[1000,3000,5000,10000]' \
  RUN_GROUP=task2-action4-plus10k GPU_IDS=1 DRY_RUN=false \
  bash examples/training/train_piper_partial_ft.sh action4

FLOW_STEPS=10000 WARMUP_STEPS=500 SAVE_STEPS='[1000,3000,5000,10000]' \
  RUN_GROUP=task2-dual2-plus10k GPU_IDS=2 DRY_RUN=false \
  bash examples/training/train_piper_partial_ft.sh dual2

FLOW_STEPS=10000 WARMUP_STEPS=500 SAVE_STEPS='[1000,3000,5000,10000]' \
  RUN_GROUP=task2-dual4-plus10k GPU_IDS=3 DRY_RUN=false \
  bash examples/training/train_piper_partial_ft.sh dual4
```

Default is BF16 + gradient checkpointing, batch=1 per GPU, no torch.compile, two
data workers, no W&B, no image augmentation, 10 flow denoising steps, chunk50,
h8 execution/validation and 0.1 held-out episode split. If a GPU cannot support
BF16 or a model OOMs, record the changed precision/batch explicitly and compare
within the same evaluation protocol. Checkpoints are at **additional** steps
1k/3k/5k/10k (not cumulative from the original 50k).

Logs:
`$WORK_ROOT/logs/piper/<run-group>/pi05-may-2-remove_the_cuboid_from_blue_plate-dual_prompt_only-seed0.log`.
Models:
`$WORK_ROOT/chkpt/2601-lerobot/piper/<run-group>/pi05-may-2-remove_the_cuboid_from_blue_plate-dual_prompt_only-seed0/checkpoints/010000/pretrained_model`.
The group and saved `config.json` distinguish profiles despite the legacy
`dual_prompt_only` output stem. The action evaluation is
`.../action_eval.jsonl` with `action_val/h8/*`. Retain complete checkpoint
directories including normalization processor files.

Report paired val h8 overall/moving/J2/J3/gripper metrics at 1k/3k/5k/10k,
including hold-position baseline. For offline trajectory pictures, use
`eval_piper_offline.py` on matched held-out episodes with identical noise seeds.
Real robot trials must remain under speed/collision supervision and must not be
started solely because offline loss decreases.

**Validation scope:** code-level grouping checks and launcher preflight;
GPU-dependent memory/gradient and robot execution have not been run by ChatGPT.

## Full Action Head + VLM partial fine-tuning (single A800, recommended)

The `fullhead_vlm2` and `fullhead_vlm4` profiles unfreeze **every** Gemma Action
Expert Transformer layer, final expert norm, action input/output projections, and
the two time-conditioning MLPs, while also unfreezing the final 2/4 PaliGemma
language-model layers. Both learned prompt banks stay trainable. Vision tower,
language embeddings and unused expert LM head stay frozen. Unlike the older
`dual2` profile, this is the **full continuous-action head** rather than the last 4
expert blocks. `cabo_enabled=false`; ordinary differential-LR AdamW is used.

The optimizer contains disjoint groups:
prompts = 2.5e-5, action projections = 1e-5, all action expert core + time MLPs
= 1e-5, selected VLM language layers = 1e-6. A fail-closed check verifies every
configured trainable tensor belongs to exactly one group and no selected parameter
is unexpectedly frozen. `train_action_head=true` requires `train_action_expert_last_n_layers=0`.

On AutoDL (after exporting `WORK_ROOT`, `DATASET_BASE`, `TOKENIZER_PATH`,
and `PRETRAINED_PATH` in the previous section):

```bash
# Free GPU, BF16, microbatch=1. Short smoke first; no motor control.
GPU_IDS=0 FLOW_STEPS=200 WARMUP_STEPS=20 SAVE_STEPS='[100,200]' \
  RUN_GROUP=smoke-task2-fullhead-vlm2-autodl DRY_RUN=true \
  bash examples/training/train_piper_partial_ft.sh fullhead_vlm2

# After checking dry run, repeat with DRY_RUN=false.
GPU_IDS=0 FLOW_STEPS=200 WARMUP_STEPS=20 SAVE_STEPS='[100,200]' \
  RUN_GROUP=smoke-task2-fullhead-vlm2-autodl DRY_RUN=false \
  bash examples/training/train_piper_partial_ft.sh fullhead_vlm2

# Only after actual GPU gradient/finite-loss/peak-memory verification:
GPU_IDS=0 RUN_GROUP=task2-fullhead-vlm2-plus10k-autodl DRY_RUN=false \
  bash examples/training/train_piper_partial_ft.sh fullhead_vlm2
```

`fullhead` (VLM frozen) serves as the essential full-action-head control;
`fullhead_vlm4` is the deeper optional VLM capacity test. Use different
`RUN_GROUP` names. Test each with the same 200-step smoke before allocating
10k. A800 80GB is not a guarantee against OOM: BF16 and checkpointing are on,
but full expert optimizer states + VLM gradients may require extra capacity.
Default checkpoint updates are [1000,3000,5000,10000] **beyond** the source 50k;
check disk space before runs.

Verify `trainable_parameters.json`: action_expert_blocks > 0,
action_expert_norm > 0, action_time_mlp > 0, action_projection > 0,
vlm_language_blocks > 0 for `fullhead_vlm2`; no unexpected 'other' trainables.
The run must also show finite gradients/parameters and updated VLM/Action weights
in a saved smoke checkpoint. The code tests are CPU checks only, not a substitute
for this GPU verification. For controlled mechanism claims, compare
`fullhead` with `fullhead_vlm2` using the same 50k initialization and training
budget, and add a matched Prompt-only continuation control.


## Resume the completed task2 fullhead_vlm2 run from 10k to 20k (AutoDL A800)

This is **not** a fresh 20k training job. It performs **10,000 additional optimizer
updates**, advancing the second adaptation stage from 10,000 to 20,000.
Its source is the final `checkpoints/010000/` from the already-completed
`task2-fullhead-vlm2-plus10k` run. The 50k Prompt-only adaptation remains
the initialization before this second stage. The final policy will have
50k Prompt-only updates plus 20k fullhead+VLM2 updates.

The source 10k schedule reached its LR floor at Step 10000. The resume launcher
preserves AdamW moments and model weights, but deliberately modifies the
**future** LR schedule: linear ramp from floor to **50% of the original group
peak** over 300 updates, then cosine decay to floor by Step 20000.
Group peaks in the resumed portion: prompts 1.25e-5, expert/projectors
5e-6, VLM language blocks 5e-7. These are LR choices, not an exact
uninterrupted replay of a hypothetical original 20k cosine schedule.

```bash
cd /root/autodl-tmp/lerobot060-piper-partial-ft
git fetch origin piper-partial-ft
git switch --detach origin/piper-partial-ft

export WORK_ROOT=/root/autodl-tmp/piper-partial-ft-work
# No PRETRAINED_PATH needed: this script resumes the 10k checkpoint *with optimizer state*.
PYTHONPATH=src python -m pytest tests/scripts/test_piper_partial_ft_resume.py -q
df -h /root/autodl-tmp
nvidia-smi

GPU_IDS=0 DRY_RUN=true \
  bash examples/training/continue_piper_fullhead_vlm2_to20k.sh

GPU_IDS=0 DRY_RUN=false \
  bash examples/training/continue_piper_fullhead_vlm2_to20k.sh
```

Defaults: 1 GPU, BF16, exact saved batch size, held-out split, and all original
dataset/processor settings. The launcher verifies original Step 10000, full
training-state presence, fullhead+VLM2 configuration, non-overlapping output path
and save steps. It refuses incomplete state instead of treating weights-only
initialization as resume.

New independent output run group:
`$WORK_ROOT/chkpt/2601-lerobot/piper/task2-fullhead-vlm2-resume10k-to20k/`.
Intermediate checkpoints: `012000`, `015000`, `020000` (step IDs are
**total in the second adaptation stage**, not 62k/65k/70k).
New action metrics start with Step 10000 and continue through Step 20000; compare
`action_val/h8/*` to the source's Step 10000 with the same fixed evaluation
selection. The original run and saved optimizer state are left untouched.

Check that the training log prints a resume at Step 10000 rather than Step 0
and runs exactly 10k more loop iterations. In the logged LR it should rise
smoothly after the restart boundary rather than remain at 1e-6. The monitor
should report finite loss and J2/J3/gripper validation improvements. Do not
use the robot during this offline training. If the source checkpoint has been
compacted or its `training_state` files were discarded, this **cannot** preserve
AdamW state; use a separate weights-only adaptation procedure instead.
