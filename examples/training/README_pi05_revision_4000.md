# PrimingVLA revision: budget-matched 4,000-update controls

For the external prompt-free **PI0.5 + LoRA** baseline, see
[`README_pi05_lora_baseline.md`](README_pi05_lora_baseline.md). Its variant is
`lora_r16`: direct 4,000-update flow training, rank/alpha 16, no SAP/CPC/prompts,
with the same initialization and data settings. Existing prompt controls and
default multi-GPU schedules are unchanged.

This is the first rebuttal experiment for the critique that the published
`full_reference` uses 1,000 more optimizer updates than `direct_dual`.
The scripts build on the original `prompt-ablation` training implementation and
use the same dual-prompt configuration, initial checkpoint, dataset, optimizer
defaults and CPC/CABO setting unless the ablation itself changes CPC.

| Variant | Action-only warm-up | Conditional bridge | Formal flow | Cumulative updates | CPC | Evaluation checkpoint |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `direct_dual` | 0 | 0 | 4,000 | **4,000** | On | `checkpoints/004000/pretrained_model` |
| `dual_prompt_only` | 0 | 0 | 4,000 | **4,000** | Off | `checkpoints/004000/pretrained_model` |
| `full_reference` | 750 | 250 | 3,000 | **4,000** | On | `checkpoints/003000/pretrained_model` |
| `action_flow_warmup` | 750 full-mask flow | 250 | 3,000 | **4,000** | On | `checkpoints/003000/pretrained_model` |

`direct_dual` is the most important new baseline. `dual_prompt_only` controls
for CPC, while `full_reference` is a reproducible same-budget reference; its
published 89.00% is not a new result from these scripts. The additional
`action_flow_warmup` uses `next_action_full_mask_probability=1.0`, which masks
**all valid positions** during the 750 action-only updates. No visible action
tokens remain, making this a full-action flow-matching warm-up instead of
40-of-50 masked inpainting. The same 250-step conditional bridge follows.
A standard evaluation
uses the **four** LIBERO suites and identical evaluation seeds and episodes.

## Training

Checkout the `revision` branch and activate the same LeRobot environment used
for the original experiment. The current defaults are:

| Variable | Default path |
| --- | --- |
| `DATASET_ROOT` | `/root/autodl-fs/datasets/libero` |
| `PRETRAINED_PATH` | `/root/autodl-fs/ckpt/pi05_libero_base` |
| `TOKENIZER_PATH` | `/root/autodl-fs/ckpt/google/paligemma-3b-pt-224` |
| `OUTPUT_ROOT` | `/root/autodl-tmp/wyn/chkpt/2601-lerobot/revision-4000` |
| `LOG_ROOT` | `/root/autodl-tmp/wyn/logs/revision-4000` |

The revision launcher sets all three input paths automatically. You can still
override any path with an environment variable if running on another server:

```bash
export GPU_IDS=0

DRY_RUN=true bash examples/training/train_pi05_revision_4000.sh direct_dual 0
bash examples/training/train_pi05_revision_4000.sh direct_dual 0
bash examples/training/train_pi05_revision_4000.sh dual_prompt_only 0
bash examples/training/train_pi05_revision_4000.sh full_reference 0
bash examples/training/train_pi05_revision_4000.sh action_flow_warmup 0
```

### Four independent GPUs

If GPU 0 has **already started** `direct_dual`, do not restart it. First check
the commands without launching anything:

```bash
DRY_RUN=true bash examples/training/launch_pi05_revision_4gpu.sh others 0
```

Then launch the remaining three independent processes in the background:

```bash
bash examples/training/launch_pi05_revision_4gpu.sh others 0
```

| GPU | Variant | Cumulative updates |
| --- | --- | ---: |
| 0 (existing) | `direct_dual` | 4,000 |
| 1 | `dual_prompt_only` | 4,000 |
| 2 | `full_reference` | 4,000 |
| 3 | `action_flow_warmup` | 4,000 |

The script preflights all pending runs before launching them, refuses to
overwrite an existing output, and prints each job's PID and startup log path.
Each job also writes its normal training log under `LOG_ROOT`. It selects one
physical GPU per process via `CUDA_VISIBLE_DEVICES`, and isolates compiler
caches. Do not run `all` when GPU 0 is already active.

Runs use unique names (`pi05-revision-<variant>-seed<seed>-total4000`) and refuse
to overwrite existing results. Prompt controls now save only their final flow
checkpoint by default; set `SAVE_FREQ=1000` to retain intermediate checkpoints.
The external `lora_r16` baseline defaults to every 1,000 updates because it saves
compact adapters. Stage/NTK diagnostic snapshots are disabled consistently to avoid
extra disk I/O. Passing `FLOW_STEPS` or a protocol-defining CLI override cannot
silently change the total 4,000-step budget. Use the same `BATCH_SIZE`,
`PRETRAINED_PATH`, `DATASET_ROOT`, `TOKENIZER_PATH`, CUDA precision and optimizer
settings across variants. If the baseline uses a different dataset or pretrained
checkpoint, its score is not an apples-to-apples comparison.

## Evaluation

Use the same checkpoint root as training (the default below, or set
`CKPT_ROOT` to your overridden `OUTPUT_ROOT`), and the same
`EPISODES_PER_TASK`, environment seeds, and evaluation software for all three
variants:

```bash
export CKPT_ROOT=/root/autodl-tmp/wyn/chkpt/2601-lerobot/revision-4000
export BASE_OUTPUT=/root/autodl-tmp/wyn/eval/2601-lerobot/revision-4000
export GPU_ID=0
export EPISODES_PER_TASK=10

bash run_eval_pi05_revision_4000.sh direct_dual all 0
bash run_eval_pi05_revision_4000.sh dual_prompt_only all 0
bash run_eval_pi05_revision_4000.sh full_reference all 0
bash run_eval_pi05_revision_4000.sh action_flow_warmup all 0
```

The revision evaluation wrapper reuses the original LIBERO evaluator and its
resume/progress logic, but selects checkpoint `004000` for the direct variants
and `003000` for the staged method. Both correspond to **4,000 cumulative
updates**, not 4,000 formal-flow updates for the staged method. Evaluation
results are isolated under `BASE_OUTPUT` and labeled by variant/seed.

LIBERO simulation assets are used at **evaluation**, not at training time.
The assets on this server are at `/root/autodl-fs/wyn/libero-assets/`.
Official LIBERO resolves assets from the `assets:` entry in
`$LIBERO_CONFIG_PATH/config.yaml` (normally `~/.libero/config.yaml`); check
that this YAML entry points to that directory before evaluating. Do **not**
set `LIBERO_CONFIG_PATH` to the assets directory itself: it must name the
directory containing `config.yaml`. Other YAML paths such as `bddl_files` and
`init_states` must remain valid.

## Interpretation and limits

- This controls the **number of optimizer updates**, not exact GPU-hours.
  SAP uses a different forward path, so report wall-clock time and peak memory
  separately before claiming equal compute.
- The formal-flow cosine scheduler spans 4,000 updates for direct training and
  3,000 updates for staged training, because the latter also has a 1,000-step
  preparation objective. The scheduler is part of the compared training recipe;
  do not call the result an isolated test of the SAP objective.
- Report all four per-suite success rates and the macro average. Repeat with
  additional **training** seeds and fixed evaluation conditions to assess
  stability before making a causal or generalization claim.
- The full-mask Action-only Flow Warm-up control holds update counts and
  trainable parameters fixed relative to SAP. It changes the **action masking
  pattern**, so it tests the benefit of preserving visible action context. It
  does not isolate every possible warm-up initialization strategy.

Quick checks (no GPU needed):

```bash
bash -n examples/training/train_pi05_revision_4000.sh examples/training/launch_pi05_revision_4gpu.sh run_eval_pi05_revision_4000.sh
python -m unittest discover -s tests/scripts -p 'test_pi05_revision_4000_launcher.py' -v
```
