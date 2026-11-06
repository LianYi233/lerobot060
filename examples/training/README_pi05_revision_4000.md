# PrimingVLA revision: budget-matched 4,000-update control

This is the first rebuttal experiment for the critique that the published
`full_reference` uses 1,000 more optimizer updates than `direct_dual`.
The scripts below keep the original `prompt-ablation` implementation intact and
use the same dual-prompt configuration, initial checkpoint, dataset, optimizer
defaults and CPC/CABO setting unless the ablation itself changes CPC.

| Variant | Action-only SAP | Conditional bridge | Formal flow | Cumulative updates | CPC | Evaluation checkpoint |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `direct_dual` | 0 | 0 | 4,000 | **4,000** | On | `checkpoints/004000/pretrained_model` |
| `dual_prompt_only` | 0 | 0 | 4,000 | **4,000** | Off | `checkpoints/004000/pretrained_model` |
| `full_reference` | 750 | 250 | 3,000 | **4,000** | On | `checkpoints/003000/pretrained_model` |

`direct_dual` is the most important new baseline. `dual_prompt_only` controls
for CPC, while `full_reference` is a reproducible same-budget reference; its
published 89.00% is not a new result from these scripts. A standard evaluation
uses the **four** LIBERO suites and identical evaluation seeds and episodes.

## Training

Checkout the `revision` branch, activate the same LeRobot environment used for
the original experiment, and set paths to the **same** standard LIBERO dataset,
π0.5 starting checkpoint, and tokenizer for all variants. For example:

```bash
export DATASET_ROOT=/path/to/standard/libero
export PRETRAINED_PATH=/path/to/pi05_libero_base
export TOKENIZER_PATH=/path/to/paligemma-3b-pt-224
export OUTPUT_ROOT=/path/to/checkpoints/revision-4000
export LOG_ROOT=/path/to/logs/revision-4000
export GPU_IDS=0

DRY_RUN=true bash examples/training/train_pi05_revision_4000.sh direct_dual 0
bash examples/training/train_pi05_revision_4000.sh direct_dual 0
bash examples/training/train_pi05_revision_4000.sh dual_prompt_only 0
bash examples/training/train_pi05_revision_4000.sh full_reference 0
```

Runs use unique names (`pi05-revision-<variant>-seed<seed>-total4000`), write
checkpoints every 1,000 formal-flow updates, and refuse to overwrite existing
results. Stage/NTK diagnostic snapshots are disabled consistently to avoid
extra disk I/O. Passing `FLOW_STEPS` or a protocol-defining CLI override cannot
silently change the total 4,000-step budget. Use the same `BATCH_SIZE`,
`PRETRAINED_PATH`, `DATASET_ROOT`, `TOKENIZER_PATH`, CUDA precision and optimizer
settings across variants. If the baseline uses a different dataset or pretrained
checkpoint, its score is not an apples-to-apples comparison.

## Evaluation

Use the same `OUTPUT_ROOT` from training (or set `CKPT_ROOT` explicitly), and
the same `EPISODES_PER_TASK`, environment seeds, and evaluation software for
all three variants:

```bash
export CKPT_ROOT="$OUTPUT_ROOT"
export BASE_OUTPUT=/path/to/eval/revision-4000
export GPU_ID=0
export EPISODES_PER_TASK=10

bash run_eval_pi05_revision_4000.sh direct_dual all 0
bash run_eval_pi05_revision_4000.sh dual_prompt_only all 0
bash run_eval_pi05_revision_4000.sh full_reference all 0
```

The revision evaluation wrapper reuses the original LIBERO evaluator and its
resume/progress logic, but selects checkpoint `004000` for the direct variants
and `003000` for the staged method. Both correspond to **4,000 cumulative
updates**, not 4,000 formal-flow updates for the staged method. Evaluation
results are isolated under `BASE_OUTPUT` and labeled by variant/seed.

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
- The Action-only Flow Warm-up control requested by Reviewer 2 is a **separate
  follow-up experiment**; it is not implemented by this matched-budget test.

Quick checks (no GPU needed):

```bash
bash -n examples/training/train_pi05_revision_4000.sh run_eval_pi05_revision_4000.sh
python -m unittest discover -s tests/scripts -p 'test_pi05_revision_4000_launcher.py' -v
```
