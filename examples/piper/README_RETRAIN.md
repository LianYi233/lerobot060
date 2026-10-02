# Piper task retraining after recorded-action diagnostics

This is a capacity experiment for real Piper action fitting, separate from the
0.05M prompt-only method. It does not change labels, hardware control, collision
protection, or existing launcher defaults.

## What changes

`train_piper_retrain.sh TASK expert_last2 SEED` starts from the base policy and uses:

| Setting | Value |
| --- | --- |
| Training data | All episodes before a per-task 10% episode holdout |
| Stages | 750 action priming + 250 conditioning bridge + 3000 formal flow updates |
| Trainable modules | Both prompt banks, action input/output maps, last two action expert blocks |
| Trainable parameters (default model) | 47,313,952 total: 47,198,208 in expert blocks + 115,744 prompts/maps |
| Frozen modules | Entire VLM, other expert blocks, time maps, final expert norm |
| Formal-flow peak learning rates | Prompts/maps: 1e-4; expert blocks: 1e-5 |
| LR schedule | Existing cosine schedule; expert/base LR ratio remains 0.1 |
| Precision | FP32; gradient checkpointing on; compile off |
| CABO / PEFT wrappers | Off / off; full checkpoints |
| Action horizon | Predict 50, execute first 8 |
| Action evaluation | Every 500 updates, 128 fixed observations per split |
| Observation sampling | Budget spread across episodes and temporal bins, fixed seed |
| Formal checkpoints | 1000, 2000, 3000, plus separate best links among these saved models |

The validation episodes do not participate in any stage's optimizer updates.
The existing normalization pipeline / dataset metadata statistics are retained;
the new audit does not refit statistics. Temporal bins are not semantic grasp-phase
labels: review the sample manifest and recorded videos to verify grasp coverage.

Each selected expert block is trainable in priming, bridge and formal flow.
Priming and bridge retain the existing fixed recipe: base peak LR 2.5e-5,
expert-block peak LR 2.5e-6. The formal flow stage starts a fresh optimizer/schedule.
No new model tensors are introduced, so full checkpoints use the existing
deployment architecture. Both training and deployment must pull this `piper`
revision to parse the new config fields. The existing `projections` profile is
available through this launcher as a matched frozen-expert control.

## Run task 1 on the training server

Run inside the training environment with Transformers **5.5.4**, matching the
previous training run and the updated deployment environment. Do not install
dependencies into the original robot environment just for this experiment.

```bash
cd /home/wyn/VLAA/lerobot060
git pull --ff-only origin piper

export WORK_ROOT=/data1/wyn/piper-work
export DATASET_BASE=/data/datasets/May-pick-and-place
export PRETRAINED_PATH=/data/models/lerobot/pi05_libero_base
export TOKENIZER_PATH=/data/models/paligemma-3b-pt-224
export GPU_IDS=0,1
export NUM_PROCESSES=2
export BATCH_SIZE=16
export RUN_GROUP="piper-task1-expert-last2-$(date +%Y%m%d-%H%M%S)"
export OUTPUT_ROOT="/data1/wyn/chkpt/2601-lerobot/${RUN_GROUP}"
export LOG_ROOT="/data1/wyn/logs/${RUN_GROUP}"

DRY_RUN=true bash examples/training/train_piper_retrain.sh 1 expert_last2 0
DRY_RUN=false bash examples/training/train_piper_retrain.sh 1 expert_last2 0
```

`GPU_IDS=2,3` selects physical cards 2 and 3 instead. Batch size is per process;
two cards with batch 16 give global batch 32. Use a new run group for every run.
Full checkpoint and optimizer files consume substantially more disk than prompt
adapters; `best_*` links do not duplicate them. For other training lengths, set
`FLOW_STEPS` and `SAVE_STEPS` together.

## Raw data audit

Before allocating a policy, a real launch reads all raw parquet rows and saves
`${LOG_ROOT}/pi05-may-TASK-no_cabo-seed0-data-audit.json`. A dry run only prints
this command; it does not inspect parquet contents.

To inspect data separately without training:

```bash
PYTHONPATH="$PWD/src" python -m lerobot.utils.piper_data_audit \
  --dataset_root "$DATASET_BASE/1-put_the_apple_on_the_yellow_plate" \
  --output "$LOG_ROOT/task1-data-audit.json"
```

The report compares `action[t]` with `state[t]`, `state[t+1]` and `state[t-1]`
without crossing episode boundaries. It verifies coordinate names, finite values,
row counts, contiguous frame indices and increasing timestamps. It reports
per-episode equality, per-coordinate errors and coordinate ranges. Large equality
is reported as a semantic observation, not automatically treated as corruption:
some datasets intentionally label the measured trajectory. Confirm how collection
records commanded versus measured positions. Parquet timestamps alone cannot
verify camera exposure timing. The audit neither shifts labels nor rewrites data.

## Choose and evaluate a checkpoint

The formal run is:

```text
$OUTPUT_ROOT/pi05-may-1-put_the_apple_on_the_yellow_plate-no_cabo-seed0/
```

- `training_diagnostics.jsonl`: flow loss, gradient/update measurements, trainable
  parameter count, expert-block updates and actual per-group learning rates.
- `action_eval.jsonl`: physical joint/gripper errors for training and held-out
  observations at horizons 1, 8 and 50, including the same-state baseline.
- `action_eval_samples.json`: exact fixed episode/frame selections.
- `best_action_checkpoints.json`: selected saved steps, metric, value and split.
- `checkpoints/best_joint/pretrained_model`: lowest held-out first-8 joint MAE
  **among saved checkpoints**.
- `checkpoints/best_gripper/pretrained_model`: separately selected lowest held-out
  first-8 gripper MAE among saved checkpoints.
- `checkpoints/003000/pretrained_model`: final formal model.

With `EVAL_SPLIT=0`, selection explicitly falls back to training errors; this is
not validation performance. There is no weighted sum of degrees and gripper units,
no automatic success detector, and no claim that the offline winner will solve the
robot task. Evaluate both joint and gripper curves before robot deployment.

## Controlled comparisons

Use the same split, batch size, seed, training length and evaluation observations:

1. Main experiment: `train_piper_retrain.sh 1 expert_last2 0`.
2. Capacity control: a fresh `RUN_GROUP`, `OUTPUT_ROOT` and `LOG_ROOT`, then
   `train_piper_retrain.sh 1 projections 0`.
3. Separate horizon experiment: another fresh output group, set `CHUNK_SIZE=16
   MASKED_STEPS=12 N_ACTION_STEPS=8`, then run the chosen profile. This changes the
   training target and priming mask; compare first-8-action metrics, not raw flow loss.

Do not combine a capacity change, new action representation, changed labels and a
short horizon in the first comparison. If the raw action convention is wrong,
correct it from collection evidence before interpreting any retraining result.
