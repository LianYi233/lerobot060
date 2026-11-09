# Fixed action-prompt learning-rate controls for CPC

These controls test whether a smaller **fixed action-prompt LR multiplier** is
enough to replace CPC (named CABO in the code). They are not full-parameter
fine-tuning: both backbones and the projection layers remain frozen, and only
the existing 16 VLM prompts and 16 action prompts train.

## Matched protocol

All three variants start a fresh run from the same pretrained checkpoint as
`full_reference` and `full_no_cpc`. They use the same training seed, data,
batch size, optimizer recipe, masks, stage boundaries, and LR schedules.

| Stage | Updates | CPC | Action-prompt LR | VLM-prompt LR |
| --- | ---: | --- | --- | --- |
| Action-only SAP | 750 | Off | Original scheduled LR | Original SAP behavior; no VLM update |
| Conditional bridge | 250 | Off | Scheduled LR multiplied by `c` | Original scheduled LR |
| Formal flow | 3,000 | Off | Scheduled LR multiplied by `c` | Original scheduled LR |

The cumulative budget is **4,000 updates**, and the final formal-flow
checkpoint is `003000`, not `004000`.

| Variant | Fixed multiplier `c` | Suggested GPU |
| --- | ---: | ---: |
| `fixed_action_lr_050` | 0.50 | 0 |
| `fixed_action_lr_025` | 0.25 | 1 |
| `fixed_action_lr_010` | 0.10 | 2 |

The multiplier is constant; the LR itself still follows the original schedule.
Only the action group's actual AdamW LR is temporarily scaled during each
optimizer step, then restored before advancing the scheduler. This scales
decoupled weight decay in the same way as CPC's LR implementation. It does not
rescale gradients, alter the VLM LR, or run the adaptive CPC candidate estimator.

The configuration field is `policy.action_prompt_lr_scale` (default `1.0`).
Values must be finite and in `(0, 1]`; a value below one is incompatible with
`cabo_enabled=true`. The three revision variants fix their own multiplier and
reject CLI attempts to override it, so the run name cannot silently disagree
with the experiment configuration.

## Check first, then train

Use the existing `lerobot` environment and the `revision` branch. The input and
output paths remain those in `train_pi05_revision_4000.sh`:

- Pretrained checkpoint: `/root/autodl-fs/ckpt/pi05_libero_base`
- Dataset: `/root/autodl-fs/datasets/libero`
- Tokenizer: `/root/autodl-fs/ckpt/google/paligemma-3b-pt-224`
- Checkpoints: `/root/autodl-tmp/wyn/chkpt/2601-lerobot/revision-4000`
- Logs: `/root/autodl-tmp/wyn/logs/revision-4000`

Preserve any overrides used for the reference runs. Changing the initial
checkpoint or dataset here would no longer be a controlled LR comparison.

First inspect available disk space and the commands without starting training:

```bash
df -h /root/autodl-tmp

GPU_IDS=0 DRY_RUN=true bash examples/training/train_pi05_revision_4000.sh fixed_action_lr_050 0
GPU_IDS=1 DRY_RUN=true bash examples/training/train_pi05_revision_4000.sh fixed_action_lr_025 0
GPU_IDS=2 DRY_RUN=true bash examples/training/train_pi05_revision_4000.sh fixed_action_lr_010 0
```

For each command, verify `SAP=750 bridge=250 flow=3000 total=4000 CPC=false`
and the expected `action_prompt_lr_scale`. DRY_RUN still verifies that the
input directories exist and that the output will not overwrite an existing run.

When there is sufficient disk and host memory, run these in three separate
terminals. Otherwise run one at a time. Training uses **`GPU_IDS`**, while the
evaluation wrapper uses **`GPU_ID`**.

```bash
# Terminal 1
GPU_IDS=0 bash examples/training/train_pi05_revision_4000.sh fixed_action_lr_050 0

# Terminal 2
GPU_IDS=1 bash examples/training/train_pi05_revision_4000.sh fixed_action_lr_025 0

# Terminal 3
GPU_IDS=2 bash examples/training/train_pi05_revision_4000.sh fixed_action_lr_010 0
```

Do not use `launch_pi05_revision_4gpu.sh others` for this batch; that older
launcher assigns a training job to GPU3, which is reserved for evaluation.

Each run has its own name, for example
`pi05-revision-fixed_action_lr_025-seed0-total4000`, and refuses to overwrite
existing outputs. Do not resume these controls from a CPC-trained `001000`
preparation checkpoint: it already contains 250 bridge updates under a different
controller. The shared stage before any LR treatment ends at cumulative 750.

## Checkpoints and storage

By default the revision wrapper disables diagnostic snapshots and saves:

1. The necessary preparation transfer at the sibling
   `..._next_action_pretrain/checkpoints/001000/pretrained_model`.
2. The final model at `.../checkpoints/003000/pretrained_model`.

The new `preserve_pi05_flow_save_freq=true` option makes the revision launcher's
`SAVE_FREQ` effective after preparation; older non-revision launchers keep
their historical behavior. Use `SAVE_FREQ=1000` only if intermediate full
checkpoints are actually needed.

At the previously observed approximately 16G per weight file, three runs need
about **96G for these six weight files alone**, plus optimizer state, compiler
caches, logs, and temporary space. Do not start all three on a nearly full disk.
These scripts never automatically delete old checkpoints or results.

## Verify that the multiplier is actually active

At each normal logging interval, the training log prints `FIXED_ACTION_LR`
even when W&B is disabled. In SAP it reports `applied=1` and `sap_bypass=1`.
During the bridge and formal flow it reports the configured multiplier,
`sap_bypass=0`, and both group LRs. For `fixed_action_lr_025`, the action LR
should be one quarter of the VLM LR in those latter stages; CPC remains off.

The numerical `step` counter restarts at zero for formal flow, as in the
reference training. Use the stage-transition messages when reading the log.

## Sequential evaluation on GPU3

After training completes:

```bash
GPU_ID=3 EPISODES_PER_TASK=10 \
bash run_eval_pi05_revision_4000_sequential.sh 0 \
  fixed_action_lr_050 fixed_action_lr_025 fixed_action_lr_010
```

The existing resumable evaluation protocol is reused. It retains per-task
results on disk but prints progress/ETA and the four-suite plus Overall table.
The default list of models in the sequential script is unchanged; explicitly
provide these three variant names. This invocation does not wait for an
unfinished training run: start it after the needed checkpoints are complete.

Compare these fixed-LR results with `full_no_cpc` (`c=1`) and
`full_reference` (adaptive CPC), using the same trained seeds and evaluation
conditions. The three multipliers are hyperparameter settings, **not three
training-seed repetitions**. Choose a multiplier on a separate, prespecified
validation protocol, then repeat the locked multiplier across common training
seeds before making a test-set superiority claim. If only exploratory test
results are available, report the whole sweep transparently as exploratory.

## Tests

Dependency-light dispatch and control-flow checks (no Torch or GPU required):

```bash
python -m unittest discover -s tests/scripts -p 'test_pi05_revision_4000_launcher.py' -v
python -m unittest discover -s tests/scripts -p 'test_pi05_fixed_action_lr_contracts.py' -v
```

Real CPU AdamW and trainer integration tests in the installed LeRobot environment:

```bash
python -m pytest \
  tests/policies/pi0_pi05/test_pi05_fixed_action_lr.py \
  tests/policies/pi0_pi05/test_pi05_cabo.py \
  tests/optim/test_cabo.py \
  tests/scripts/test_lerobot_train_gradient_clipping.py \
  tests/scripts/test_pi05_integrated_pretraining.py -q
```

These unit tests do not train the full model or start LIBERO rollouts.
