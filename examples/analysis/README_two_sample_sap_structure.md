# PrimingVLA: two-sample action-structure probe

This is a deliberately **small, diagnostic Stage-I SAP experiment**, not a LIBERO
success-rate benchmark. It optimizes **only the 16 action prompt tokens (16 × 1024
= 16,384 parameters)** of the actual frozen PI0.5 expert. The original SAP
flow-inpainting code (`PI05Pytorch.forward` with
`PI05TrainingObjective.ACTION_INPAINTING`) is reused, without a proxy network or
observation conditioning. The VLM, action expert, and projection layers stay
frozen; CPC is disabled as in Stage I.

**What counts as two samples?** Exactly **two non-overlapping 50-timestep action
chunks** from two specified LIBERO demonstration episodes, not two frames, two
independent training datasets, or two minibatches drawn from a larger dataset.
The pretrained `pi05_libero_base` and its standard LIBERO quantile statistics
are shared across groups. Therefore, the conclusion concerns the **additional
learning of action prompts**, *not* learning robotics from only two examples or
generalizing to unseen tasks.

**Low-disk-space note:** this script uses **PyArrow's in-memory filtered reads**
of the two episode Parquet files; it does not instantiate `LeRobotDataset` or
convert every LIBERO file into a Hugging Face Arrow cache. Its first output
line also reports the exact LeRobot source path being imported. If a previous
version failed with `Errno 28`, inspect `df -h` and the size of the HF datasets
cache separately before retrying; a partially created cache is not deleted by
this script. The script automatically puts its own worktree `src/` first on
Python's import path, avoiding stale editable installs from other checkouts.

## Hypothesis and mathematical grounding

With MSE-based conditional reconstruction, the population-optimal missing-action
estimator is the conditional expectation

`E[A_masked | A_visible]`.

For this repository's flow convention, `x_t = (1-t) a + t z` and `u_t = z-a`,
so the clean-action estimate is **`a_hat = x_t - t * v_theta`**. It becomes exact
if the estimated velocity is exact. At evaluation, the visible actions remain
clean; only masked actions traverse the noisy flow path.

Related primary papers:

1. Wu et al., **Masked Trajectory Models for Prediction, Representation, and
   Control**, ICML 2023: https://proceedings.mlr.press/v202/wu23o.html
2. Lipman et al., **Flow Matching for Generative Modeling**, ICLR 2023:
   https://mlanthology.org/iclr/2023/lipman2023iclr-flow/
3. Alain and Bengio, **What Regularized Auto-Encoders Learn from the
   Data-Generating Distribution**, JMLR 2014:
   https://www.jmlr.org/papers/v15/alain14a.html

The papers justify masked trajectory reconstruction, conditional-flow
regression, and local reconstruction sensitivity as *analytical motivations*;
they do not prove that this finite-sample SAP experiment will succeed.

## Three matched training conditions

| Variant | Training input | Hidden positions | Main question |
| --- | --- | --- | --- |
| `sap` | Correctly ordered 50-step actions | Random 40/50 | Can SAP use temporal context? |
| `shuffled` | A **fresh per-update time permutation** of the same two chunks | Random 40/50 | Does destroying temporal order hurt? |
| `full_mask` | Same two original chunks | All 50/50 | Can a context-free action prior explain the effect? |

All three have the **same two chunks, checkpoint, random seed, prompt
initialization, AdamW/scheduler recipe, and number of updates**. The 750 updates
use the *first 750 steps of the original 1000-update Stage-I LR schedule*,
which continues through the 250 bridge updates in the main experiment. The shuffled
run preserves the multiset of actions and each channel's marginal values at
every update, but randomizes their time coordinates independently. The
full-mask control has more supervised positions (50 versus 40), which must be
acknowledged when interpreting its comparison. The `init` control is the exact
prompt initialization saved before any update.

## Launch on the PrimingVLA server

Use the new `sap-two-sample-structure` branch, install this checkout with
`pip install -e .` in the existing LeRobot environment, and confirm that
the paths below exist. No existing training checkpoint is modified.

```bash
git fetch origin sap-two-sample-structure
git switch --track origin/sap-two-sample-structure

export DATASET_ROOT=/root/autodl-fs/datasets/libero
export BASE_PI05=/root/autodl-fs/ckpt/pi05_libero_base
export OUT_ROOT=/root/autodl-tmp/wyn/chkpt/2601-lerobot/two-sample-sap

# Example: start 10 frames into each of episodes 0 and 1. Each must contain
# at least 60 frames; the code rejects padded or overlapping windows.
for variant in sap shuffled full_mask; do
  CUDA_VISIBLE_DEVICES=0 python examples/analysis/two_sample_sap_structure.py train \
    --variant "$variant" \
    --dataset-root "$DATASET_ROOT" \
    --base-checkpoint "$BASE_PI05" \
    --chunks 0:10,1:10 \
    --steps 750 --seed 0 \
    --device cuda \
    --output "$OUT_ROOT/$variant"
done

CUDA_VISIBLE_DEVICES=0 python examples/analysis/two_sample_sap_structure.py analyze \
  --sap "$OUT_ROOT/sap" \
  --shuffled "$OUT_ROOT/shuffled" \
  --full-mask "$OUT_ROOT/full_mask" \
  --output "$OUT_ROOT/analysis" \
  --repeats 8 --masked-steps 40 --t 0.5
```

The commands train **sequentially** on one GPU. You can use `--steps 150` for a
quick smoke test, but all three variants must use the same step count; use
`750` for the paper's exact SAP preparation budget. Each run refuses to
overwrite an existing output directory. Select two full windows from the
**same task** if possible, with different temporal patterns; check the episode
metadata to ensure that choice before running. Action channels follow the
dataset, usually 7 for LIBERO; they are padded to the model's internal 32D.

**Outputs**:

- Each training directory: `two_chunks.pt`, `action_prompt.pt`,
  `train_metadata.json` (including a hash that verifies sample identity).
- Analysis directory: `metrics.json`, `two_sample_structure.png`,
  `context_influence.png`, `influence.npz`.

The saves contain only action chunks and prompt weights, not multi-GB copies
of the frozen PI0.5 backbone.

## Metrics and interpretation

Evaluation samples **new deterministic random masks and Gaussian noise** on
these **same** two training chunks; 8 repeats are 8 corrupted views, **not**
8 independent test trajectories. The 40 masked timesteps are scored in the
original action dimensions. For an identical mask and noise vector, a second
evaluation permutes only the **10 visible context actions** while keeping all
masked noisy inputs and ground-truth targets fixed.

- **Masked MSE** (`masked_mse`; smaller is better): one-step conditional
  reconstruction error on previously unseen masks and noise. Compare `sap`
  with `init`, `shuffled`, `full_mask`, and simple linear interpolation.
- **Context-order gain** (`context_order_gain`):
  `MSE(permuted visible context) - MSE(correct visible context)`.
  Positive gain means that the *correctly ordered* visible context helps;
  positive gain **together with** better masked MSE is stronger evidence of
  useful contextual structure than lower training loss alone.
- **Directional influence map**: a finite-difference approximation to
  `|| d a_hat[h, :] / d visible_action[k, d] ||_2` for one selected input
  channel `d` (default 0). Only the 10 visible columns and 40 masked output
  rows are meaningful; a nonzero response alone does *not* prove learning.

### Pre-registered decision rule

Evidence **consistent with action temporal-structure learning** would be:

1. `MSE(SAP) < MSE(initial)` (the prompt improved masked reconstruction),
2. `MSE(SAP) < MSE(shuffled)` (time order matters beyond the action marginals),
3. `MSE(SAP, correctly ordered context) < MSE(SAP, permuted context)`
   (predictions make useful use of visible context).

The `full_mask` and linear-interpolation comparisons diagnose how much can be
explained by an unconditional prior or trivial smoothness. If the three primary
conditions fail, do **not** claim that the two-sample experiment demonstrates
temporal structure; it may indicate memorization, pretraining effects, weak
context use, or simply insufficient 750-step adaptation.

The analysis probes *conditional reconstruction on two familiar trajectories*
and *local dependence on visible actions*. It does not estimate mutual
information, prove physically correct dynamics, isolate entirely new structures
from those already present in the frozen pretrained expert, or establish
generalization to unseen trajectories. With only two independent chunks, CKA
and population covariance estimates are inappropriate as primary metrics.

## V2: full flow sampling, action transitions, and path-signature geometry

The evaluation-only companion `two_sample_sap_structure_v2.py` adds **two
structural measures** while keeping the original `analyze` output unchanged:

1. **Action Transition Recovery** in the last action dimension (LIBERO gripper
   by default). It detects signed state changes at threshold zero, counts
   only transitions whose *two adjacent ground-truth positions* are masked,
   and reports recall within a two-step tolerance, timing MAE among matched
   transitions, and false transitions. If no transition is eligible (or none
   is recovered), timing MAE is `null`; this is not zero error. A separate
   first-difference MSE covers continuous motion dimensions.
2. **Second-order Path Signature Error**, using the antisymmetric level-2
   iterated integral (Lévy area) of the `action + normalized time` path.
   It measures ordered coupling and timing. Absolute and relative L2 errors
   are reported; a near-zero reference signature yields a `null` relative
   error. A low error alone does not establish learning: compare against
   matched controls and linear interpolation.

Unlike the old single-step denoising probe, V2 begins **every masked action
dimension at pure Gaussian noise at flow time t=1**, Euler-integrates the
actual Stage-I `predict_inpainting_velocity` to `t=0`, and hard-clamps the
visible action values after every step. All padded internal action dimensions
also start with noise when masked. Ground-truth hidden actions do *not* enter
the sampler. Initial, SAP, shuffled, and full-mask prompts reuse identical
noise/masks/inference budgets; interpolation uses the same visible positions.

Two mask protocols are reported separately:

- `random`: 40 of 50 timesteps selected randomly (as in SAP training).
- `long_gap`: the **fixed middle 40/50 timesteps** masked, with five visible
  steps at each end. Because the mask location is fixed, it does not reveal
  ground-truth event timing. This tests much longer-range completion.

### Run without retraining (recommended first)

```bash
cd /root/autodl-tmp/lerobot060-sap-two-sample
export OUT_ROOT=/root/autodl-tmp/wyn/chkpt/2601-lerobot/two-sample-sap
export DATASET_ROOT=/root/autodl-fs/datasets/libero

CUDA_VISIBLE_DEVICES=0 python examples/analysis/two_sample_sap_structure_v2.py \
  --sap "$OUT_ROOT/sap" \
  --shuffled "$OUT_ROOT/shuffled" \
  --full-mask "$OUT_ROOT/full_mask" \
  --output "$OUT_ROOT/analysis_v2" \
  --ode-steps 16 --repeats 2 --masked-steps 40 --batch-size 1
```

The default evaluation uses the **original two training chunks**, under new
masks/noise. This is a diagnostic, not an independent test. To test **unseen
trajectories while continuing to train on only two chunks**, use different
episodes with the same low-memory Parquet reader:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/analysis/two_sample_sap_structure_v2.py \
  --sap "$OUT_ROOT/sap" \
  --shuffled "$OUT_ROOT/shuffled" \
  --full-mask "$OUT_ROOT/full_mask" \
  --output "$OUT_ROOT/analysis_v2_heldout" \
  --dataset-root "$DATASET_ROOT" \
  --eval-chunks 2:10,3:10,4:10,5:10 \
  --ode-steps 16 --repeats 2 --batch-size 1
```

The selectors are examples; every window must fit 50 unpadded frames. All
episodes containing the training chunks (here 0 and 1) are rejected. For
stronger generalization evidence use 20+ held-out episodes. Repeated masks
are **not** independent trajectories.

The outputs are `metrics_v2.json`, `per_chunk_v2.json`, and
`structure_v2.png`; the original `analysis/` files and all checkpoints are
left untouched. Empty event recall is `null`, never silently treated as zero.

**More training steps?** Not yet. The **750 SAP updates** already reproduce
the original Stage-I priming budget, and all three controls received the
same number of updates. Check full-ODE structural metrics first. If SAP
still fails to outperform controls, a distinct 1500/3000-update sensitivity
study could retrain **all variants** with equal budgets and evaluate on
independent episodes; longer training of two examples may just memorize them.

Further primary references:

- Chevyrev and Kormilitzin, *A Primer on the Signature Method in Machine
  Learning*: https://arxiv.org/abs/1603.03788
- Barcelos et al., *Path Signatures for Diversity in Probabilistic Trajectory
  Optimisation*, IJRR 2024: https://doi.org/10.1177/02783649241233300
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023:
  https://arxiv.org/abs/2210.02747
