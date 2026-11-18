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

## V4: Four data-scale experiments on a local A100 (all outputs on /data1)

V4 is the **new** episode-disjoint experiment. It is separate from the original
two-chunk V1/V2/V3 scripts and keeps their data, prompts, and results unchanged.
The immediate question is whether having only two independent training chunks
prevents SAP from separating from a time-shuffled control.

Four data scales run **sequentially**: `N=2,32,128,512`, with two matched
training groups (`sap`, `shuffled`) at each scale. This means **8 training
runs**, followed by a held-out flow-field evaluation for each scale. `init`
is evaluated without training. Full-mask and downstream Bridge/Adaptation
comparisons are deliberately deferred until these first data-scale results are
interpretable; they are not silently claimed as part of the four-run sweep.

**Dataset design:** A one-time preparation stage reads LIBERO's episode metadata,
splits episodes by task when that metadata is available, and selects up to one
full 50-step action chunk per episode. Training and testing share **no episode
IDs**. Action windows are fetched with a filtered Arrow scan of the full
`data/` Parquet directory by global `episode_index` (without building a Hugging
Face Arrow cache). Training chunks are ordered in
three interleaved motion categories (gripper transition, continuous movement,
steady) to avoid choosing only the first few, mostly static actions. The train
sets are nested: `train_2 ⊂ train_32 ⊂ train_128 ⊂ train_512`.
Every experiment evaluates on the same **200 independent test episodes**
(provided the dataset contains enough valid episodes). Repeated masks/noise and
flow times are *not* counted as independent test episodes.

The reader uses the per-episode `frame_index` if available, falling back to
ordered `index` values within each episode. It **does not trust** the metadata
`data/chunk_index` and `data/file_index` as physical locations: in the user's
merged LIBERO copy, the metadata points episode 183 to `file-012.parquet` but
that file holds episodes 29/30; the full data directory correctly contains all
303 rows of episode 183. The reader validates the actual row count against the
metadata episode length and rejects missing, incomplete, or noncontiguous
episodes with explicit diagnostics. Preparation stops if fewer than 512 valid
train windows or 200 valid held-out windows remain.

**Training design:** The PI0.5 LIBERO checkpoint, action Prompt initialization,
number of trainable parameters (16 × 1024), 750 optimization steps, batch size
of 2, 40/50 randomly masked training timesteps, and model optimizer/scheduler
are shared by all groups. At each update `shuffled` performs a fresh temporal
permutation on the same sampled action chunks, retaining their value
distribution. The 2-chunk group is retrained with V4's **nested** sample choice:
the old two-chunk checkpoint does not share V4's training data.

**Primary evaluation:** PI0.5's exact masked conditional flow target
`u_t=z-a` on 200 held-out action chunks with 40/50 masked timesteps, at
flow times `0.1,0.3,0.5,0.7,0.9`. This is an action-only Flow Matching
diagnostic, **not** a complete-ODE motion sample. As a separately labeled
mask-generalization test, V4 also evaluates 20/50 masking. The primary scalar
is episode-averaged flow-target *relative MSE*. A positive
`MSE(shuffled) - MSE(sap)` supports a relative benefit of correctly ordered
training, although not by itself a proof of identifiable action dynamics.
The paired 95% bootstrap interval resamples **episodes**, not noise/mask trials;
train-seed uncertainty requires additional repeated training runs. The evaluation
also reports the relative change in predicted velocity caused by learned prompts
versus the saved common initialization.

### Local A100: launch all four data-scale groups in order

Use a local checkout of `LianYi233/lerobot060` on the
`sap-two-sample-structure` branch. Install it into the existing GPU
environment (`pip install -e .`). Required Python packages include PyTorch,
PyArrow, NumPy, and the LeRobot dependencies; Matplotlib is used to draw a
summary figure. No GPU training or checkpoint upload is performed by this README.

The paths below are the LIBERO dataset and `pi05_libero_base` locations
specified for the user's local A100 server. Verify these locations before
launching. The **inputs** are read from `/data`; all **new training outputs**
and caches are saved under `/data1`, not under the old AutoDL paths.

```bash
cd /path/to/your/lerobot060
git fetch origin sap-two-sample-structure
git switch sap-two-sample-structure
git pull --ff-only origin sap-two-sample-structure

export DATASET_ROOT=/data/datasets/libero
export BASE_PI05=/data/models/lerobot/pi05_libero_base/

# All new prompts, prepared action data, logs, figures and caches go to /data1.
export OUT_ROOT=/data1/wyn/chkpt/2601-lerobot/sap-structure-v4
export GPU_ID=0

bash examples/analysis/run_sap_structure_v4_local_a100.sh
```

The runner validates the input directories and the `/data1` mount **before**
starting. It executes:

1. `prepare` once to save `prepared/prepared_v4.pt` and the full selection
   manifest in `prepared/manifest_v4.json`.
2. `N=2`: train SAP and Shuffled, then evaluate both on held-out episodes.
3. `N=32`, then `N=128`, then `N=512`: repeat the matched procedure.
4. Create `summary-seed0/sample_size_summary_v4.json` and
   `sample_size_v4.png` / `.pdf`.

The runner prints a compact `[RESULT]` line for each scale and mask ratio:
SAP relative flow MSE, shuffled relative flow MSE, their paired difference,
and a paired episode-bootstrap 95% interval. Full numerical records are retained
under `eval/n{N}-seed0/per_episode_v4.json`.

All files are new to `OUT_ROOT`; existing complete groups are skipped on
rerun, while **incomplete output directories cause an explicit stop** rather
than automatic deletion or overwrite. For a second training seed, use a
different prepared directory/output root; the provided script is currently
designed as a matched *one-seed pilot*. Change `TRAIN_STEPS` and match all
groups if doing a separate longer-training study. An increased dataset size
gets fewer presentations per unique chunk at fixed 750 updates, so a later
750-versus-1500/3000 budget sensitivity comparison is necessary if the
512-chunk group appears undertrained.

**Scope:** The held-out episodes are withheld from the *new SAP adaptation*
only. They may have appeared during the pretraining of `pi05_libero_base`.
Thus V4 tests whether Stage-I **Action Prompt updates** learn reusable
sequence information on top of the same pretrained expert, not whether PI0.5
learns LIBERO dynamics from scratch. A downstream matched-budget
SAP/Shuffled/Full-mask/Direct-Dual comparison remains a separate Experiment C.

## V3: theory-grounded structure probes (no retraining)

The new **evaluation-only** script `two_sample_sap_structure_v3.py` adds four
different analyses of the *same* pretrained PI0.5 and the three existing
750-update Action Prompt checkpoints. It does **not** load the full dataset
into a Hugging Face Arrow cache and does **not** modify any checkpoints.

All comparisons reuse the same mask positions, Gaussian noise, two training
chunks (or explicitly selected held-out episodes), ODE budget, and random seed.
An `init` control uses the *saved initial prompt*; a `linear_interpolation`
control is used for all trajectory comparisons but cannot predict a flow field.

| Analysis name | Mathematical connection | Primary reported metric | When measured |
| --- | --- | --- | --- |
| `temporal_gain` | Conditional prediction under squared error | `temporal_predictive_gain = MSE(permuted context) - MSE(correct context)` | Random masks and long gap |
| `hankel` | Time-delay embedding / space-time POD / system identification | Distance between top-r singular subspace projectors; normalized SVD energy-spectrum error | Hidden *contiguous* region, continuous channels only |
| `flow_field` | PI0.5 action-only Flow Matching objective | Cosine alignment to the exact masked training velocity `z-a`; relative flow-MSE; context-order gain | Both masks, times `0.1,0.3,0.5,0.7,0.9` |
| `lag_correlation` | Autocorrelation / multi-joint dynamical coordination | Relative Frobenius error of lagged cross-joint covariance of action increments | Hidden *contiguous* region, continuous channels only |

### The exact interpretation of the theory

Let `Y` denote hidden actions, `C_ord` the ordered observed actions and
`C_bag` their unordered multiset. The conditional-expectation projection
theorem gives the **Bayes-optimal** squared-error risk decomposition

`R*(C_bag) - R*(C_ord) = E[ || E[Y|C_ord] - E[Y|C_bag] ||^2 ] >= 0`.

The implemented temporal-gain metric is **not** an estimator of that population
identity. It is a *paired intervention on a particular finite-capacity policy*:
the visible action vectors are randomly permuted across visible time positions,
with exactly the same hidden target and masked noise. This corruption can
create out-of-distribution input. Therefore, always compare SAP against
`init`, `shuffled`, `full_mask`, and interpolation, and report
`order_gain_vs_init` and `order_gain_vs_shuffled` (difference in differences).
Report both all-action and continuous-motion-only gains, to ensure that a
binary gripper state does not dominate the analysis.

Hankel analysis uses the **masked 20-step interval alone** (not all 50 steps
with 30 copied visible ground-truth values). For each 20x6 continuous-action
path, a window of 5 constructs a block time-delay Hankel matrix. All model
paths are centered/scaled with the *same reference statistics* (0.05 minimum
standard deviation). If `U_r` and `Uhat_r` contain top-r left singular
vectors, the subspace distance is

`||U_r U_r^T - Uhat_r Uhat_r^T||_F / sqrt(2r)`.

The extra SVD-spectrum metric compares normalized **squared** singular values.
An exactly static reference yields `null`, not a spurious structure score.
These are *finite-sequence diagnostic proxies*, not proof that a nonlinear
robotic action system is globally low rank or Koopman invariant. Only the
contiguous `long_gap` mode is eligible for these time-delay analyses: selecting
randomly scattered hidden steps would falsely treat unequal time intervals as
adjacent.

`flow_field` uses **the exact training flow convention** from
`PI05Pytorch.forward(ACTION_INPAINTING)`:
`x_t=(1-t)*a+t*z` for hidden timesteps; visible timesteps remain clamped to
clean actions. The target is `u_t=z-a`. Only hidden **original** action
dimensions are scored (not the internal zero-padded dimensions). Ground-truth
hidden actions are present in the *diagnostic* noising path, as in training,
but they are never supplied to the complete-ODE generator. This helps
distinguish poor local vector-field learning from ODE integration difficulties;
it does not by itself establish learned temporal structure.

`lag_correlation` constructs the velocity-increment sequence of each hidden
contiguous segment and compares the lag-0 through lag-5 cross-joint covariance
matrices to the reference. It uses shared reference-based velocity scaling.
The sample has few temporal observations, so its result is descriptive, not a
consistent estimate of a population autocorrelation function.

### Run sequentially on the existing trained prompts

First pull the updated **same branch** `sap-two-sample-structure` into the
separate existing checkout. Your old `sap/`, `shuffled/`, and `full_mask/`
directories are retained unchanged.

```bash
cd /root/autodl-tmp/lerobot060-sap-two-sample
git pull --ff-only origin sap-two-sample-structure

export OUT_ROOT=/root/autodl-tmp/wyn/chkpt/2601-lerobot/two-sample-sap
export DATASET_ROOT=/root/autodl-fs/datasets/libero

# Step 1: inexpensive Flow-Matching diagnosis, no full ODE solver.
CUDA_VISIBLE_DEVICES=0 python examples/analysis/two_sample_sap_structure_v3.py \
  --sap "$OUT_ROOT/sap" --shuffled "$OUT_ROOT/shuffled" \
  --full-mask "$OUT_ROOT/full_mask" \
  --analyses flow_field \
  --masked-steps 20 --repeats 4 \
  --output "$OUT_ROOT/analysis_v3_flow"

# Step 2: ordered-context risk + two dynamics tests, sharing full-ODE rollouts.
CUDA_VISIBLE_DEVICES=0 python examples/analysis/two_sample_sap_structure_v3.py \
  --sap "$OUT_ROOT/sap" --shuffled "$OUT_ROOT/shuffled" \
  --full-mask "$OUT_ROOT/full_mask" \
  --analyses temporal_gain,hankel,lag_correlation \
  --masked-steps 20 --ode-steps 32 --repeats 4 \
  --output "$OUT_ROOT/analysis_v3_dynamics"
```

Alternatively, run `--analyses all --output "$OUT_ROOT/analysis_v3_all"`
once to share the same probes and compare every metric together.
All output directories must be new/empty; the script refuses to overwrite a
previous analysis. The numerical results and **per-chunk raw observations**
are written to `metrics_v3.json` and `per_chunk_v3.json`. Depending on
`--analyses`, it also creates:

- `temporal_gain_v3.png`: ordered versus permuted context-risk gain;
- `hankel_v3.png`: reference/model mode-energy spectra and subspace/spectrum error;
- `flow_field_v3.png`: exact flow-target cosine versus flow time;
- `lag_correlation_v3.png`: lagged covariance magnitude and error.

To retain the original two-chunk training setup while measuring generalization,
specify **independent episodes** using `--eval-chunks` and `--dataset-root`:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/analysis/two_sample_sap_structure_v3.py \
  --sap "$OUT_ROOT/sap" --shuffled "$OUT_ROOT/shuffled" \
  --full-mask "$OUT_ROOT/full_mask" \
  --dataset-root "$DATASET_ROOT" --eval-chunks 2:10,3:10,4:10,5:10 \
  --analyses all --masked-steps 20 --ode-steps 32 --repeats 4 \
  --output "$OUT_ROOT/analysis_v3_heldout"
```

The episode selectors are examples: verify that the selected episodes have
50 unpadded consecutive steps. The code **rejects any training episode** in
held-out evaluation. Repeating masks on the same trajectory is not an
independent statistical seed, and using only two training chunks cannot
establish a general structure-learning claim. Low error on these metrics is
meaningful only if SAP separates from its shuffled and untrained controls.

Relevant theory and primary studies:

- Conditional expectation and squared-error projection:
  https://ee.stanford.edu/~gray/LPCIP-sig-029-ebook.pdf
- Frame and Towne, *Space-time POD and the Hankel Matrix*, 2023:
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0289637
- Han et al., *On the Utility of Koopman Operator Theory in Learning Dexterous
  Manipulation Skills*, CoRL 2023: https://proceedings.mlr.press/v229/han23a.html
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023:
  https://arxiv.org/abs/2210.02747
