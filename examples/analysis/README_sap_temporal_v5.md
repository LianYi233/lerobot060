# SAP V5: mathematical temporal-action evidence (local A100)

This is a **new**, non-destructive extension of the existing V4
`sap-two-sample-structure` branch. It does **not** overwrite `sap-structure-v4`.
It uses actual frozen PI0.5 `predict_inpainting_velocity`, actual Stage-I
prompt weights, and **complete noise-to-action Euler integration** for
trajectory evaluation. A synthetic experiment reuses V4's unchanged
`PI05TrainingObjective.ACTION_INPAINTING` prompt trainer.

The claim under examination is **predictive temporal structure in action
trajectories**, not identification of the robot/environment state transition
model from actions alone.

## 1. Run in the agreed P0 → P1 → P2 → P3 → P4 order

On the **local A100**, after pulling `sap-two-sample-structure` into your
existing repository checkout, install the checkout in the *existing working*
LeRobot Python environment, and launch:

```bash
cd /home/wyn/VLAA/lerobot060
git status --short  # Check for local edits before changing branches.
git switch sap-two-sample-structure
git pull --ff-only origin sap-two-sample-structure

export DATASET_ROOT=/data/datasets/libero
export BASE_PI05=/data/models/lerobot/pi05_libero_base/
export V4_ROOT=/data1/wyn/chkpt/2601-lerobot/sap-structure-v4
export OUT_ROOT=/data1/wyn/chkpt/2601-lerobot/sap-temporal-v5
export GPU_ID=0

bash examples/analysis/run_sap_temporal_v5_local_a100.sh
```

The script expects completed V4 prompts at
`$V4_ROOT/train/n128-sap-seed0/` and
`$V4_ROOT/train/n128-shuffled-seed0/`. Set `REAL_N=32` or `REAL_N=512`
to use a different **existing** pair. The default is `REAL_N=128`.

All **new** artifacts and HF/Torch/matplotlib/temp caches are under
`/data1`. LIBERO and `pi05_libero_base` are **read only** under `/data`.
Existing finished V5 stages are skipped. An incomplete output directory
causes an explicit error rather than silent deletion or overwrite.

If you want to review the results after each stage (recommended), run:

```bash
PHASE=P0 bash examples/analysis/run_sap_temporal_v5_local_a100.sh
PHASE=P1 bash examples/analysis/run_sap_temporal_v5_local_a100.sh
PHASE=P2 bash examples/analysis/run_sap_temporal_v5_local_a100.sh
PHASE=P3 bash examples/analysis/run_sap_temporal_v5_local_a100.sh
PHASE=P4 bash examples/analysis/run_sap_temporal_v5_local_a100.sh
```

The five stage labels include the **P0 prerequisite check** plus four
substantive follow-up experiments. `PHASE=all` runs them in order.

## 2. What is mathematically being tested?

For squared risk and an ordered context `C`, its unordered multiset `B`,
and fixed auxiliary inputs `X`, the conditional-expectation projection
identity states:

`R*(X,B) - R*(X,C) = E || E[Y|X,C] - E[Y|X,B] ||² >= 0`.

This is a statement about **Bayes-optimal** risks. None of our finite-capacity
interventions automatically estimates the right-hand side. Our experiments
instead ask whether ordered information is predictively useful in real data,
whether SAP takes advantage of it, and whether its learned features support
future-action readout.

The flow diagnostic uses the original `x_t=(1-t)*a+t*z` target `u_t=z-a`.
Its `t` is **noise interpolation time**, NOT physical action time. In a
flow-field diagnostic the masked hidden action enters the noisy target by
definition; this can make flow MSE deceptively easy, so the complete-ODE
rollouts start at `t=1` with **pure Gaussian noise** on every hidden channel.

### P0 — CPU-only ordered-vs-unordered predictive information preflight

**Input:** all V4 prepared train windows and disjoint test episodes.

**Test:** predicting next `5/10/20` actions from the first 20 using two
capacity-matched ridge regressors:

- **Ordered:** all 20 action vectors in their original time positions.
- **Unordered:** the *same whole 20 vectors* sorted lexicographically,
  retaining each multi-joint vector and the same feature dimension,
  but removing their temporal ordering.

The regularization strength is selected on a separate validation subset;
numbers are measured on V4's held-out episodes. A positive paired
`bag_minus_ordered` suggests an available **linear predictive benefit** of
time order. A negative/near-zero result indicates that these particular
action windows may not be a discriminative temporal benchmark; it is NOT a
proof that real actions contain no dynamical information.

**File:** `P0-data-check/data_order_preflight_v5.json`.

### P1 — ordered-context interventions plus true future/contiguous gaps

**Models:** `init`, existing V4 SAP, existing V4 Shuffled; exact same noise,
mask, held-out episode IDs and inference steps.

**Masks** (all 50-step chunks):

| Mode | Visible context | Completely hidden | Scored |
| --- | --- | --- | --- |
| `future5` | `0..19` | `20..49` | `20..24` |
| `future10` | `0..19` | `20..49` | `20..29` |
| `future20` | `0..19` | `20..49` | `20..39` |
| `middle10` | all except `20..29` | `20..29` | `20..29` |
| `middle20` | all except `15..34` | `15..34` | `15..34` |
| `random20` | 30 randomly visible | 20 randomly hidden | those 20 |

The future tests deliberately hide `20..49` even when scoring only five
future steps: **the rest of the future MUST NOT remain visible**.

For each model, evaluate the **same masked target** with correctly ordered
visible context and an independent permutation of the **visible whole action
vectors**. Record high-noise masked flow MSE and, in ordered context, the
masked-action MSE of complete Euler ODE generation. The optional
`--permuted-ode` flag also generates a fully sampled corrupted-context path.

Interpretation uses *paired episode-level* quantities:

1. `Init MSE − SAP MSE` (positive means SAP improved on Init).
2. `Shuffled MSE − SAP MSE` (positive means SAP better than Shuffled).
3. `[MSE_perm−MSE_order]_SAP − [MSE_perm−MSE_order]_Init` (additional
   order sensitivity; positive **alone** is NOT sufficient because
   permutation can produce out-of-distribution brittleness).

Report both correct-order absolute risk and intervention sensitivity.
Bootstrap uncertainty comes from independent **episodes**. One training seed
does not yield a training-seed confidence interval; ideally repeat SAP and
Shuffled with 3 training seeds.

**Files:** `P1-blocked/metrics_temporal_v5.json`,
`per_episode_flow_temporal_v5.json`, `per_episode_ode_temporal_v5.json`,
`rollouts_temporal_v5.npz`.

**Speed:** P1 is the first GPU-heavy stage. The default is 200 test episodes,
3 mask modes, 1 repeat, batch 4, 16 Euler steps. For a quick pilot use
`PHASE=P1 TEST_LIMIT=32 ODE_STEPS=8`; use a **separate OUT_ROOT** for that
pilot because complete outputs are not overwritten.

### P2 — ground-truth AR(2) modal-dynamics recovery

Generate normalized 7-D action sequences with six continuous channels
following known stable second-order processes and a binary last channel:

`a[k+1] = 2*r*cos(omega)*a[k] - r²*a[k-1] + epsilon[k]`.

Three independently initialized regimes have different `(r,omega)` values.
Because these parameters determine the **ground-truth poles**
`r*exp(± i*omega)`, the comparison has a known mathematical target.

- Prepare **256** unique training and **120** disjoint test sequences.
- Train SAP and per-update Shuffled with the **original V4 Stage-I trainer**,
  frozen backbones, identical `2000` steps and batch `4`.
- On synthetic test, show only the first 20 true actions; hide all 30 future
  actions, integrate the model from noise, and score the first 20 generated
  future actions.
- Fit AR(2) coefficients to **only the generated hidden region** and compare
  against known `(2*r*cos(omega), -r²)`. Report paired episode-level SAP-vs-Init
  and SAP-vs-Shuffled differences with bootstrap uncertainty, as well as the
  estimated pole radius `r` and oscillation frequency `omega` when real complex
  poles are mathematically valid. Report analogous fits on real synthetic
  test paths as the attainable finite-noise baseline.

These results show recovery of an action temporal process, **not** physical
state-transition identification from actions alone. The fixed pretrained
checkpoint may be out of distribution on synthetic sequences; failure is
an informative limitation, not an argument for excluding the result.

**Files:** `P2-synthetic/prepared`, `P2-synthetic/train`,
`P2-synthetic/eval`, `P2-synthetic/recovery/ar2_recovery_v5.json`.

### P3 — frozen Action Expert predictive-state probe

For each held-out episode, feed the true first 20 actions to the expert and
**pure noise** in the remaining 30 positions. Using a forward-pre-hook on
the *actual* `model.action_out_proj`, capture the expert's final hidden
state at the **last visible** action token.

With all model weights and prompts frozen, fit matched dual ridge readouts
for future action offsets `1, 5, 10` on a **60/20/20 split entirely inside
the SAP-held-out episode pool**. Predict only continuous channels and report
test MSE and variance-normalized `R²`.

This estimates **linearly accessible** predictive information, not true
mutual information; different feature condition numbers may affect probing.

**File:** `P3-probe/future_probe_v5.json`.

### P4 — time-augmented path signature product-kernel MMD

From P1's saved full-ODE trajectories, compute the piecewise-linear **level
1 and 2** path signatures on each full 50-step path plus normalized time.
Use a Gaussian kernel on those features multiplied by a separate Gaussian
kernel on the observed context (masked positions set to zero). Compare the
joint distributions of `(visible context, full trajectory)` using a biased
MMD squared, with identical episodes and conditions for all models.

This is a **finite, truncated signature proxy**, NOT the characteristic
robust signature kernel of the full Chevyrev–Oberhauser theory. A positive
result supports better **trajectory distribution matching** and must be
interpreted alongside P1's direct conditional forecast loss.

**File:** `P4-signature/signature_mmd_v5.json`.

### Appendix figure (automatically generated after P4 when all phases exist)

`figures/sap_temporal_evidence_v5.png` and `.pdf` contain six panels:
P0 order information, P1 high-noise flow loss, P1 full-ODE rollout loss,
P2 synthetic coefficient recovery, P3 predictive probe, and P4 finite path
signature MMD. The plot deliberately displays **all three variants**, even
when an experiment does not favor SAP. A JSON manifest hashes each input.

To regenerate in a new directory without rewriting any existing results:

```bash
python examples/analysis/plot_sap_temporal_v5.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/figures-manual-1"
```

## 3. Manual/advanced commands

The main script is `examples/analysis/sap_temporal_evidence_v5.py`:

```bash
python examples/analysis/sap_temporal_evidence_v5.py data-check --help
python examples/analysis/sap_temporal_evidence_v5.py blocked-eval --help
python examples/analysis/sap_temporal_evidence_v5.py synthetic-prepare --help
python examples/analysis/sap_temporal_evidence_v5.py synthetic-score --help
python examples/analysis/sap_temporal_evidence_v5.py probe --help
python examples/analysis/sap_temporal_evidence_v5.py signature-mmd --help
```

Test pure numerical logic without GPU:

```bash
python -m unittest tests.scripts.test_pi05_sap_temporal_evidence_v5 \
                   tests.scripts.test_pi05_two_sample_structure_v4 -v
```

## 4. Reproducibility and limitations

- V4 splits are episode-disjoint for **new SAP training**, but may overlap
  with `pi05_libero_base` *pretraining*. No unseen-task claim is made.
- Flow `t` is noise interpolation time; action index `k` is physical sequence
  time. Do not confuse them when writing the paper.
- P1's permuted-context score is an **intervention**, not a consistent
  estimator of the Bayes-risk information gap, and high sensitivity may
  simply reflect out-of-distribution input.
- On the real data, 200 adaptation-heldout episodes still do not substitute
  for multiple independent training seeds.
- P2 is mathematically identifiable only under its specified synthetic
  generating model, not for the full robot/environment system.
- The scientific stopping rule is to **report negative results** if SAP does
  not outperform Shuffled and/or Init on temporally discriminative metrics.
  These experiments are not expected or guaranteed to prove a favorable result.

Primary sources: [Flow Matching for Generative Modeling (ICLR 2023)](https://mlanthology.org/iclr/2023/lipman2023iclr-flow/),
[Predictive Representations of State (NeurIPS 2001)](https://proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html),
[Error Bounds for Flow Matching Methods (TMLR 2024)](https://arxiv.org/abs/2305.16860),
[Signature Moments to Characterize Laws of Stochastic Processes (JMLR 2022)](https://jmlr.org/papers/v23/20-1466.html),
and [Transformers Represent Belief State Geometry (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html).
