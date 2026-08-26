# PI0.5 Cross-Module Task Complexity Gap

## Motivation

The opening experiment is designed to test a measurable hypothesis rather than the qualitative statement that the VLM is pretrained while the action expert is not:

> Under the same downstream VLA objective and the same robot demonstrations, the pretrained VLM and the action pathway occupy task-conditioned tangent spaces of substantially different effective dimensionality.

We call this discrepancy the **Cross-Module Task Complexity Gap**.

## Practical kernel used in the diagnostic

For sample `i` and module `m` (`VLM` or `Action`), let `l_i` be the per-sample observation-conditioned PI0.5 flow loss and define the gradient feature

`g_i^m = d l_i / d theta_m`.

The diagnostic forms the module-wise Gram matrix

`K_m[i,j] = <g_i^m, g_j^m>`.

For a multi-billion-parameter model the dense gradients are too large to store for all samples, so each gradient is mapped through the same deterministic CountSketch before constructing the Gram matrix. Pairwise inner products are therefore estimated in a fixed low-dimensional sketch space.

This matrix should be described in the paper as a **task-conditioned empirical tangent-kernel proxy** (or loss-gradient/Fisher kernel), not as the exact vector-output NTK.

## Primary metrics

From the eigenvalues of each module-wise kernel, report:

1. **Spectral effective rank** (entropy rank). This is the main scale-invariant task-effective dimension measure.
2. **Participation ratio** as a second effective-dimension estimate.
3. **d90**, the minimum number of eigenmodes required to explain 90% of the kernel spectral mass.
4. **Action/VLM gap ratios** for the above quantities.

The raw kernel trace is also saved, but it is scale-sensitive and should be treated as an optimization-strength diagnostic rather than the main complexity measure.

## Minimal opening experiment

Use the unadapted PI0.5 LIBERO checkpoint and a fixed LIBERO subset. The two modules are probed with exactly the same observation-conditioned flow objective and exactly the same samples.

Recommended settings for a pilot:

- samples: 8-16
- sketch dimension: 256
- seeds: 3
- one GPU
- no `torch.compile`
- no weight updates

For the final paper figure, increase to 16-32 samples if memory/runtime permits and average across 3-5 fixed sample subsets or seeds.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m lerobot.scripts.lerobot_ntk_gap \
  --policy-path=/data/models/lerobot/pi05_libero_base \
  --dataset-root=/data/datasets/libero \
  --dataset-repo-id=libero \
  --video-backend=pyav \
  --batch-size=8 \
  --num-samples=8 \
  --sketch-dim=256 \
  --seed=0 \
  --output=/data/wyn/ntk_gap/base_seed0.json
```

Repeat with `--seed=1` and `--seed=2`.

## Figure 1 for the paper

A compact two-panel motivation figure is sufficient.

**Panel (a): normalized kernel spectrum**

Plot normalized eigenvalues for `VLM` and `Action Expert` on the same rank axis. A more distributed action spectrum indicates that more independent tangent modes participate in the downstream robot objective.

**Panel (b): task-effective dimension**

Show either effective rank or d90 as two bars (`VLM`, `Action Expert`) and annotate the Action/VLM ratio.

The intended observation is:

`Task-effective dimension(Action) >> Task-effective dimension(VLM)`

This supports the claim that direct joint VLA fine-tuning couples modules with different downstream adaptation complexity.

## How this motivates the three stages

If the initial gap is observed, the method can be presented as **progressive complexity alignment**:

1. **Action Prior Acquisition — Compress.** Learn trajectory structure first, reducing the action pathway's unresolved task complexity.
2. **Conditioning Bridge — Align.** Keep the VLM fixed while the action expert learns to consume its semantic representation, aligning the action learning subspace with a stationary semantic subspace.
3. **Balanced Joint Adaptation — Balance.** Only after the action pathway has been prepared do both modules co-adapt, while controlling their relative update dynamics.

This gives the paper a single storyline:

**Measure the complexity gap -> compress action complexity -> align conditioning -> balance joint adaptation.**

## Stronger follow-up experiment

For a more complete method analysis, run the same diagnostic at three checkpoints:

- Base PI0.5 initialization
- After Action Prior Acquisition
- After Conditioning Bridge

The strongest result would be a progressive reduction in the Action/VLM task-effective-dimension ratio before joint fine-tuning. This is not required for the opening motivation figure, but it directly connects the metric to the mechanism of BridgeVLA.
