# π₀.₅ (pi05)

This repository contains the Hugging Face port of **π₀.₅**, adapted from [OpenPI](https://github.com/Physical-Intelligence/openpi) by the Physical Intelligence.
It is designed as a **Vision-Language-Action model with open-world generalization**.

---

## Model Overview

| Feature              | π₀                                                     | π₀.₅                                      |
| -------------------- | ------------------------------------------------------ | ----------------------------------------- |
| Time Conditioning    | Concatenates time with actions via `action_time_mlp_*` | Uses `time_mlp_*` for AdaRMS conditioning |
| AdaRMS               | Not used                                               | Used in action expert                     |
| Tokenizer Length     | 48 tokens                                              | 200 tokens                                |
| Discrete State Input | False (Uses `state_proj` layer)                        | True                                      |
| Parameter Count      | Higher (includes state embedding)                      | Lower (no state embedding)                |

---

## Training Defaults

This repository's variant uses `[VLM tokens] [prompt tokens] [action tokens]`, with 16 trainable
prompt embeddings by default (`num_prompt_tokens=16`, `prompt_init_std=0.02`). Prompts use the action
expert's embedding width. VLM tokens see the VLM prefix; prompts see the prefix and all prompts;
actions see all three blocks. Set `num_prompt_tokens=0` for an ablation without prompts.

The entire VLM, including the vision encoder, stays frozen in every training stage.
`freeze_vision_encoder=true` and `train_expert_only=true` are enforced even when loading older
configurations. Prompts and the complete action path train during action-only Stage 1, the
observation-conditioned bridge, and Stage 2. At inference, the VLM prefix is cached and prompts are
processed with actions at each denoising step; the action chunk length is unchanged.

Fresh configurations use a unified AdamW learning rate of `2.5e-5` for prompts and action-side
parameters, followed by cosine decay to a `1e-5` floor. Global gradient clipping is disabled
(`optimizer_grad_clip_norm=0.0`). VLM-relative gradient clipping (`clip_action_head_by_vlm`) must
remain `false` for flow training.

CABO is optional and uses the prompt as its update reference. Enable it with
`--policy.cabo_enabled=true --policy.cabo_prompt_update_ratio=1.0`. It keeps the prompt's full
scheduled learning rate and attenuates the current step's learning rate for the action expert and
the projections/timestep MLP so that each group's relative learning update is no larger than the
prompt's. The relative update is `||delta_theta|| / ||theta||`, including AdamW preconditioning and
the learning rate, but excluding weight decay. A ratio above `1.0` increases the required prompt
advantage. CABO applies from the first update in every stage, requires nonzero prompt tokens, and
keeps the complete VLM frozen. Legacy VLM-based CABO ratios, EMA, warmup, and floor settings are
loadable but inactive; the ordinary learning-rate scheduler is unchanged. CABO is disabled by default.

Loading a base or older full-model checkpoint without prompt weights initializes new prompts. Loading a
checkpoint saved by this variant preserves its learned prompts. To migrate an older training run,
initialize a new run from its model weights: the old optimizer state may not support direct resume
after adding prompts and freezing the VLM. Enabling prompt-based CABO requires named `prompt`,
`action_expert`, and `action_projection` optimizer groups, so changing from a non-CABO or older
VLM-based CABO run also requires a fresh optimizer initialized from saved model weights. See the
[training guide](./pi05.mdx) for the two-stage curriculum and checkpoint details.

Legacy PEFT adapters without saved `prompt_tokens` require `num_prompt_tokens=0`; for prompt
training, use a prompt-enabled adapter or initialize from a full-model checkpoint.

---

## Relative Actions

π₀.₅ supports training with **relative actions**, where the model learns relative offsets
from the current robot state instead of absolute joint positions. This mirrors the
relative-action transform in OpenPI (`DeltaActions`) and can improve performance.

### How it works

1. **During preprocessing**, absolute actions are converted to relative offsets:
   `relative = action - state` (for selected joints).
2. The relative actions are normalized using statistics computed from the relative distribution.
3. **During postprocessing**, predicted relative actions are converted back to absolute:
   `absolute = relative + state`.

Joints listed in `relative_exclude_joints` (e.g., gripper) are kept absolute.

### Configuration

| Parameter                 | Type        | Default       | Description                                                      |
| ------------------------- | ----------- | ------------- | ---------------------------------------------------------------- |
| `use_relative_actions`    | `bool`      | `False`       | Enable relative-action training                                  |
| `relative_exclude_joints` | `list[str]` | `["gripper"]` | Joint names to keep absolute (matched by substring)              |
| `action_feature_names`    | `list[str]` | `None`        | Auto-populated from dataset metadata at runtime by `make_policy` |

### Training example

```bash
uv run lerobot-train \
  --policy.type=pi05 \
  --dataset.repo_id=your_org/your_dataset \
  --policy.num_prompt_tokens=16 \
  --policy.train_expert_only=true \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]'
```

When `use_relative_actions=true`, the training script automatically:

- Computes relative action statistics from the dataset (sampled chunk-level relative actions)
- Replaces the standard action stats with relative stats for normalization
- Broadcasts these stats across all ranks in distributed training

---

## Citation

If you use this work, please cite both **OpenPI** and the π₀.₅ paper:

```bibtex
@misc{openpi2024,
  author       = {Physical Intelligence Lab},
  title        = {OpenPI: PyTorch Implementation of π0 and π0.5 Policies},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Physical-Intelligence/openpi}},
  license      = {Apache-2.0}
}

@misc{intelligence2025pi05visionlanguageactionmodelopenworld,
  title        = {π₀.₅: a Vision-Language-Action Model with Open-World Generalization},
  author       = {Physical Intelligence and Kevin Black and Noah Brown and James Darpinian and Karan Dhabalia and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Manuel Y. Galliker and Dibya Ghosh and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Devin LeBlanc and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Allen Z. Ren and Lucy Xiaoyang Shi and Laura Smith and Jost Tobias Springenberg and Kyle Stachowicz and James Tanner and Quan Vuong and Homer Walke and Anna Walling and Haohuan Wang and Lili Yu and Ury Zhilinsky},
  year         = {2025},
  eprint       = {2504.16054},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2504.16054},
}
```

---

## License

This port follows the **Apache 2.0 License**, consistent with the original [OpenPI repository](https://github.com/Physical-Intelligence/openpi).
