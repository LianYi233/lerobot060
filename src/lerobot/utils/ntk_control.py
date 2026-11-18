"""Lightweight checkpoint validation for the uninterrupted prompt-learning control."""

import json
from pathlib import Path

from lerobot.utils.ntk_protocol import COMPARABLE_FIELDS


PROMPT_ONLY_STEPS = (0, 750, 1000, 4000)
PROMPT_ONLY_NAMES = ("before", "step_750", "step_1000", "final")
_TRAINING_FIELDS = (
    "cabo_enabled",
    "cabo_prompt_update_ratio",
    "clip_action_head_by_vlm",
    "freeze_vision_encoder",
    "train_expert_only",
)


def _read_json(path):
    return json.loads(path.read_text())


def resolve_prompt_only_stages(args):
    """Resolve exact cumulative snapshots from one 4000-update, full-flow training run.

    ``through_step`` permits analyzing available snapshots before training finishes.
    Checkpoint contents are never substituted with nearby updates or staged weights.
    """
    through_step = getattr(args, "through_step", 4000)
    if through_step not in PROMPT_ONLY_STEPS:
        raise ValueError(f"--through-step must be one of {PROMPT_ONLY_STEPS}")
    if not args.run_dir:
        raise ValueError("--run-dir is required for the prompt-only control")
    run = Path(args.run_dir).expanduser().resolve()
    stages = []
    weight_paths = []
    for name, step in zip(PROMPT_ONLY_NAMES, PROMPT_ONLY_STEPS, strict=True):
        if step > through_step:
            break
        path = (run / f"checkpoints/{step:06d}/pretrained_model").resolve()
        for filename in ("config.json", "train_config.json", "model.safetensors", "policy_preprocessor.json"):
            if not (path / filename).is_file():
                raise FileNotFoundError(
                    f"{name}: missing {path / filename}. Save exact 0/750/1000/4000 snapshots with "
                    "examples/training/train_pi05_prompt_ntk_control.sh; use --through-step for a "
                    "partially completed run."
                )
        step_file = path.parent / "training_state/training_step.json"
        if not step_file.is_file():
            raise FileNotFoundError(f"Cannot verify checkpoint step: {step_file}")
        saved_step = _read_json(step_file).get("step")
        if type(saved_step) is not int or saved_step != step:
            raise ValueError(f"{name}: checkpoint has {saved_step} local updates, expected {step}")
        config = _read_json(path / "config.json")
        train_config = _read_json(path / "train_config.json")
        if config.get("training_stage") != "flow":
            raise ValueError(f"{name}: prompt-only control requires a flow checkpoint")
        for field in ("next_action_pretrain_steps", "next_action_bridge_steps"):
            if config.get(field) != 0:
                raise ValueError(f"{name}: prompt-only control requires {field}=0")
        for field in ("num_vlm_prompt_tokens", "num_prompt_tokens"):
            count = config.get(field)
            if type(count) is not int or count <= 0:
                raise ValueError(f"{name}: prompt-only control requires positive {field}")
        if train_config.get("steps") != 4000:
            raise ValueError(f"{name}: prompt-only control requires train_config.steps=4000")
        seed = train_config.get("seed")
        if type(seed) is not int or seed < 0:
            raise ValueError(f"{name}: train_config must record a nonnegative integer seed")
        if type(config.get("cabo_enabled")) is not bool:
            raise ValueError(f"{name}: config must record the cabo_enabled setting")
        weights = (path / "model.safetensors").resolve()
        if any(weights.samefile(previous) for previous in weight_paths):
            raise ValueError("The selected checkpoints must contain distinct model files")
        weight_paths.append(weights)
        stages.append(
            {
                "name": name,
                "local_step": step,
                "total_step": step,
                "checkpoint": str(path),
                "config": config,
                "training_seed": seed,
            }
        )
    for field in (*COMPARABLE_FIELDS, *_TRAINING_FIELDS):
        if any(stage["config"].get(field) != stages[0]["config"].get(field) for stage in stages[1:]):
            raise ValueError(f"Incomparable prompt-only checkpoint configuration: {field}")
    if any(stage["training_seed"] != stages[0]["training_seed"] for stage in stages[1:]):
        raise ValueError("Prompt-only checkpoints must have the same training seed")
    return stages
