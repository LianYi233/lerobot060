"""CPU contract tests for the shared-prefix AdamW/scheduler 2x2 ablation."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("datasets")
pytest.importorskip("transformers")

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.optim.cabo import (  # noqa: E402
    CABO_ACTION_PROMPT_GROUP,
    CABO_GROUP_NAME,
    CABO_VLM_PROMPT_GROUP,
)
from lerobot.optim.optimizers import save_optimizer_state  # noqa: E402
from lerobot.optim.schedulers import save_scheduler_state  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.scripts import lerobot_train as training  # noqa: E402
from lerobot.utils.constants import PRETRAINED_MODEL_DIR, TRAINING_STATE_DIR  # noqa: E402


class _SingleRankAccelerator:
    num_processes = 1


def _make_optimizer(params):
    return torch.optim.AdamW(
        [
            {"params": [params[0]], CABO_GROUP_NAME: CABO_VLM_PROMPT_GROUP},
            {"params": [params[1]], CABO_GROUP_NAME: CABO_ACTION_PROMPT_GROUP},
        ],
        lr=2.5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )


def _fixture(tmp_path: Path, moments: str, scheduler: str):
    policy = PI05Config(
        device="cpu",
        push_to_hub=False,
        stage3_adamw_moments=moments,
        stage3_lr_scheduler=scheduler,
    )
    prefix_checkpoint = tmp_path / "prefix" / "checkpoints" / "001000"
    policy.pretrained_path = prefix_checkpoint / PRETRAINED_MODEL_DIR
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="libero", root=tmp_path / "data"),
        policy=policy,
        output_dir=tmp_path / "flow",
        job_name="transition-test",
        steps=3_000,
        batch_size=32,
        seed=3,
        pi05_stage3_transfer_checkpoint=prefix_checkpoint,
        optimizer=policy.get_optimizer_preset(),
        scheduler=policy.get_scheduler_preset(),
    )
    stage1_cfg = training._make_pi05_next_action_pretraining_config(cfg)
    (prefix_checkpoint / PRETRAINED_MODEL_DIR).mkdir(parents=True)
    (prefix_checkpoint / TRAINING_STATE_DIR).mkdir()
    cfg.dataset.root.mkdir()
    with (prefix_checkpoint / PRETRAINED_MODEL_DIR / "train_config.json").open("w") as handle:
        json.dump(stage1_cfg.to_dict(), handle, default=str)

    source_params = [torch.nn.Parameter(torch.ones(2, 3)), torch.nn.Parameter(torch.ones(2, 3))]
    source_optimizer = _make_optimizer(source_params)
    source_scheduler = cfg.scheduler.build(source_optimizer, 1_000)
    for step in range(1_000):
        for index, parameter in enumerate(source_params):
            parameter.grad = torch.full_like(parameter, 0.1) if index == 1 or step >= 750 else None
        source_optimizer.step()
        source_scheduler.step()
        source_optimizer.zero_grad()
    state_dir = prefix_checkpoint / TRAINING_STATE_DIR
    save_optimizer_state(source_optimizer, state_dir)
    save_scheduler_state(source_scheduler, state_dir)
    return cfg, source_optimizer, source_scheduler


@pytest.mark.parametrize(
    ("moments", "scheduler"),
    [
        ("carry", "carry"),
        ("reset", "carry"),
        ("carry", "restart"),
    ],
)
def test_stage3_independent_moment_scheduler_transfer(tmp_path, moments, scheduler):
    cfg, source_optimizer, source_scheduler = _fixture(tmp_path, moments, scheduler)
    training._validate_pi05_stage3_source(cfg)

    new_params = [torch.nn.Parameter(torch.zeros(2, 3)), torch.nn.Parameter(torch.zeros(2, 3))]
    new_optimizer = _make_optimizer(new_params)
    new_scheduler = cfg.scheduler.build(new_optimizer, 3_000)
    fresh_initial_lr = new_optimizer.param_groups[0]["lr"]
    optimizer, lr_scheduler = training._restore_pi05_stage3_transition(
        cfg, new_optimizer, new_scheduler, _SingleRankAccelerator()
    )

    for source_group, new_group in zip(source_optimizer.param_groups, optimizer.param_groups, strict=True):
        source_param = source_group["params"][0]
        new_param = new_group["params"][0]
        if moments == "carry":
            assert torch.equal(optimizer.state[new_param]["exp_avg"], source_optimizer.state[source_param]["exp_avg"])
            assert torch.equal(optimizer.state[new_param]["exp_avg_sq"], source_optimizer.state[source_param]["exp_avg_sq"])
            assert optimizer.state[new_param]["step"].item() == source_optimizer.state[source_param]["step"].item()
        else:
            assert not optimizer.state[new_param]

        if scheduler == "carry":
            assert new_group["lr"] == pytest.approx(source_scheduler.get_last_lr()[0])
            assert lr_scheduler.last_epoch == 1_000
        else:
            assert new_group["lr"] == pytest.approx(fresh_initial_lr)
            assert lr_scheduler.last_epoch == 0


def test_stage3_rejects_mismatched_source_seed(tmp_path):
    cfg, _, _ = _fixture(tmp_path, "carry", "restart")
    cfg.seed = 4
    with pytest.raises(ValueError, match="seed mismatch"):
        training._validate_pi05_stage3_source(cfg)
