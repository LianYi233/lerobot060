"""CPU-only safety checks for the opt-in Piper partial-ft profiles.

No robot, GPU, real model weights, or network are required.
Run: python -m pytest tests/scripts/test_piper_partial_ft.py -q
"""
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "examples/training/train_piper_partial_ft.sh"
TASK = "2-remove_the_cuboid_from_blue_plate"


def make_config(vlm_layers, *, full_head=False):
    return PI05Config(
        partial_ft_enabled=True,
        train_vlm_last_n_layers=vlm_layers,
        train_action_expert_last_n_layers=0 if full_head else 4,
        train_action_projections=True,
        cabo_enabled=False,
        use_peft=False,
        train_action_head=full_head,
        next_action_pretrain_steps=0,
        next_action_bridge_steps=0,
    )


@pytest.mark.parametrize("last", [0, 2, 4])
def test_partial_ft_config_keeps_vision_frozen_and_vlm_grad_path(last):
    cfg = make_config(last)
    assert cfg.freeze_vision_encoder
    assert cfg.train_expert_only == (last == 0)
    assert cfg.separate_frozen_observations == (last == 0)
    assert cfg.partial_ft_enabled


def test_vlm_trainability_cannot_be_enabled_without_opt_in():
    with pytest.raises(ValueError, match="partial_ft_enabled"):
        PI05Config(train_vlm_last_n_layers=2)
    with pytest.raises(ValueError, match="cabo_enabled=False"):
        PI05Config(
            partial_ft_enabled=True,
            train_action_projections=True,
            train_action_expert_last_n_layers=4,
        )


@pytest.mark.parametrize("last", [0, 2, 4])
def test_full_action_head_allows_vlm_adaptation_but_not_last_n_mode(last):
    cfg = make_config(last, full_head=True)
    assert cfg.train_action_head is True
    assert cfg.train_action_expert_last_n_layers == 0
    assert cfg.train_action_projections is True
    assert cfg.train_expert_only == (last == 0)
    assert cfg.separate_frozen_observations == (last == 0)
    with pytest.raises(ValueError, match="train_action_expert_last_n_layers=0"):
        PI05Config(
            partial_ft_enabled=True,
            train_action_head=True,
            train_action_expert_last_n_layers=4,
            train_action_projections=True,
            cabo_enabled=False,
            next_action_pretrain_steps=0,
            next_action_bridge_steps=0,
        )


class FakeCore(torch.nn.Module):
    """Small stand-in for verifying exact optimizer grouping (NOT gradient quality)."""

    def __init__(self, last, *, full_head=False):
        super().__init__()
        self.vlm_prompt_tokens = torch.nn.Embedding(16, 2)
        self.prompt_tokens = torch.nn.Embedding(16, 2)
        self.action_in_proj = torch.nn.Linear(2, 2)
        self.action_out_proj = torch.nn.Linear(2, 2)
        self.expert = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
        self.expert_norm = torch.nn.Linear(2, 2)
        self.time_mlp_in = torch.nn.Linear(2, 2)
        self.time_mlp_out = torch.nn.Linear(2, 2)
        self.vlm = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
        self.last = last
        self.full_head = full_head
        for module in (self.expert_norm, self.time_mlp_in, self.time_mlp_out):
            module.requires_grad_(full_head)
        for i, module in enumerate(self.vlm):
            module.requires_grad_(last > 0 and i >= 4 - last)

    def _freeze_vlm(self):
        pass

    def trainable_expert_blocks(self):
        return list(self.expert)

    def trainable_action_core_modules(self):
        return [self.expert, self.expert_norm, self.time_mlp_in, self.time_mlp_out]

    def trainable_vlm_blocks(self):
        return list(self.vlm[-self.last:]) if self.last else []


@pytest.mark.parametrize("last", [0, 2, 4])
@pytest.mark.parametrize("full_head", [False, True])
def test_optimizer_groups_exhaustive_disjoint_and_scaled(last, full_head):
    model = FakeCore(last, full_head=full_head)
    config = make_config(last, full_head=full_head)
    config.optimizer_lr = 2.5e-5
    config.action_projection_lr_scale = .4
    config.action_expert_lr_scale = .4
    config.vlm_lr_scale = .04
    policy = SimpleNamespace(model=model, config=config, parameters=model.parameters)
    groups = PI05Policy.get_optim_params(policy)
    names = [group["name"] for group in groups]
    assert names == ([
        "prompts", "action_projections",
        "action_head_core" if full_head else "action_expert_blocks",
    ] + (["vlm_last_layers"] if last else []))
    actual = [id(param) for group in groups for param in group["params"]]
    expected = {id(param) for param in model.parameters() if param.requires_grad}
    assert len(actual) == len(set(actual))
    assert set(actual) == expected
    assert [group["lr"] for group in groups] == pytest.approx(
        [2.5e-5, 1.0e-5, 1.0e-5] + ([1e-6] if last else [])
    )
    optimizer = torch.optim.AdamW(groups)
    assert len(optimizer.param_groups) == len(groups)


@pytest.fixture
def fake_reality(tmp_path):
    model = tmp_path / "task2-50k"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "num_vlm_prompt_tokens": 16,
        "num_prompt_tokens": 16,
        "piper_train_normalization": True,
        "use_relative_actions": False,
        "training_stage": "flow",
    }))
    save_file({
        "model.vlm_prompt_tokens.weight": torch.zeros(16, 2048),
        "model.prompt_tokens.weight": torch.zeros(16, 1024),
    }, model / "model.safetensors")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (model / name).write_text("{}")

    dataset = tmp_path / "data" / TASK
    for name in (
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "data/chunk-000/file-000.parquet",
    ):
        path = dataset / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    cameras = ["observation.images.cam_high", "observation.images.cam_wrist"]
    for cam in cameras:
        path = dataset / f"videos/{cam}/chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    dataset_info = {
        "codebase_version": "v3.0", "fps": 10, "total_episodes": 10, "total_frames": 100,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [7]},
            "action": {"dtype": "float32", "shape": [7]},
            **{key: {"dtype": "video", "shape": [224, 224, 3]} for key in cameras},
        },
    }
    (dataset / "meta/info.json").write_text(json.dumps(dataset_info))
    (dataset / "meta/stats.json").write_text(json.dumps({
        key: {"q01": [0] * 7, "q99": [1] * 7} for key in ("observation.state", "action")
    }))
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text("{}")
    (tokenizer / "tokenizer.json").write_text("{}")
    env = {
        **os.environ,
        "PRETRAINED_PATH": str(model),
        "DATASET_BASE": str(dataset.parent),
        "TOKENIZER_PATH": str(tokenizer),
        "WORK_ROOT": str(tmp_path / "work"),
        "DRY_RUN": "true",
        "GPU_IDS": "1",
        "FLOW_STEPS": "1000",
        "SAVE_STEPS": "[100,1000]",
        "WARMUP_STEPS": "100",
    }
    for key in ("OUTPUT_ROOT", "LOG_ROOT", "RUN_GROUP", "NUM_PROCESSES", "DTYPE", "MIXED_PRECISION"):
        env.pop(key, None)
    return env


@pytest.mark.parametrize("profile,last,full_head", [
    ("action4", 0, False), ("dual2", 2, False), ("dual4", 4, False),
    ("fullhead", 0, True), ("fullhead_vlm2", 2, True), ("fullhead_vlm4", 4, True),
])
def test_launcher_dry_run_has_no_training_and_correct_cli(fake_reality,profile,last,full_head):
    result = subprocess.run(
        ["bash", str(LAUNCHER), profile],
        env={**fake_reality, "RUN_GROUP": f"test-{profile}"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    output = result.stdout
    shell_line = next(line for line in output.splitlines() if line.startswith('env '))
    tokens = set(shlex.split(shell_line))
    for token in (
        "--policy.partial_ft_enabled=true",
        f"--policy.train_vlm_last_n_layers={last}",
        f"--policy.train_action_expert_last_n_layers={0 if full_head else 4}",
        f"--policy.train_action_head={'true' if full_head else 'false'}",
        "--policy.action_projection_lr_scale=0.4",
        "--policy.action_expert_lr_scale=0.4",
        "--policy.vlm_lr_scale=0.04",
        "--policy.next_action_pretrain_steps=0",
        "--policy.next_action_bridge_steps=0",
        "--policy.cabo_enabled=false",
        "--dataset.eval_split=0.1",
        "--steps=1000",
        "--save_steps=[100,1000]",
    ):
        assert token in tokens, (token, output)
    assert not (Path(fake_reality["WORK_ROOT"]) / "chkpt").exists()


def test_launcher_rejects_multi_gpu_and_wrong_checkpoint(fake_reality):
    for env in (
        {**fake_reality, "GPU_IDS": "1,2"},
        {**fake_reality, "PRETRAINED_PATH": fake_reality["WORK_ROOT"]},
    ):
        result = subprocess.run(["bash", str(LAUNCHER), "dual2"], env=env,
                                text=True, capture_output=True, check=False)
        assert result.returncode != 0
