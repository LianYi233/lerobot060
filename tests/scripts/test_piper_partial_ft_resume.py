"""CPU-only checks for Piper continuation from 10k to 20k.

Command: PYTHONPATH=src python -m pytest tests/scripts/test_piper_partial_ft_resume.py -q
"""
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest
import torch

from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "examples/training/continue_piper_fullhead_vlm2_to20k.sh"


def _scheduler(*, restart=False):
    return CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=500,
        num_decay_steps=10000,
        peak_lr=2.5e-5,
        decay_lr=1e-6,
        **(
            dict(restart_at_step=10000, restart_warmup_steps=300, restart_peak_ratio=0.5)
            if restart else {}
        ),
    )


def test_restart_lr_matches_previous_floor_then_warms_and_decays():
    param = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([{"params": [param], "lr": 2.5e-5}])
    old = _scheduler().build(optimizer, 10000)
    # Exactly the saved state at the end of an original 10k cosine.
    for _ in range(10000):
        optimizer.step()
        old.step()
    old_state = old.state_dict()
    prior_lr = optimizer.param_groups[0]["lr"]
    assert prior_lr == pytest.approx(1e-6)

    # Construct a NEW restart lambda before loading saved optimizer/scheduler state.
    param2 = torch.nn.Parameter(torch.zeros(()))
    optimizer2 = torch.optim.AdamW([{"params": [param2], "lr": 2.5e-5}])
    restarted = _scheduler(restart=True).build(optimizer2, 20000)
    optimizer2.load_state_dict(optimizer.state_dict())
    restarted.load_state_dict(old_state)
    assert restarted.last_epoch == 10000
    assert optimizer2.param_groups[0]["lr"] == pytest.approx(prior_lr)

    for _ in range(300):
        optimizer2.step()
        restarted.step()
    assert optimizer2.param_groups[0]["lr"] == pytest.approx(1.25e-5)
    for _ in range(20000 - 10300):
        optimizer2.step()
        restarted.step()
    assert restarted.last_epoch == 20000
    assert optimizer2.param_groups[0]["lr"] == pytest.approx(1e-6)


@pytest.mark.parametrize("bad", [
    dict(restart_at_step=20000),
    dict(restart_at_step=-1),
    dict(restart_at_step=10000, restart_warmup_steps=10000),
    dict(restart_at_step=10000, restart_peak_ratio=1.1),
    dict(restart_at_step=10000, restart_peak_ratio=0.01),
])
def test_restart_rejects_bad_settings(bad):
    p = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([p], lr=2.5e-5)
    fields = {
        "num_warmup_steps": 500,
        "num_decay_steps": 10000,
        "peak_lr": 2.5e-5,
        "decay_lr": 1e-6,
        **bad,
    }
    with pytest.raises(ValueError):
        CosineDecayWithWarmupSchedulerConfig(**fields).build(optimizer, 20000)


@pytest.fixture
def completed_checkpoint(tmp_path):
    run = tmp_path / "source"
    root = run / "checkpoints/010000"
    model = root / "pretrained_model"
    state = root / "training_state"
    model.mkdir(parents=True)
    state.mkdir()
    data = tmp_path / "data" / "2-remove_the_cuboid_from_blue_plate"
    data.mkdir(parents=True)
    cfg = {
        "steps": 10000,
        "batch_size": 1,
        "dataset": {"root": str(data), "eval_split": 0.1},
        "scheduler": {
            "type": "cosine_decay_with_warmup",
            "num_warmup_steps": 500,
            "num_decay_steps": 10000,
            "peak_lr": 2.5e-5,
            "decay_lr": 1e-6,
        },
        "policy": {
            "partial_ft_enabled": True,
            "train_action_head": True,
            "train_action_expert_last_n_layers": 0,
            "train_action_projections": True,
            "train_vlm_last_n_layers": 2,
            "cabo_enabled": False,
            "piper_train_normalization": True,
            "use_relative_actions": False,
        },
    }
    (model / "train_config.json").write_text(json.dumps(cfg))
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json",
                 "policy_postprocessor.json"):
        (model / name).touch()
    (state / "training_step.json").write_text(
        json.dumps({"step": 10000, "batch_size": 1, "num_processes": 1})
    )
    for name in ("optimizer_state.safetensors", "optimizer_param_groups.json",
                 "scheduler_state.json", "rng_state.safetensors"):
        (state / name).touch()
    return run


def _run(checkpoint, tmp_path, *, dry_run=True):
    env = {
        **os.environ,
        "SOURCE_RUN": str(checkpoint),
        "WORK_ROOT": str(tmp_path),
        "RUN_GROUP": "new20k",
        "DRY_RUN": "true" if dry_run else "false",
        "GPU_IDS": "0",
        "SOURCE_STEP": "10000",
        "TARGET_STEPS": "20000",
        "SAVE_STEPS": "[12000,15000,20000]",
    }
    return subprocess.run(["bash", str(LAUNCHER)], env=env, check=False,
                          capture_output=True, text=True)


def test_dry_run_uses_real_resume_not_fresh_weights(completed_checkpoint, tmp_path):
    result = _run(completed_checkpoint, tmp_path)
    assert result.returncode == 0, (result.stdout, result.stderr)
    line = next(x for x in result.stdout.splitlines() if x.startswith("env "))
    args = set(shlex.split(line))
    for item in (
        "--resume=true",
        "--module",
        "lerobot.scripts.lerobot_train",
        "--num_machines=1",
        "--dynamo_backend=no",
        "--steps=20000",
        "--save_steps=[12000,15000,20000]",
        "--scheduler.restart_at_step=10000",
        "--scheduler.restart_warmup_steps=300",
        "--scheduler.restart_peak_ratio=0.5",
    ):
        assert item in args
    assert any(x.startswith("--config_path=") and
               "checkpoints/010000/pretrained_model/train_config.json" in x for x in args)
    assert "lerobot-train" not in args
    assert not (tmp_path / "chkpt").exists()


def test_missing_optimizer_state_fails_closed(completed_checkpoint, tmp_path):
    state = completed_checkpoint / "checkpoints/010000/training_state/optimizer_state.safetensors"
    state.unlink()
    result = _run(completed_checkpoint, tmp_path)
    assert result.returncode != 0
    assert "missing:" in result.stderr


def test_destination_collision_fails_closed(completed_checkpoint, tmp_path):
    target = tmp_path / "chkpt/2601-lerobot/piper/new20k/"
    target.mkdir(parents=True)
    folder = target / "pi05-may-2-remove_the_cuboid_from_blue_plate-dual_prompt_only-seed0"
    folder.mkdir()
    result = _run(completed_checkpoint, tmp_path)
    assert result.returncode != 0
    assert "Refusing to overwrite" in result.stderr


def test_task3_resume_dry_run_preserves_matching_task_identity(completed_checkpoint, tmp_path):
    root = completed_checkpoint / "checkpoints/010000/pretrained_model/train_config.json"
    cfg = json.loads(root.read_text())
    folder = "3-move_the_tennis_from_yellow_plate_to_blue_plate"
    dataset = tmp_path / "data" / folder
    dataset.mkdir(parents=True)
    cfg["dataset"]["root"] = str(dataset)
    cfg["dataset"]["repo_id"] = "may-pick-and-place/" + folder
    root.write_text(json.dumps(cfg))
    env = {
        **os.environ,
        "SOURCE_RUN": str(completed_checkpoint),
        "WORK_ROOT": str(tmp_path),
        "DRY_RUN": "true",
        "GPU_IDS": "0",
        "PIPER_TASK": "3",
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "task=3" in result.stdout
    cmdline = next(line for line in result.stdout.splitlines() if line.startswith("env "))
    tokens = set(shlex.split(cmdline))
    assert any(token.startswith("--output_dir=") and
               "task3-fullhead-vlm2-resume10k-to20k" in token and folder in token
               for token in tokens)


def test_task3_resume_rejects_task2_training_config(completed_checkpoint, tmp_path):
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env={
            **os.environ,
            "SOURCE_RUN": str(completed_checkpoint),
            "WORK_ROOT": str(tmp_path),
            "PIPER_TASK": "3",
            "DRY_RUN": "true",
        },
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "AssertionError" in result.stderr


def test_task4_resume_from_completed_task4_10k_checkpoint(completed_checkpoint, tmp_path):
    train_cfg_file = completed_checkpoint / "checkpoints/010000/pretrained_model/train_config.json"
    cfg = json.loads(train_cfg_file.read_text())
    folder = "4-pick_the_red_cube_into_the_yellow_plate"
    dataset = tmp_path / "data" / folder
    dataset.mkdir(parents=True)
    cfg["dataset"]["root"] = str(dataset)
    cfg["dataset"]["repo_id"] = "may-pick-and-place/" + folder
    train_cfg_file.write_text(json.dumps(cfg))
    env = {
        **os.environ,
        "SOURCE_RUN": str(completed_checkpoint),
        "WORK_ROOT": str(tmp_path),
        "DRY_RUN": "true",
        "GPU_IDS": "0",
        "PIPER_TASK": "4",
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "task=4" in result.stdout
    line = next(x for x in result.stdout.splitlines() if x.startswith("env "))
    args = set(shlex.split(line))
    assert any(x.startswith("--output_dir=") and
               "task4-fullhead-vlm2-resume10k-to20k" in x and folder in x for x in args)
    assert "--resume=true" in args
    assert "--steps=20000" in args
    assert "--scheduler.restart_at_step=10000" in args


def test_task4_resume_rejects_task2_training_config(completed_checkpoint, tmp_path):
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env={
            **os.environ,
            "SOURCE_RUN": str(completed_checkpoint),
            "WORK_ROOT": str(tmp_path),
            "PIPER_TASK": "4",
            "DRY_RUN": "true",
        },
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "AssertionError" in result.stderr
