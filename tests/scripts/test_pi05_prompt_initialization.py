"""Initialization swaps must preserve the control run's subsequent stochastic training inputs."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from lerobot.utils.prompt_initialization import (
    KEY,
    METADATA_KEY,
    apply_initial_prompt,
    export_initial_prompt,
    prompt_hash,
    read_initial_prompt,
    save_import_record,
    write_prompt,
)


def packet(tmp_path, prompt=None):
    if prompt is None:
        prompt = np.random.default_rng(1).normal(0, 0.02, (2, 2048)).astype(np.float32)
    metadata = {
        "schema_version": 1,
        "source_step": 0,
        "source_training_seed": 1,
        "num_vlm_prompt_tokens": prompt.shape[0],
        "prompt_sha256": prompt_hash(prompt),
    }
    path = tmp_path / "seed1.safetensors"
    write_prompt(path, prompt, metadata)
    return path, prompt, metadata


class Policy(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.config = SimpleNamespace(vlm_prompt_only=True, num_prompt_tokens=0, num_vlm_prompt_tokens=2)
        self.model = torch.nn.Module()
        self.model.backbone = torch.nn.Linear(2048, 7).to(dtype=dtype).requires_grad_(False)
        self.model.vlm_prompt_tokens = torch.nn.Embedding(2, 2048).to(dtype=dtype)


def training_draws():
    return (
        random.random(),
        np.random.rand(8),
        torch.normal(0.0, 1.0, (2, 50, 32)),
        torch.distributions.Beta(1.5, 1.0).sample((2,)),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_swap_preserves_rng_future_flow_draws_parameter_identity_and_frozen_weights(tmp_path, dtype):
    original = np.random.default_rng(1).normal(0, 0.02, (2, 2048)).astype(np.float32)
    prompt = torch.from_numpy(original).to(dtype=dtype).float().numpy()
    path, prompt, _ = packet(tmp_path, prompt)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    policy = Policy(dtype)
    parameter = policy.model.vlm_prompt_tokens.weight
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    backbone = {name: value.clone() for name, value in policy.model.backbone.state_dict().items()}
    states = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    expected = [training_draws() for _ in range(3)]
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    record = apply_initial_prompt(policy, path)
    # Persistence must not advance RNG either.
    save_import_record(policy, record, tmp_path / "new_run", 0)
    assert random.getstate() == states[0]
    np.testing.assert_array_equal(np.random.get_state()[1], states[1][1])
    assert np.random.get_state()[2:] == states[1][2:]
    assert torch.equal(torch.get_rng_state(), states[2])
    for actual, reference in zip([training_draws() for _ in range(3)], expected, strict=True):
        assert actual[0] == reference[0]
        np.testing.assert_array_equal(actual[1], reference[1])
        assert torch.equal(actual[2], reference[2])
        assert torch.equal(actual[3], reference[3])
    assert policy.model.vlm_prompt_tokens.weight is parameter
    assert optimizer.param_groups[0]["params"][0] is parameter
    np.testing.assert_array_equal(parameter.detach().float().numpy(), prompt)
    for name, value in policy.model.backbone.state_dict().items():
        assert torch.equal(value, backbone[name])
    assert record["loaded_prompt_sha256"] == prompt_hash(prompt)
    assert record["replaced_prompt_sha256"] != record["loaded_prompt_sha256"]
    path.unlink()
    transferred, metadata = read_initial_prompt(tmp_path / "new_run/prompt_init.safetensors", 2)
    np.testing.assert_array_equal(transferred, prompt)
    assert metadata["training_seed"] == 0 and metadata["source_training_seed"] == 1


def make_run(tmp_path):
    run = tmp_path / "pi05-base-vlm2-seed1"
    (run / "prompt_rank").mkdir(parents=True)
    model = run / "checkpoints/003000/pretrained_model"
    model.mkdir(parents=True)
    (model / "train_config.json").write_text(
        json.dumps(
            {
                "seed": 1,
                "policy": {
                    "type": "pi05",
                    "vlm_prompt_only": True,
                    "num_vlm_prompt_tokens": 2,
                    "num_prompt_tokens": 0,
                },
            }
        )
    )
    (run / "prompt_rank/manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "num_vlm_prompt_tokens": 2,
                "forward_precision": "bfloat16",
                "pretrained_path": "/old-server/models/pi05_libero_base",
            }
        )
    )
    prompt = np.random.default_rng(1).normal(0, 0.02, (2, 2048)).astype(np.float32)
    np.savez(run / "prompt_rank/step_000000.npz", prompt=prompt)
    np.savez(run / "prompt_rank/step_003000.npz", prompt=prompt + 1)
    return run, prompt


def test_export_uses_actual_step_zero_and_survives_transfer_without_source_files(tmp_path):
    run, initial = make_run(tmp_path)
    output = tmp_path / "transfer/init.safetensors"
    metadata = export_initial_prompt(run, output, expect_seed=1)
    imported, actual = read_initial_prompt(output, 2)
    np.testing.assert_array_equal(imported, initial)
    assert metadata == actual
    assert metadata["source_training_seed"] == 1
    assert metadata["source_step"] == 0
    with pytest.raises(FileExistsError):
        export_initial_prompt(run, output, expect_seed=1)
    with pytest.raises(ValueError, match="expected 0"):
        export_initial_prompt(run, tmp_path / "bad.safetensors", expect_seed=0)
    (run / "prompt_rank/step_000000.npz").unlink()
    with pytest.raises(FileNotFoundError):
        export_initial_prompt(run, tmp_path / "bad.safetensors", expect_seed=1)


@pytest.mark.parametrize("change", ["nan", "shape", "checksum", "step", "full_model"])
def test_rejects_corrupt_wrong_shape_and_noninitial_files(tmp_path, change):
    path, prompt, metadata = packet(tmp_path)
    arrays = {KEY: prompt}
    if change == "nan":
        prompt[0, 0] = np.nan
    elif change == "shape":
        arrays[KEY] = prompt[:1]
    elif change == "checksum":
        prompt[0, 0] += 0.01
    elif change == "step":
        metadata["source_step"] = 3000
    else:
        arrays["backbone"] = prompt
    save_file(arrays, path, metadata={METADATA_KEY: json.dumps(metadata)})
    with pytest.raises(ValueError):
        read_initial_prompt(path, 2)


def test_precision_loss_and_unfrozen_backbone_fail_before_mutation(tmp_path):
    path, _, _ = packet(tmp_path)
    policy = Policy(torch.bfloat16)
    original = policy.model.vlm_prompt_tokens.weight.detach().clone()
    with pytest.raises(ValueError, match="exactly"):
        apply_initial_prompt(policy, path)
    assert torch.equal(policy.model.vlm_prompt_tokens.weight, original)
    policy.model.backbone.requires_grad_(True)
    with pytest.raises(ValueError, match="Only the VLM prompt"):
        apply_initial_prompt(policy, path)


def test_export_cli(tmp_path):
    import os
    import subprocess
    import sys

    run, initial = make_run(tmp_path)
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "cli.safetensors"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.export_pi05_prompt_init",
            "export",
            str(run),
            "--output",
            str(output),
            "--expect-seed",
            "1",
        ],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(read_initial_prompt(output, 2)[0], initial)
