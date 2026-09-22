"""Analytic NTK, paired-seed, checkpoint timeline and plotting regression checks."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.scripts.analyze_pi05_ntk_stages import make_parser, resolve_stages
from lerobot.scripts.plot_pi05_ntk_stages import plot_results
from lerobot.utils.ntk import compress_gradients, projected_jacobian_rows, spectral_metrics


def test_known_spectra_and_energy_normalization():
    result = spectral_metrics(np.eye(4) * 6, 12)
    assert result["effective_rank"] == pytest.approx(4)
    assert result["parameter_normalized_energy"] == pytest.approx(2)
    assert spectral_metrics(np.ones((3, 3)), 3)["effective_rank"] == pytest.approx(1)
    zero = spectral_metrics(np.zeros((3, 3)), 3)
    assert zero["effective_rank"] == zero["parameter_normalized_energy"] == 0
    # This spectrum distinguishes entropy rank from participation ratio.
    spectrum = spectral_metrics(np.diag([3, 1]), 4)
    assert spectrum["effective_rank"] == pytest.approx(np.exp(-0.75 * np.log(0.75) - 0.25 * np.log(0.25)))
    assert spectrum["participation_rank"] == pytest.approx(1.6)
    with pytest.raises(ValueError, match="semidefinite"):
        spectral_metrics(np.diag([1, -1]), 2)


def test_shared_output_probes_recover_exact_multioutput_linear_ntk():
    weight = torch.nn.Parameter(torch.tensor([[0.3, 0.2], [0.7, -0.1]]))
    unused = torch.nn.Parameter(torch.ones(3))
    groups = {"prompts/vlm": [("weight", weight), ("unused", unused)]}
    inputs = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.0, 2.0]])
    # Full two-dimensional Hadamard probes: average r r^T = I, so no projection error.
    probes = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    features, trace = [], 0.0
    for vector in inputs:
        rows, norms = projected_jacobian_rows(weight @ vector, probes, groups)
        features.append(rows["prompts/vlm"])
        trace += np.mean(norms["prompts/vlm"])
    features = np.asarray(features)
    kernel = np.einsum("iqp,jqp->ij", features, features) / 2
    expected = 2 * (inputs @ inputs.T).numpy()
    np.testing.assert_allclose(kernel, expected)
    result = spectral_metrics(kernel, 7, trace)
    assert result["parameter_normalized_energy"] == pytest.approx(np.trace(expected) / 7)
    # Unused parameters contribute zero while remaining in P.
    assert np.count_nonzero(features[..., -3:]) == 0


def test_countsketch_preserves_true_energy_and_is_chunk_independent():
    parameter = torch.nn.Parameter(torch.zeros(10007))
    generator = torch.Generator().manual_seed(29)
    gradient = torch.randn(parameter.shape, generator=generator)
    named = [("model.weights", parameter)]
    a, energy_a = compress_gradients(named, [gradient], 1024, 27, chunk_size=137)
    b, energy_b = compress_gradients(named, [gradient], 1024, 27, chunk_size=4000)
    np.testing.assert_allclose(a, b, atol=2e-6)
    assert energy_a == pytest.approx(gradient.double().square().sum().item())
    assert energy_a == pytest.approx(energy_b)
    c, _ = compress_gradients(named, [gradient], 1024, 28)
    assert not np.allclose(a, c)


def _fake_run(root, flow_steps=2000):
    run = root / "run"
    pretrain = root / "run_next_action_pretrain"
    for base, step in [(pretrain, 0), (pretrain, 750), (pretrain, 1000), (run, flow_steps)]:
        path = base / f"checkpoints/{step:06d}/pretrained_model"
        path.mkdir(parents=True)
        config = {
            "training_stage": "next_action" if base == pretrain else "flow",
            "next_action_bridge_steps": 250,
            "next_action_pretrain_steps": 0 if base == pretrain else 1000,
        }
        (path / "config.json").write_text(json.dumps(config))
        (path / "train_config.json").write_text(
            json.dumps({"steps": 1000 if base == pretrain else flow_steps})
        )
        (path / "policy_preprocessor.json").write_text("{}")
        (path / "model.safetensors").touch()
        state = path.parent / "training_state"
        state.mkdir()
        (state / "training_step.json").write_text(json.dumps({"step": step}))
    return run


@pytest.mark.parametrize("flow_steps,total", [(2000, 3000), (3000, 4000)])
def test_timeline_distinguishes_global_and_local_steps(tmp_path, flow_steps, total):
    run = _fake_run(tmp_path, flow_steps)
    args = make_parser().parse_args(["--run-dir", str(run), "--final-flow-steps", str(flow_steps)])
    stages = resolve_stages(args)
    assert [stage["total_step"] for stage in stages] == [0, 750, 1000, total]
    assert "run_next_action_pretrain" in stages[0]["checkpoint"]
    (Path(stages[1]["checkpoint"]).parent / "training_state/training_step.json").write_text('{"step": 751}')
    with pytest.raises(ValueError, match="751"):
        resolve_stages(args)


def test_missing_priming_is_not_silently_substituted(tmp_path):
    run = _fake_run(tmp_path)
    (tmp_path / "run_next_action_pretrain/checkpoints/000750/pretrained_model/model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="ntk_save_stage_snapshots"):
        resolve_stages(make_parser().parse_args(["--run-dir", str(run)]))


def test_plot_complete_paired_time_series(tmp_path):
    pytest.importorskip("matplotlib")
    records = []
    stages = [
        {"name": name, "total_step": step}
        for name, step in zip(("before", "priming", "stage2", "final"), (0, 750, 1000, 3000), strict=True)
    ]
    for index, stage in enumerate(stages):
        for seed in (0, 1):
            groups = {}
            for module, scale in (("vlm", 1.0), ("action", 5.0)):
                groups[f"backbone/{module}"] = spectral_metrics(
                    np.diag([1.0 + index + seed, 2.0]) * scale,
                    20,
                )
            records.append(
                {
                    "stage": stage["name"],
                    "total_step": stage["total_step"],
                    "local_step": 0,
                    "seed": seed,
                    "groups": groups,
                }
            )
    result = {"manifest": {"stages": stages, "seeds": [0, 1]}, "records": records}
    path = tmp_path / "results.json"
    path.write_text(json.dumps(result))
    plot_results(path)
    assert len(list(tmp_path.glob("*.png"))) == len(list(tmp_path.glob("*.pdf"))) == 4
    assert len((tmp_path / "metrics.csv").read_text().splitlines()) == 17
    result["records"].pop()
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="incomplete"):
        plot_results(path)
