#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi05 import (  # noqa: E402
    PI05Config,
    PI05Policy,
    modeling_pi05,
)
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

_WIDTH = 8
_ACTION_DIM = 3
_MAX_ACTION_DIM = 4


class _TinySelfAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)


class _TinyLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self_attn = _TinySelfAttention(width)


class _TinyExpertBackbone(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer(width)])


class _TinyExpert(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.model = _TinyExpertBackbone(width)


class _TinyPaliGemmaWithExpert(nn.Module):
    """Small expert stand-in that preserves the next-action gradient/data flow."""

    def __init__(self, _paligemma_config, action_expert_config, **_kwargs):
        super().__init__()
        width = action_expert_config.width
        self.paligemma = nn.Linear(width, width, bias=False)
        self.gemma_expert = _TinyExpert(width)
        self.last_call = None

    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        adarms_cond=None,
    ):
        del past_key_values, use_cache
        assert inputs_embeds[0] is None
        suffix_embs = inputs_embeds[1]
        context_steps = suffix_embs.shape[1] // 2
        transformed = self.gemma_expert.model.layers[0].self_attn.q_proj(suffix_embs)
        # Make query outputs depend on the complete context so action_in_proj receives gradients.
        suffix_out = transformed + transformed[:, :context_steps].mean(dim=1, keepdim=True)
        self.last_call = {
            "attention_mask": attention_mask.detach().clone(),
            "position_ids": position_ids.detach().clone(),
            "suffix_embs": suffix_embs.detach().clone(),
            "adarms_cond": adarms_cond[1].detach().clone(),
        }
        return (None, suffix_out), None


@pytest.fixture(autouse=True)
def _tiny_pi05_modules(monkeypatch):
    monkeypatch.setattr(
        modeling_pi05,
        "get_gemma_config",
        lambda _variant: SimpleNamespace(width=_WIDTH),
    )
    monkeypatch.setattr(
        modeling_pi05,
        "PaliGemmaWithExpertModel",
        _TinyPaliGemmaWithExpert,
    )


def _make_config(stage: str = "next_action") -> PI05Config:
    return PI05Config(
        training_stage=stage,
        dtype="float32",
        device="cpu",
        max_action_dim=_MAX_ACTION_DIM,
        max_state_dim=_MAX_ACTION_DIM,
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,)),
        },
    )


def _make_policy(stage: str = "next_action") -> PI05Policy:
    return PI05Policy(_make_config(stage))


def _save_checkpoint(path, policy: PI05Policy, state_dict=None) -> None:
    path.mkdir()
    source = policy.state_dict() if state_dict is None else state_dict
    tensors = {key: value.detach().cpu().contiguous() for key, value in source.items()}
    save_file(tensors, path / "model.safetensors")


def test_next_action_uses_shared_queries_two_attention_blocks_and_fixed_positions():
    policy = _make_policy()
    core = policy.model
    context = torch.randn(2, 25, _MAX_ACTION_DIM)
    context_is_pad = torch.zeros(2, 25, dtype=torch.bool)
    context_is_pad[1, 24] = True

    predictions = core.predict_next_actions(context, context_is_pad)

    call = core.paligemma_with_expert.last_call
    suffix_embs = call["suffix_embs"]
    assert predictions.shape == (2, 25, _MAX_ACTION_DIM)
    assert core.next_action_query.shape == (_WIDTH,)
    assert suffix_embs.shape == (2, 50, _WIDTH)
    safe_context = torch.where(
        context_is_pad.unsqueeze(-1),
        torch.zeros_like(context),
        context,
    )
    torch.testing.assert_close(suffix_embs[:, :25], core.action_in_proj(safe_context))
    torch.testing.assert_close(
        suffix_embs[:, 25:],
        core.next_action_query.view(1, 1, -1).expand(2, 25, -1),
    )
    torch.testing.assert_close(call["position_ids"], torch.arange(50).unsqueeze(0).expand(2, -1))

    allowed = call["attention_mask"][:, 0].eq(0)
    assert allowed[0, :25, :25].all()
    assert not allowed[0, :25, 25:].any()
    assert allowed[0, 25:, :].all()
    # Padding removes a source token both as a query row and an attention key.
    assert not allowed[1, 24, :].any()
    assert not allowed[1, :, 24].any()
    assert allowed[1, 25:, 25:].all()


def test_next_action_predictions_do_not_depend_on_target_actions():
    policy = _make_policy()
    actions = torch.randn(2, 50, _MAX_ACTION_DIM)
    changed_targets = actions.clone()
    changed_targets[:, 25:] = torch.randn_like(changed_targets[:, 25:]) * 1000
    context_is_pad = torch.zeros(2, 25, dtype=torch.bool)

    prediction_a = policy.model.predict_next_actions(actions[:, :25], context_is_pad)
    prediction_b = policy.model.predict_next_actions(changed_targets[:, :25], context_is_pad)

    torch.testing.assert_close(prediction_a, prediction_b)


def test_next_action_masked_loss_uses_only_valid_targets_and_actual_action_dims():
    policy = _make_policy()
    actions = torch.randn(2, 50, _ACTION_DIM)
    action_is_pad = torch.zeros(2, 50, dtype=torch.bool)
    action_is_pad[1, 27:] = True
    actions[1, 27:] = torch.nan
    batch = {ACTION: actions, "action_is_pad": action_is_pad}

    padded_actions = policy.prepare_action(batch)
    raw_losses = policy.model.forward(
        actions=padded_actions,
        action_is_pad=action_is_pad,
    )[:, :, :_ACTION_DIM]
    target_valid = ~action_is_pad[:, 25:]
    valid_elements = target_valid.unsqueeze(-1).expand_as(raw_losses)
    masked_losses = torch.where(valid_elements, raw_losses, torch.zeros_like(raw_losses))
    expected_loss = masked_losses.sum() / valid_elements.sum()
    expected_per_sample = masked_losses.sum(dim=(1, 2)) / valid_elements.sum(dim=(1, 2))

    loss, metrics = policy.forward(batch)
    per_sample, _ = policy.forward(batch, reduction="none")

    torch.testing.assert_close(loss, expected_loss)
    assert torch.isfinite(loss)
    torch.testing.assert_close(per_sample, expected_per_sample)
    assert metrics["next_action/valid_target_count"] == 27
    assert metrics["next_action/valid_target_fraction"] == pytest.approx(27 / 50)
    assert len(metrics["loss_per_dim"]) == _ACTION_DIM


def test_next_action_all_padded_targets_return_graph_connected_zero():
    policy = _make_policy()
    actions = torch.randn(2, 50, _ACTION_DIM)
    action_is_pad = torch.ones(2, 50, dtype=torch.bool)
    actions[:] = torch.nan

    loss, metrics = policy.forward({ACTION: actions, "action_is_pad": action_is_pad})
    loss.backward()

    assert loss.item() == 0.0
    assert loss.requires_grad
    assert metrics["next_action/valid_target_count"] == 0
    assert metrics["next_action/valid_target_fraction"] == 0.0
    assert policy.model.next_action_out_proj.weight.grad is not None
    assert torch.count_nonzero(policy.model.next_action_out_proj.weight.grad) == 0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in policy.parameters()
    )


def test_flow_masked_loss_uses_only_valid_timesteps_and_actual_action_dims(monkeypatch):
    policy = _make_policy("flow")
    raw_losses = torch.ones(2, 50, _MAX_ACTION_DIM, requires_grad=True)
    with torch.no_grad():
        raw_losses[1, :2, :_ACTION_DIM] = 2.0
        raw_losses[1, 2:, :_ACTION_DIM] = torch.nan
        raw_losses[:, :, _ACTION_DIM:] = torch.nan
    action_is_pad = torch.zeros(2, 50, dtype=torch.bool)
    action_is_pad[1, 2:] = True
    batch = {
        ACTION: torch.randn(2, 50, _ACTION_DIM),
        "action_is_pad": action_is_pad,
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }
    monkeypatch.setattr(policy, "_preprocess_images", lambda _batch: ([], []))
    monkeypatch.setattr(policy.model, "forward", lambda *_args, **_kwargs: raw_losses)

    loss, metrics = policy.forward(batch)
    per_sample, _ = policy.forward(batch, reduction="none")
    loss.backward()

    expected_loss = torch.tensor((50 * _ACTION_DIM + 2 * 2 * _ACTION_DIM) / (52 * _ACTION_DIM))
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(per_sample, torch.tensor([1.0, 2.0]))
    assert torch.isfinite(loss)
    assert metrics["valid_action_fraction"] == pytest.approx(52 / 100)
    assert metrics["all_padding_samples"] == 0.0
    assert metrics["loss_per_dim"] == pytest.approx([54 / 52] * _ACTION_DIM)
    assert raw_losses.grad is not None
    assert torch.count_nonzero(raw_losses.grad[1, 2:, :_ACTION_DIM]) == 0
    assert torch.count_nonzero(raw_losses.grad[:, :, _ACTION_DIM:]) == 0


def test_flow_all_padded_actions_return_graph_connected_zero(monkeypatch):
    policy = _make_policy("flow")
    raw_losses = torch.randn(2, 50, _MAX_ACTION_DIM, requires_grad=True)
    batch = {
        ACTION: torch.randn(2, 50, _ACTION_DIM),
        "action_is_pad": torch.ones(2, 50, dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }
    monkeypatch.setattr(policy, "_preprocess_images", lambda _batch: ([], []))
    monkeypatch.setattr(policy.model, "forward", lambda *_args, **_kwargs: raw_losses)

    loss, metrics = policy.forward(batch)
    loss.backward()

    assert loss.item() == 0.0
    assert loss.requires_grad
    assert metrics["valid_action_fraction"] == 0.0
    assert metrics["all_padding_samples"] == 2.0
    assert raw_losses.grad is not None
    assert torch.count_nonzero(raw_losses.grad) == 0


def test_flow_without_padding_mask_preserves_unmasked_mean(monkeypatch):
    policy = _make_policy("flow")
    raw_losses = torch.arange(2 * 50 * _MAX_ACTION_DIM, dtype=torch.float32).reshape(2, 50, _MAX_ACTION_DIM)
    batch = {
        ACTION: torch.randn(2, 50, _ACTION_DIM),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }
    monkeypatch.setattr(policy, "_preprocess_images", lambda _batch: ([], []))
    monkeypatch.setattr(policy.model, "forward", lambda *_args, **_kwargs: raw_losses)

    loss, metrics = policy.forward(batch)

    torch.testing.assert_close(loss, raw_losses[:, :, :_ACTION_DIM].mean())
    assert metrics["valid_action_fraction"] == 1.0


def test_flow_padding_mask_shape_is_validated(monkeypatch):
    policy = _make_policy("flow")
    batch = {
        ACTION: torch.randn(2, 50, _ACTION_DIM),
        "action_is_pad": torch.zeros(50, 2, dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }
    monkeypatch.setattr(policy, "_preprocess_images", lambda _batch: ([], []))
    monkeypatch.setattr(
        policy.model,
        "forward",
        lambda *_args, **_kwargs: torch.zeros(2, 50, _MAX_ACTION_DIM, requires_grad=True),
    )

    with pytest.raises(ValueError, match="action_is_pad must have shape"):
        policy.forward(batch)


def test_pi05_next_action_sampler_drop_matches_context_horizon():
    assert _make_config("flow").drop_n_last_frames == 0
    next_action_config = PI05Config(
        training_stage="next_action",
        chunk_size=12,
        n_action_steps=12,
        next_action_context_steps=7,
        next_action_prediction_steps=5,
    )

    assert next_action_config.drop_n_last_frames == 7


def test_pi05_distributed_loss_normalizer_counts_valid_actual_action_elements():
    batch = {
        ACTION: torch.randn(2, 50, _ACTION_DIM),
        "action_is_pad": torch.zeros(2, 50, dtype=torch.bool),
    }
    batch["action_is_pad"][1, 27:] = True

    next_action_normalizer = _make_policy("next_action").get_distributed_loss_normalizer(batch)
    flow_normalizer = _make_policy("flow").get_distributed_loss_normalizer(batch)

    assert next_action_normalizer.item() == (25 + 2) * _ACTION_DIM
    assert flow_normalizer.item() == (50 + 27) * _ACTION_DIM


def test_next_action_validates_padding_mask_and_temporal_order():
    policy = _make_policy()
    actions = torch.randn(1, 50, _ACTION_DIM)

    with pytest.raises(ValueError, match="action_is_pad is required"):
        policy.forward({ACTION: actions})
    with pytest.raises(ValueError, match="must have shape"):
        policy.forward({ACTION: actions, "action_is_pad": torch.zeros(1, 49, dtype=torch.bool)})

    invalid_order = torch.zeros(1, 50, dtype=torch.bool)
    invalid_order[0, 24] = True
    with pytest.raises(ValueError, match="valid after padded context"):
        policy.forward({ACTION: actions, "action_is_pad": invalid_order})


def test_next_action_only_trains_transferable_expert_input_and_stage_parameters():
    policy = _make_policy()
    batch = {
        ACTION: torch.randn(2, 50, _ACTION_DIM),
        "action_is_pad": torch.zeros(2, 50, dtype=torch.bool),
    }

    loss, _ = policy.forward(batch)
    loss.backward()

    trainable = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}
    expected_prefixes = (
        "model.paligemma_with_expert.gemma_expert.",
        "model.action_in_proj.",
        "model.next_action_out_proj.",
    )
    assert "model.next_action_query" in trainable
    assert all(name == "model.next_action_query" or name.startswith(expected_prefixes) for name in trainable)
    assert all(dict(policy.named_parameters())[name].grad is not None for name in trainable)

    for frozen_prefix in (
        "model.paligemma_with_expert.paligemma.",
        "model.action_out_proj.",
        "model.time_mlp_in.",
        "model.time_mlp_out.",
    ):
        frozen = [
            parameter for name, parameter in policy.named_parameters() if name.startswith(frozen_prefix)
        ]
        assert frozen
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in frozen)

    optimizer_parameters = policy.get_optim_params()
    assert all(isinstance(parameter, nn.Parameter) for parameter in optimizer_parameters)
    assert {id(parameter) for parameter in optimizer_parameters} == {
        id(parameter) for parameter in policy.parameters() if parameter.requires_grad
    }


def test_next_action_forward_supports_gradient_checkpointing():
    policy = _make_policy()
    policy.model.gradient_checkpointing_enabled = True
    policy.train()

    loss, _ = policy.forward(
        {
            ACTION: torch.randn(2, 50, _ACTION_DIM),
            "action_is_pad": torch.zeros(2, 50, dtype=torch.bool),
        }
    )
    loss.backward()

    assert policy.model.action_in_proj.weight.grad is not None
    assert policy.model.next_action_query.grad is not None
    assert policy.model.next_action_out_proj.weight.grad is not None


@pytest.mark.skipif(
    not hasattr(torch, "compile") or not torch._dynamo.is_dynamo_supported(),
    reason="torch.compile is unavailable on this platform",
)
def test_next_action_forward_supports_torch_compile():
    policy = _make_policy()
    forward_kwargs = {
        "actions": policy.prepare_action({ACTION: torch.randn(1, 50, _ACTION_DIM)}),
        "action_is_pad": torch.zeros(1, 50, dtype=torch.bool),
    }
    explanation = torch._dynamo.explain(policy.model.forward)(**forward_kwargs)
    compiled_forward = torch.compile(policy.model.forward, backend="eager")

    losses = compiled_forward(**forward_kwargs)

    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
    assert losses.shape == (1, 25, _MAX_ACTION_DIM)


@pytest.mark.parametrize(
    "method_name",
    ["select_action", "predict_action_chunk", "predict_action_chunk_with_context"],
)
def test_next_action_checkpoint_rejects_action_inference_before_batch_preprocessing(method_name):
    policy = _make_policy()

    with pytest.raises(RuntimeError, match="next-action pretraining checkpoint"):
        getattr(policy, method_name)({})


def test_flow_checkpoint_initializes_stage_parameters_after_loading(tmp_path):
    source = _make_policy("flow")
    with torch.no_grad():
        source.model.action_in_proj.bias.fill_(1.25)
        source.model.action_out_proj.weight.fill_(2.5)
        source.model.action_out_proj.bias.fill_(-0.75)
    checkpoint = tmp_path / "flow"
    _save_checkpoint(checkpoint, source)

    loaded = PI05Policy.from_pretrained(
        checkpoint,
        config=_make_config("next_action"),
        local_files_only=True,
    )

    torch.testing.assert_close(loaded.model.next_action_query, loaded.model.action_in_proj.bias)
    torch.testing.assert_close(loaded.model.next_action_out_proj.weight, loaded.model.action_out_proj.weight)
    torch.testing.assert_close(loaded.model.next_action_out_proj.bias, loaded.model.action_out_proj.bias)


def test_next_action_checkpoint_resumes_stage_parameters_and_transfers_strictly_to_flow(tmp_path):
    source = _make_policy()
    with torch.no_grad():
        source.model.action_in_proj.weight.fill_(0.25)
        source.model.next_action_query.fill_(3.0)
        source.model.next_action_out_proj.weight.fill_(4.0)
        source.model.next_action_out_proj.bias.fill_(5.0)
    checkpoint = tmp_path / "next_action"
    _save_checkpoint(checkpoint, source)

    resumed = PI05Policy.from_pretrained(
        checkpoint,
        config=_make_config("next_action"),
        local_files_only=True,
    )
    torch.testing.assert_close(resumed.model.next_action_query, source.model.next_action_query)
    torch.testing.assert_close(
        resumed.model.next_action_out_proj.weight, source.model.next_action_out_proj.weight
    )

    flow = PI05Policy.from_pretrained(
        checkpoint,
        config=_make_config("flow"),
        local_files_only=True,
    )
    assert not hasattr(flow.model, "next_action_query")
    assert not hasattr(flow.model, "next_action_out_proj")
    assert flow.config.cabo_active
    assert all(parameter.requires_grad for parameter in flow.parameters())
    torch.testing.assert_close(flow.model.action_in_proj.weight, source.model.action_in_proj.weight)
    assert set(flow.state_dict()) == set(_make_policy("flow").state_dict())


@pytest.mark.parametrize("corruption", ["partial_stage", "unknown", "missing_shared", "shape_mismatch"])
def test_checkpoint_loading_rejects_non_stage_incompatibilities(tmp_path, corruption):
    source = _make_policy()
    state_dict = dict(source.state_dict())
    if corruption == "partial_stage":
        state_dict.pop("model.next_action_query")
    elif corruption == "unknown":
        state_dict["model.unknown_parameter"] = torch.zeros(1)
    elif corruption == "missing_shared":
        state_dict.pop("model.action_in_proj.weight")
    else:
        state_dict["model.action_in_proj.weight"] = torch.zeros(1, 1)
    checkpoint = tmp_path / corruption
    _save_checkpoint(checkpoint, source, state_dict)

    with pytest.raises(RuntimeError, match="Failed to load PI05 pretrained weights"):
        PI05Policy.from_pretrained(
            checkpoint,
            config=_make_config("next_action"),
            local_files_only=True,
        )
