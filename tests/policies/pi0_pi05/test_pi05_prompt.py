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

"""Exercise learned prompts through the real PiGemma attention and cache paths on CPU."""

import re
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.optim.cabo import (  # noqa: E402
    CABO_ACTION_EXPERT_GROUP,
    CABO_ACTION_PROJECTION_GROUP,
    CABO_GROUP_NAME,
    CABO_PROMPT_GROUP,
    temporary_optimizer_group_lr_scales,
)
from lerobot.policies.pi05 import PI05Config, PI05Policy, modeling_pi05  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

_EXPERT_WIDTH = 16
_HORIZON = 4
_ACTION_DIM = 3
_PROMPTS = 2
_PROMPT_KEY = "model.prompt_tokens.weight"


@pytest.fixture(autouse=True)
def _tiny_real_pi05(monkeypatch):
    """Reduce allocation sizes, retaining all production model implementations."""
    original_paligemma = modeling_pi05.PaliGemmaForConditionalGenerationWithPiGemma
    original_expert = modeling_pi05.PiGemmaForCausalLM

    def tiny_paligemma(config):
        config._vocab_size = 32
        config.image_token_index = 31
        config.text_config.vocab_size = 32
        config.text_config._attn_implementation = "eager"
        config.vision_config.hidden_size = 16
        config.vision_config.intermediate_size = 32
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 4
        config.vision_config.patch_size = 8
        config.vision_config.projection_dim = 32
        return original_paligemma(config)

    def tiny_expert(config):
        config.vocab_size = 32
        config._attn_implementation = "eager"
        return original_expert(config)

    monkeypatch.setattr(
        modeling_pi05,
        "get_gemma_config",
        lambda variant: modeling_pi05.GemmaConfig(
            width=32 if variant == "gemma_2b" else _EXPERT_WIDTH,
            depth=2,
            mlp_dim=64,
            num_heads=8,
            num_kv_heads=1,
            head_dim=4,
        ),
    )
    monkeypatch.setattr(modeling_pi05, "PaliGemmaForConditionalGenerationWithPiGemma", tiny_paligemma)
    monkeypatch.setattr(modeling_pi05, "PiGemmaForCausalLM", tiny_expert)
    torch.manual_seed(37)


def _config(**kwargs):
    options = {
        "device": "cpu",
        "dtype": "float32",
        "chunk_size": _HORIZON,
        "n_action_steps": _HORIZON,
        "max_action_dim": _ACTION_DIM,
        "image_resolution": (16, 16),
        "num_prompt_tokens": _PROMPTS,
        "next_action_masked_steps": 2,
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,))},
    }
    options.update(kwargs)
    return PI05Config(**options)


def _inputs():
    return {
        "images": [torch.randn(2, 3, 16, 16)],
        "img_masks": [torch.tensor([True, False])],
        "tokens": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "masks": torch.tensor([[True, True, True], [True, True, False]]),
    }


def _actions():
    return torch.randn(2, _HORIZON, _ACTION_DIM)


def _save_checkpoint(path, policy, state_dict=None):
    path.mkdir()
    state_dict = policy.state_dict() if state_dict is None else state_dict
    # Real PaliGemma may tie its language embedding and output weights.
    save_file(
        {key: value.detach().cpu().clone() for key, value in state_dict.items()}, path / "model.safetensors"
    )


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_prompt_count_is_rejected(value):
    with pytest.raises(ValueError, match="num_prompt_tokens"):
        _config(num_prompt_tokens=value)


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_prompt_initialization_is_rejected(value):
    with pytest.raises(ValueError, match="prompt_init_std"):
        _config(prompt_init_std=value)


@pytest.mark.parametrize("num_prompts", [0, _PROMPTS])
def test_prompt_and_action_attention_blocks(num_prompts):
    core = PI05Policy(_config(num_prompt_tokens=num_prompts)).model
    inputs = _inputs()
    prefix, prefix_pad, prefix_att = core.embed_prefix(**inputs)
    actions = _actions()
    suffix, suffix_pad, suffix_att, _ = core.embed_suffix(actions, torch.tensor([0.3, 0.7]))

    assert suffix.shape == (2, num_prompts + _HORIZON, _EXPERT_WIDTH)
    torch.testing.assert_close(suffix[:, num_prompts:], core.action_in_proj(actions))
    torch.testing.assert_close(
        suffix[:, :num_prompts], core.prompt_tokens.weight.unsqueeze(0).expand(2, -1, -1)
    )
    allowed = modeling_pi05.make_att_2d_masks(
        torch.cat([prefix_pad, suffix_pad], dim=1), torch.cat([prefix_att, suffix_att], dim=1)
    )
    prefix_len = prefix.shape[1]
    action_start = prefix_len + num_prompts
    assert not allowed[:, :prefix_len, prefix_len:].any()
    assert not allowed[:, prefix_len:action_start, action_start:].any()
    assert allowed[:, prefix_len:action_start, prefix_len:action_start].all()
    assert allowed[:, action_start:, prefix_len:].all()
    torch.testing.assert_close(
        allowed[:, prefix_len:, :prefix_len],
        prefix_pad[:, None, :].expand(-1, num_prompts + _HORIZON, -1),
    )


def test_inpainting_prompts_remain_valid_and_positions_include_them(monkeypatch):
    core = PI05Policy(_config(training_stage="next_action")).model
    original_forward = core.paligemma_with_expert.forward
    captured = {}

    def capture_forward(**kwargs):
        captured.update(kwargs)
        return original_forward(**kwargs)

    monkeypatch.setattr(core.paligemma_with_expert, "forward", capture_forward)
    action_is_pad = torch.tensor([[False, False, False, False], [False, False, True, True]])
    velocity = core.predict_inpainting_velocity(
        _actions(), torch.tensor([0.2, 0.8]), action_is_pad, ~action_is_pad
    )

    assert velocity.shape == (2, _HORIZON, _ACTION_DIM)
    assert torch.isfinite(velocity).all()
    allowed = captured["attention_mask"][:, 0].eq(0)
    assert allowed[:, :_PROMPTS, :_PROMPTS].all()
    assert not allowed[:, :_PROMPTS, _PROMPTS:].any()
    assert allowed[0, _PROMPTS:, :].all()
    assert allowed[1, _PROMPTS : _PROMPTS + 2, : _PROMPTS + 2].all()
    assert not allowed[1, -2:, :].any()
    assert not allowed[1, :, -2:].any()
    torch.testing.assert_close(
        captured["position_ids"], torch.arange(_PROMPTS + _HORIZON).unsqueeze(0).expand(2, -1)
    )
    torch.testing.assert_close(
        captured["inputs_embeds"][1][:, :_PROMPTS], core.prompt_tokens.weight.unsqueeze(0).expand(2, -1, -1)
    )


@torch.no_grad()
def test_action_changes_cannot_change_vlm_or_prompt_hidden_states():
    core = PI05Policy(_config()).model.eval()
    prefix, prefix_pad, prefix_att = core.embed_prefix(**_inputs())
    time = torch.tensor([0.3, 0.7])

    def forward_hidden_states(actions):
        suffix, suffix_pad, suffix_att, condition = core.embed_suffix(actions, time)
        pad = torch.cat([prefix_pad, suffix_pad], dim=1)
        attention = modeling_pi05.make_att_2d_masks(pad, torch.cat([prefix_att, suffix_att], dim=1))
        outputs, _ = core.paligemma_with_expert(
            attention_mask=core._prepare_attention_masks_4d(attention),
            position_ids=pad.cumsum(dim=1) - 1,
            inputs_embeds=[prefix, suffix],
            adarms_cond=[None, condition],
            use_cache=False,
        )
        return outputs

    first_prefix, first_suffix = forward_hidden_states(_actions())
    second_prefix, second_suffix = forward_hidden_states(_actions() * 10)

    torch.testing.assert_close(first_prefix[prefix_pad], second_prefix[prefix_pad], rtol=0, atol=0)
    torch.testing.assert_close(first_suffix[:, :_PROMPTS], second_suffix[:, :_PROMPTS], rtol=0, atol=0)
    assert not torch.allclose(first_suffix[:, _PROMPTS:], second_suffix[:, _PROMPTS:])


@pytest.mark.parametrize("stage", ["flow", "next_action"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_prompt_and_action_gradients_update_without_changing_vlm(stage, checkpointing):
    policy = PI05Policy(
        _config(
            training_stage=stage,
            gradient_checkpointing=checkpointing,
            freeze_vision_encoder=False,
            train_expert_only=False,
        )
    )
    policy.eval()
    policy.train()
    core = policy.model
    vlm = core.paligemma_with_expert.paligemma
    assert all(not module.training for module in vlm.modules())
    assert all(not parameter.requires_grad for parameter in vlm.parameters())
    assert core.paligemma_with_expert.gemma_expert.training
    frozen_before = {name: tensor.clone() for name, tensor in vlm.state_dict().items()}
    prompt_before = core.prompt_tokens.weight.detach().clone()
    expert_weight = core.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.v_proj.weight
    expert_before = expert_weight.detach().clone()
    optimizer_parameters = policy.get_optim_params()
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    assert id(core.prompt_tokens.weight) in optimizer_ids
    assert not optimizer_ids.intersection(id(parameter) for parameter in vlm.parameters())
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=0.01)
    actions = _actions()
    losses = core(
        **_inputs(),
        actions=actions,
        noise=torch.randn_like(actions),
        time=torch.tensor([0.2, 0.8]),
        action_is_pad=torch.zeros(2, _HORIZON, dtype=torch.bool),
    )
    assert losses.shape == actions.shape
    losses.mean().backward()

    for parameter in [core.prompt_tokens.weight, expert_weight, core.action_out_proj.weight]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() > 0
    assert all(parameter.grad is None for parameter in vlm.parameters())
    optimizer.step()
    assert not torch.equal(core.prompt_tokens.weight, prompt_before)
    assert not torch.equal(expert_weight, expert_before)
    for name, value in vlm.state_dict().items():
        torch.testing.assert_close(value, frozen_before[name], rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["flow", "next_action"])
@pytest.mark.parametrize("ratio", [1.0, 2.0])
def test_cabo_bounds_real_action_branch_updates_by_prompt_updates_in_both_stages(stage, ratio):
    policy = PI05Policy(_config(training_stage=stage, cabo_enabled=True, cabo_prompt_update_ratio=ratio))
    policy.train()
    core = policy.model
    vlm = core.paligemma_with_expert.paligemma
    frozen_before = {name: tensor.clone() for name, tensor in vlm.state_dict().items()}
    groups = policy.get_optim_params()
    assert [group[CABO_GROUP_NAME] for group in groups] == [
        CABO_PROMPT_GROUP,
        CABO_ACTION_EXPERT_GROUP,
        CABO_ACTION_PROJECTION_GROUP,
    ]
    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    assert groups[0]["params"] == [core.prompt_tokens.weight]
    assert not set(grouped_ids).intersection(id(parameter) for parameter in vlm.parameters())
    # Force both action groups above the prompt budget so the test exercises the cap.
    groups[0]["lr"] = 0.0001
    optimizer = torch.optim.AdamW(groups, lr=0.01, weight_decay=0.0, foreach=False)
    policy.validate_optimizer_step_control(optimizer)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        actions = _actions()
        core(
            **_inputs(),
            actions=actions,
            noise=torch.randn_like(actions),
            time=torch.tensor([0.2, 0.8]),
            action_is_pad=torch.zeros(2, _HORIZON, dtype=torch.bool),
        ).mean().backward()
        assert core.prompt_tokens.weight.grad is not None
        assert core.prompt_tokens.weight.grad.count_nonzero() > 0
        assert all(parameter.grad is None for parameter in vlm.parameters())
        active_groups = [
            [parameter for parameter in group["params"] if parameter.grad is not None]
            for group in optimizer.param_groups
        ]
        before = [[parameter.detach().clone() for parameter in group] for group in active_groups]
        control = policy.compute_optimizer_step_control({}, optimizer, SimpleNamespace(num_processes=1))
        assert not control.skip_optimizer_step
        assert all(0.0 <= scale < 1.0 for scale in control.group_scales.values())
        with temporary_optimizer_group_lr_scales(optimizer, control.group_scales):
            optimizer.step()

        delta_norms = []
        parameter_norms = []
        for parameters, previous in zip(active_groups, before, strict=True):
            delta_norms.append(
                sum(
                    (parameter.detach().double() - old.double()).square().sum().item()
                    for parameter, old in zip(parameters, previous, strict=True)
                )
            )
            parameter_norms.append(sum(old.double().square().sum().item() for old in previous))
        rates = [(delta / norm) ** 0.5 for delta, norm in zip(delta_norms, parameter_norms, strict=True)]
        assert rates[0] > 0.0
        assert rates[0] >= ratio * max(rates[1:]) - 1e-7
        whole_action_rate = (sum(delta_norms[1:]) / sum(parameter_norms[1:])) ** 0.5
        assert rates[0] >= ratio * whole_action_rate - 1e-7

    for name, value in vlm.state_dict().items():
        torch.testing.assert_close(value, frozen_before[name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("num_prompts", [0, _PROMPTS])
@torch.no_grad()
def test_cached_denoising_matches_full_forward(dtype, num_prompts):
    core = PI05Policy(_config(dtype=dtype, num_prompt_tokens=num_prompts)).model.eval()
    inputs = _inputs()
    actions = _actions()
    time = torch.tensor([0.25, 0.75])
    full_velocity = core.predict_velocity(**inputs, x_t=actions, time=time)
    prefix, prefix_pad, prefix_att = core.embed_prefix(**inputs)
    _, cache = core.paligemma_with_expert(
        attention_mask=core._prepare_attention_masks_4d(
            modeling_pi05.make_att_2d_masks(prefix_pad, prefix_att)
        ),
        position_ids=prefix_pad.cumsum(dim=1) - 1,
        inputs_embeds=[prefix, None],
        use_cache=True,
    )
    prefix_length = cache.get_seq_length()
    cached_velocity = core.denoise_step(prefix_pad, cache, actions, time)
    second_velocity = core.denoise_step(prefix_pad, cache, actions, time)

    assert cached_velocity.shape == actions.shape
    tolerance = 0.02 if dtype == "bfloat16" else 1e-5
    torch.testing.assert_close(cached_velocity, full_velocity, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(second_velocity, cached_velocity, rtol=0, atol=0)
    assert cache.get_seq_length() == prefix_length


@torch.no_grad()
def test_same_noise_sampling_changes_with_prompts_and_preserves_vlm_context():
    core = PI05Policy(_config()).model.eval()
    inputs = _inputs()
    noise = _actions()
    first_actions, first_context = core.sample_actions(
        **inputs, noise=noise.clone(), num_steps=2, return_prefix_features=True
    )
    core.prompt_tokens.weight.copy_(torch.randn_like(core.prompt_tokens.weight))
    second_actions, second_context = core.sample_actions(
        **inputs, noise=noise.clone(), num_steps=2, return_prefix_features=True
    )

    assert first_actions.shape == noise.shape
    assert torch.isfinite(second_actions).all()
    assert not torch.allclose(first_actions, second_actions)
    torch.testing.assert_close(first_context, second_context, rtol=0, atol=0)


def test_legacy_checkpoint_initializes_only_missing_prompts(tmp_path):
    policy = PI05Policy(_config(num_prompt_tokens=0))
    legacy_state = {key: value for key, value in policy.state_dict().items() if key != _PROMPT_KEY}
    checkpoint = tmp_path / "legacy"
    _save_checkpoint(checkpoint, policy, legacy_state)

    loaded = PI05Policy.from_pretrained(checkpoint, config=_config(), strict=True)

    assert loaded.model.prompt_tokens.weight.shape == (_PROMPTS, _EXPERT_WIDTH)
    assert loaded.model.prompt_tokens.weight.requires_grad
    assert torch.isfinite(loaded.model.prompt_tokens.weight).all()
    assert loaded.model.prompt_tokens.weight.count_nonzero() > 0
    for key, value in legacy_state.items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)


def test_prompt_checkpoint_roundtrips_across_training_stages(tmp_path):
    policy = PI05Policy(_config(training_stage="next_action"))
    with torch.no_grad():
        policy.model.prompt_tokens.weight.copy_(torch.randn_like(policy.model.prompt_tokens.weight))
    checkpoint = tmp_path / "prompts"
    _save_checkpoint(checkpoint, policy)

    loaded = PI05Policy.from_pretrained(checkpoint, config=_config(training_stage="flow"), strict=True)

    for key, value in policy.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    assert loaded.model.prompt_tokens.weight.requires_grad
    assert all(
        not parameter.requires_grad for parameter in loaded.model.paligemma_with_expert.paligemma.parameters()
    )


def test_prompt_checkpoint_shape_mismatch_fails(tmp_path):
    policy = PI05Policy(_config())
    checkpoint = tmp_path / "wrong_prompt_count"
    _save_checkpoint(checkpoint, policy)

    with pytest.raises(RuntimeError, match="size mismatch for model.prompt_tokens.weight"):
        PI05Policy.from_pretrained(checkpoint, config=_config(num_prompt_tokens=_PROMPTS + 1), strict=True)


def test_legacy_checkpoint_still_rejects_missing_action_weights(tmp_path):
    policy = PI05Policy(_config())
    state_dict = {
        key: value
        for key, value in policy.state_dict().items()
        if key not in {_PROMPT_KEY, "model.action_out_proj.weight"}
    }
    checkpoint = tmp_path / "incomplete"
    _save_checkpoint(checkpoint, policy, state_dict)

    with pytest.raises(RuntimeError, match="model.action_out_proj.weight"):
        PI05Policy.from_pretrained(checkpoint, config=_config(), strict=True)


def test_default_peft_targets_include_action_expert_and_exclude_vlm():
    policy = PI05Policy(_config())
    targets = policy._get_default_peft_targets()
    matched = {name for name, _ in policy.named_modules() if re.fullmatch(targets["target_modules"], name)}

    assert "model.paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj" in matched
    assert "model.paligemma_with_expert.gemma_expert.model.layers.1.self_attn.v_proj" in matched
    assert "model.action_out_proj" in matched
    assert all("paligemma_with_expert.paligemma." not in name for name in matched)
    assert targets["modules_to_save"] == ["model.prompt_tokens"]


@pytest.mark.parametrize("existing", [None, ["model.action_out_proj"], ["model.prompt_tokens"]])
def test_custom_peft_config_preserves_prompt_checkpoint(existing):
    policy = PI05Policy(_config(pretrained_path="test-base"))
    config = SimpleNamespace(modules_to_save=existing)

    policy._validate_peft_config(config)

    assert config.modules_to_save.count("model.prompt_tokens") == 1
    for module in existing or []:
        assert module in config.modules_to_save


def test_real_peft_trains_and_saves_prompt_while_freezing_vlm_adapters(tmp_path):
    peft = pytest.importorskip("peft")
    policy = PI05Policy(_config(pretrained_path=tmp_path / "base"))
    # Deliberately broad custom targets also insert adapters into the VLM.
    adapter_config = peft.LoraConfig(r=2, target_modules=["q_proj", "v_proj"])
    wrapped = policy.wrap_with_peft(adapter_config)
    wrapped.train()
    core = policy.model
    vlm = core.paligemma_with_expert.paligemma
    assert any("lora_" in name for name, _ in vlm.named_parameters())
    assert all(not parameter.requires_grad for parameter in vlm.parameters())
    assert all(not module.training for module in vlm.modules())

    actions = _actions()
    core(
        **_inputs(),
        actions=actions,
        noise=torch.randn_like(actions),
        time=torch.tensor([0.2, 0.8]),
    ).mean().backward()

    active_prompt = core.prompt_tokens.modules_to_save["default"].weight
    assert active_prompt.requires_grad
    assert active_prompt.grad is not None
    assert torch.isfinite(active_prompt.grad).all()
    assert active_prompt.grad.count_nonzero() > 0
    assert all(parameter.grad is None for parameter in vlm.parameters())
    adapter_path = tmp_path / "adapter"
    wrapped.save_pretrained(adapter_path, save_embedding_layers=False)
    weights = load_file(adapter_path / "adapter_model.safetensors")
    saved_prompts = [value for key, value in weights.items() if key.endswith("prompt_tokens.weight")]
    assert len(saved_prompts) == 1
    torch.testing.assert_close(saved_prompts[0], active_prompt, rtol=0, atol=0)
