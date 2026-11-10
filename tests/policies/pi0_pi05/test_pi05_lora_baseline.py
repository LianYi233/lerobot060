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

"""Exercise the prompt-free external LoRA baseline with real, tiny CPU PiGemma models."""

import copy
import re
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from safetensors.torch import load_file  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi05 import PI05Config, PI05Policy, modeling_pi05  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05_LORA_TARGET_MODULES  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402

_VLM_WIDTH = 32
_EXPERT_WIDTH = 16
_HORIZON = 4
_ACTION_DIM = 3


@pytest.fixture(autouse=True)
def _tiny_real_pi05(monkeypatch):
    """Keep real attention, PEFT, checkpointing, and cache implementations, not a fake LoRA."""
    original_paligemma = modeling_pi05.PaliGemmaForConditionalGenerationWithPiGemma
    original_expert = modeling_pi05.PiGemmaForCausalLM
    matmul_precision = torch.get_float32_matmul_precision()
    cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32

    def tiny_paligemma(config):
        config._vocab_size = 32
        config.image_token_index = 31
        config.text_config.vocab_size = 32
        config.vision_config.hidden_size = 16
        config.vision_config.intermediate_size = 32
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 4
        config.vision_config.patch_size = 8
        config.vision_config.projection_dim = _VLM_WIDTH
        return original_paligemma(config)

    def tiny_expert(config):
        config.vocab_size = 32
        return original_expert(config)

    monkeypatch.setattr(
        modeling_pi05,
        "get_gemma_config",
        lambda variant: modeling_pi05.GemmaConfig(
            width=_VLM_WIDTH if variant == "gemma_2b" else _EXPERT_WIDTH,
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
    yield
    torch.set_float32_matmul_precision(matmul_precision)
    torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32


def _config(**overrides):
    options = {
        "device": "cpu",
        "dtype": "float32",
        "adaptation_mode": "lora",
        "num_vlm_prompt_tokens": 0,
        "num_prompt_tokens": 0,
        "next_action_pretrain_steps": 0,
        "next_action_bridge_steps": 0,
        "cabo_enabled": False,
        "chunk_size": _HORIZON,
        "n_action_steps": _HORIZON,
        "max_action_dim": _ACTION_DIM,
        "image_resolution": (16, 16),
        "pretrained_path": "test-base",
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,))},
    }
    options.update(overrides)
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_stage", "next_action"),
        ("num_vlm_prompt_tokens", 1),
        ("num_prompt_tokens", 1),
        ("next_action_pretrain_steps", 1),
        ("next_action_bridge_steps", 1),
        ("cabo_enabled", True),
        ("action_prompt_lr_scale", 0.5),
        ("clip_action_head_by_vlm", True),
    ],
)
def test_lora_config_rejects_prompt_training_protocol(field, value):
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


def test_lora_configuration_enables_vlm_autograd_without_unfreezing_vision():
    config = _config(separate_frozen_observations=True, train_expert_only=True, freeze_vision_encoder=False)
    assert config.separate_frozen_observations is False
    assert config.train_expert_only is False
    assert config.freeze_vision_encoder is True
    assert not config.next_action_pretraining_active
    assert not config.optimizer_step_control_active
    with pytest.raises(ValueError, match="adaptation_mode"):
        _config(adaptation_mode="unknown")


def test_default_prompt_configuration_is_unchanged():
    config = PI05Config(device="cpu", freeze_vision_encoder=False, train_expert_only=False)
    assert config.adaptation_mode == "prompt"
    assert config.num_vlm_prompt_tokens == config.num_prompt_tokens == 16
    assert config.cabo_enabled and config.next_action_pretraining_active
    assert config.freeze_vision_encoder and config.train_expert_only
    policy = PI05Policy(config)
    policy.train()
    assert {name for name, param in policy.named_parameters() if param.requires_grad} == {
        "model.vlm_prompt_tokens.weight",
        "model.prompt_tokens.weight",
    }


def test_lora_requires_attached_adapters_and_uses_only_exact_transformer_targets():
    policy = PI05Policy(_config())
    with pytest.raises(ValueError, match="non-empty active LoRA"):
        policy.get_optim_params()

    targets = {
        name: module
        for name, module in policy.named_modules()
        if re.fullmatch(PI05_LORA_TARGET_MODULES, name)
    }
    assert len(targets) == 2 * 2 * 7  # two stacks, two tiny layers, seven Linear projections
    assert all(isinstance(module, torch.nn.Linear) for module in targets.values())
    expected_count = sum(16 * (module.in_features + module.out_features) for module in targets.values())
    wrapped = policy.wrap_with_peft()
    actual_targets = {name for name, module in policy.named_modules() if hasattr(module, "lora_A")}
    assert actual_targets == set(targets)
    assert wrapped.peft_config["default"].r == wrapped.peft_config["default"].lora_alpha == 16
    assert not wrapped.peft_config["default"].modules_to_save
    assert policy.config.use_peft
    assert sum(param.numel() for param in wrapped.parameters() if param.requires_grad) == expected_count
    assert all(
        ".lora_A." in name or ".lora_B." in name
        for name, param in wrapped.named_parameters()
        if param.requires_grad
    )
    assert policy.model.vlm_prompt_tokens.weight.numel() == policy.model.prompt_tokens.weight.numel() == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_modules": ["q_proj", "v_proj"]},
        {"modules_to_save": ["model.action_out_proj"]},
        {"bias": "all"},
        {"lora_dropout": 0.1},
        {"init_lora_weights": False},
        {"use_dora": True},
        {"use_rslora": True},
        {"rank_pattern": {"q_proj": 8}},
        {"alpha_pattern": {"q_proj": 8}},
    ],
)
def test_lora_rejects_silent_changes_to_baseline_targets_or_extra_parameters(overrides):
    policy = PI05Policy(_config())
    kwargs = policy._get_default_peft_targets() | overrides
    with pytest.raises(ValueError, match="PI05 LoRA"):
        policy.wrap_with_peft(peft.LoraConfig(**kwargs))


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_real_flow_backprop_updates_both_lora_stacks_only(monkeypatch, checkpointing, dtype):
    policy = PI05Policy(_config(gradient_checkpointing=checkpointing, dtype=dtype))
    inputs, actions, time = _inputs(), _actions(), torch.tensor([0.2, 0.8])
    policy.eval()
    with torch.no_grad():
        base_velocity = policy.model.predict_velocity(**inputs, x_t=actions, time=time)
    wrapped = policy.wrap_with_peft()
    with torch.no_grad():
        initial_velocity = policy.model.predict_velocity(**inputs, x_t=actions, time=time)
    torch.testing.assert_close(initial_velocity, base_velocity, rtol=0, atol=0)

    # An external unfreeze followed by train() must not leak into any base weights.
    wrapped.requires_grad_(True)
    wrapped.train()
    core = policy.model
    assert core.paligemma_with_expert.paligemma.training
    assert core.paligemma_with_expert.gemma_expert.training
    assert not core.paligemma_with_expert.paligemma.model.vision_tower.training
    before = {name: param.detach().clone() for name, param in wrapped.named_parameters()}
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=0.01, weight_decay=0.0)
    optimizer_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    assert optimizer_ids == {id(param) for param in wrapped.parameters() if param.requires_grad}
    monkeypatch.setattr(policy, "_preprocess_images", lambda batch: (inputs["images"], inputs["img_masks"]))
    batch = {
        ACTION: actions,
        "action_is_pad": torch.zeros(2, _HORIZON, dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: inputs["tokens"],
        OBS_LANGUAGE_ATTENTION_MASK: inputs["masks"],
    }
    loss, metrics = wrapped(batch)
    assert torch.isfinite(loss)
    assert "stage1/bridge_active" not in metrics
    loss.backward()
    for name, param in wrapped.named_parameters():
        if not param.requires_grad:
            assert param.grad is None, name
        elif param.grad is not None:
            assert torch.isfinite(param.grad).all(), name
    # The final VLM layer's o_proj/MLP do not feed action loss. Check reachable early
    # layers of each stack, rather than claiming that every targeted matrix updates.
    for stack in (
        core.paligemma_with_expert.paligemma.model.language_model,
        core.paligemma_with_expert.gemma_expert.model,
    ):
        grad = stack.layers[0].self_attn.v_proj.lora_B["default"].weight.grad
        assert grad is not None and grad.count_nonzero() > 0
    control = policy.compute_optimizer_step_control({}, optimizer, SimpleNamespace(num_processes=1))
    assert not control.group_scales and not control.skip_optimizer_step
    optimizer.step()
    changed = []
    for name, param in wrapped.named_parameters():
        if not param.requires_grad:
            torch.testing.assert_close(param, before[name], rtol=0, atol=0, msg=name)
        elif not torch.equal(param, before[name]):
            changed.append(name)
    assert any("paligemma.model.language_model" in name for name in changed)
    assert any("gemma_expert.model" in name for name in changed)

    core.eval()
    with torch.no_grad():
        full_velocity = core.predict_velocity(**inputs, x_t=actions, time=time)
        prefix, prefix_pad, _ = core.embed_prefix(**inputs)
        _, cache = core.paligemma_with_expert(
            attention_mask=core._prepare_attention_masks_4d(
                modeling_pi05.make_pi05_att_2d_masks(
                    prefix_pad,
                    prefix_length=prefix.shape[1],
                    num_vlm_prompt_tokens=0,
                    num_prompt_tokens=0,
                )
            ),
            position_ids=prefix_pad.cumsum(dim=1) - 1,
            inputs_embeds=[prefix, None],
            use_cache=True,
        )
        cached_velocity = core.denoise_step(prefix_pad, cache, actions, time)
    tolerance = 0.02 if dtype == "bfloat16" else 1e-5
    torch.testing.assert_close(cached_velocity, full_velocity, rtol=tolerance, atol=tolerance)


def test_freezing_preserves_only_the_active_adapter():
    policy = PI05Policy(_config())
    wrapped = policy.wrap_with_peft()
    wrapped.add_adapter("other", copy.deepcopy(wrapped.peft_config["default"]))
    for active, inactive in (("default", "other"), ("other", "default")):
        wrapped.set_adapter(active)
        wrapped.requires_grad_(True)
        wrapped.train()
        trainable = {name for name, param in wrapped.named_parameters() if param.requires_grad}
        assert trainable
        assert all(f".{active}." in name and ".lora_" in name for name in trainable)
        assert all(f".{inactive}." not in name for name in trainable)
        assert {id(param) for param in policy.get_optim_params()} == {
            id(param) for param in wrapped.parameters() if param.requires_grad
        }


def test_disabled_adapter_is_not_reenabled_by_training_mode():
    policy = PI05Policy(_config())
    wrapped = policy.wrap_with_peft()
    active_ids = {id(param) for param in policy.get_optim_params()}
    with wrapped.disable_adapter():
        wrapped.train()
        assert not any(param.requires_grad for param in wrapped.parameters())
    wrapped.train()
    assert {id(param) for param in policy.get_optim_params()} == active_ids


@pytest.mark.parametrize("is_trainable", [True, False])
def test_native_peft_checkpoint_roundtrip_preserves_predictions_and_trainability(tmp_path, is_trainable):
    base_path, adapter_path = tmp_path / "base", tmp_path / "adapter"
    policy = PI05Policy(_config(pretrained_path=base_path))
    # The production base checkpoint uses the same vocabulary weights in these aliases.
    paligemma = policy.model.paligemma_with_expert.paligemma
    with torch.no_grad():
        paligemma.lm_head.weight.copy_(paligemma.model.language_model.embed_tokens.weight)
    policy.save_pretrained(base_path)
    wrapped = policy.wrap_with_peft()
    wrapped.train()
    inputs, actions, time = _inputs(), _actions(), torch.tensor([0.25, 0.75])
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=0.01, weight_decay=0.0)
    policy.model(**inputs, actions=actions, noise=_actions(), time=time).mean().backward()
    optimizer.step()
    wrapped.eval()
    with torch.no_grad():
        expected = policy.model.predict_velocity(**inputs, x_t=actions, time=time)
    wrapped.save_pretrained(adapter_path)
    policy.config.save_pretrained(adapter_path)  # The trainer writes this alongside native PEFT output.
    weights = load_file(adapter_path / "adapter_model.safetensors")
    assert weights
    assert all(".lora_A." in name or ".lora_B." in name for name in weights)
    assert all("prompt_tokens" not in name and "vision_tower" not in name for name in weights)

    fresh = PI05Policy.from_pretrained(base_path, config=_config(pretrained_path=base_path))
    adapter_config = peft.PeftConfig.from_pretrained(adapter_path)
    fresh._validate_peft_config(adapter_config)
    assert adapter_config.base_model_name_or_path == str(base_path)
    reloaded = peft.PeftModel.from_pretrained(
        fresh, adapter_path, config=adapter_config, is_trainable=is_trainable
    )
    fresh.model._freeze_backbones()
    reloaded.eval()
    with torch.no_grad():
        actual = fresh.model.predict_velocity(**inputs, x_t=actions, time=time)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reloaded.train()
    if is_trainable:
        assert {id(param) for param in fresh.get_optim_params()} == {
            id(param) for param in reloaded.parameters() if param.requires_grad
        }
    else:
        assert not any(param.requires_grad for param in reloaded.parameters())
        with pytest.raises(ValueError, match="non-empty active LoRA"):
            fresh.get_optim_params()
