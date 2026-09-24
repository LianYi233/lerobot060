"""Read-only PI0.5 attention diagnostics for single-environment evaluation.

Capture post-RoPE queries and cached keys at one Gemma layer. The original
attention backend still computes the policy output; only selected query rows
are recomputed in float32 for visualization. No gradients or extra policy
forward passes are used.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class AttentionVideoConfig:
    source: str = "action"
    layer: int = -1
    camera: int = 0
    denoise: str = "mean"
    alpha: float = 0.55
    vmax: float = 0.0  # zero: relative to the maximum in this prediction's map

    def __post_init__(self):
        if self.source not in {"action", "vlm_prompt"}:
            raise ValueError("ATTENTION_SOURCE must be action or vlm_prompt")
        if self.denoise not in {"mean", "first", "last"}:
            raise ValueError("ATTENTION_DENOISE must be mean, first, or last")
        if self.camera < 0:
            raise ValueError("ATTENTION_CAMERA must be a non-negative camera index")
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("ATTENTION_ALPHA must be in [0, 1]")
        if not math.isfinite(self.vmax) or self.vmax < 0:
            raise ValueError("ATTENTION_VMAX must be finite and non-negative")

    @classmethod
    def from_env(cls):
        return cls(
            source=os.environ.get("ATTENTION_SOURCE", "action"),
            layer=int(os.environ.get("ATTENTION_LAYER", "-1")),
            camera=int(os.environ.get("ATTENTION_CAMERA", "0")),
            denoise=os.environ.get("ATTENTION_DENOISE", "mean"),
            alpha=float(os.environ.get("ATTENTION_ALPHA", "0.55")),
            vmax=float(os.environ.get("ATTENTION_VMAX", "0")),
        )


@torch.no_grad()
def selected_attention(query, key, mask, scaling, start, stop):
    """Average selected query rows and heads, with softmax over ALL key tokens."""
    if not 0 <= start < stop <= query.shape[-2]:
        raise ValueError(f"Invalid query range [{start}, {stop}) for {query.shape[-2]} tokens")
    query = query.detach()[:, :, start:stop].float()
    key = key.detach().float()
    if query.shape[1] % key.shape[1]:
        raise ValueError("Query head count must be divisible by KV head count")
    key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
    logits = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if mask is None or mask.ndim != 4:
        raise ValueError("PI0.5 visualization requires its explicit 4D visibility mask")
    selected_mask = mask[..., start:stop, : key.shape[-2]]
    if selected_mask.dtype == torch.bool:
        allowed = selected_mask
        logits = logits.masked_fill(~allowed, -torch.inf)
    else:
        # PI0.5 uses a finite float32 negative sentinel for disallowed positions.
        allowed = selected_mask > -1e4
        logits = logits + selected_mask.float()
    if not allowed.any(dim=-1).all():
        raise ValueError("Selected attention queries include a fully masked/padded row")
    weights = torch.softmax(logits, dim=-1).masked_fill(~allowed, 0)
    if not torch.isfinite(weights).all():
        raise ValueError("Non-finite attention probabilities")
    return weights.mean(dim=(1, 2))


def _diagnostic_attention(module, query, key, value, attention_mask, **kwargs):
    recorder = module._pi05_attention_recorder
    recorder.observe(query, key, attention_mask, kwargs.get("scaling", module.scaling))
    return module._pi05_attention_original_backend(module, query, key, value, attention_mask, **kwargs)


def overlay_attention(rgb, patch_map, alpha=0.55, vmax=0.0):
    """Red/orange overlay; zero stays transparent and uniform maps stay uniform."""
    patch_map = np.asarray(patch_map, dtype=np.float32)
    if not np.isfinite(patch_map).all() or (patch_map < 0).any():
        raise ValueError("Attention map must contain finite non-negative probabilities")
    maximum = float(vmax or patch_map.max())
    if maximum <= 0:
        return rgb.copy()
    intensity = np.clip(patch_map / maximum, 0, 1)
    heat = np.asarray(
        Image.fromarray(intensity).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR)
    )
    # A monotone warm palette; opacity also increases with attention.
    color = np.stack([255 - 90 * heat, 215 * (1 - heat), 120 * (1 - heat)], axis=-1)
    opacity = (alpha * heat)[..., None]
    return np.clip(rgb * (1 - opacity) + color * opacity, 0, 255).astype(np.uint8)


class PI05AttentionRecorder:
    """Temporary instrumentation. Restore all model methods/configs on exit."""

    def __init__(self, policy, config: AttentionVideoConfig):
        self.eval_policy = policy
        self.policy = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
        self.config = config
        self._restore = []
        self.reset_episode()

    def _set(self, obj, name, value):
        self._restore.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, value)

    def __enter__(self):
        try:
            self._attach()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for obj, name, existed, old in reversed(self._restore):
            if existed:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        self._restore.clear()

    def _attach(self):
        from transformers import AttentionInterface
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.gemma.modeling_gemma import eager_attention_forward

        policy = self.policy
        if getattr(policy, "name", None) != "pi05":
            raise ValueError("Attention videos currently support PI0.5 only")
        if policy.config.compile_model or policy._rtc_enabled():
            raise ValueError("Attention videos require compile_model=false and RTC disabled")
        if self.config.source == "vlm_prompt" and not policy.config.num_vlm_prompt_tokens:
            raise ValueError("This checkpoint has no VLM prompts; use ATTENTION_SOURCE=action")
        model = policy.model
        composite = model.paligemma_with_expert
        decoder = (
            composite.gemma_expert.model
            if self.config.source == "action"
            else composite.paligemma.model.language_model
        )
        self.layer = self.config.layer
        if self.layer < 0:
            self.layer += len(decoder.layers)
        if not 0 <= self.layer < len(decoder.layers):
            raise ValueError(f"ATTENTION_LAYER out of range for {len(decoder.layers)} layers")
        attention = decoder.layers[self.layer].self_attn
        implementation = attention.config._attn_implementation
        if implementation not in {"sdpa", "eager"}:
            raise ValueError(f"Unsupported attention backend for diagnostics: {implementation}")
        backend = ALL_ATTENTION_FUNCTIONS.get_interface(implementation, eager_attention_forward)
        AttentionInterface.register("pi05_eval_visualization", _diagnostic_attention)
        config = copy.copy(attention.config)
        config._attn_implementation = "pi05_eval_visualization"
        self._set(attention, "config", config)
        self._set(attention, "_pi05_attention_original_backend", backend)
        self._set(attention, "_pi05_attention_recorder", self)
        self.patch_size = composite.paligemma.config.vision_config.patch_size

        original_prefix = model.embed_prefix
        original_select = policy.select_action

        def embed_prefix(images, img_masks, tokens, masks):
            result = original_prefix(images, img_masks, tokens, masks)
            self.begin_prediction(images, img_masks, masks, result[0].shape[1])
            return result

        def select_action(batch):
            self.step += 1
            self.new_prediction = False
            image_keys = [key for key in policy.config.image_features if key in batch]
            image_keys += [key for key in policy.config.image_features if key not in batch]
            self.camera_feature = (
                image_keys[self.config.camera] if self.config.camera < len(image_keys) else None
            )
            result = original_select(batch)
            if self.new_prediction:
                self.finish_prediction()
            return result

        self._set(model, "embed_prefix", embed_prefix)
        self._set(policy, "select_action", select_action)
        self._set(self.eval_policy, "_eval_attention_visualizer", self)

    def reset_episode(self):
        self.step = -1
        self.prediction_step = -1
        self.new_prediction = False
        self.latest_map = None
        self.image = None
        self.maps = []
        self.steps = []
        self.counts = []
        self.rows = []

    def begin_prediction(self, images, img_masks, language_mask, prefix_length):
        if images[0].shape[0] != 1:
            raise ValueError("Attention videos require eval.batch_size=1")
        index = self.config.camera
        if index >= len(images) or not bool(img_masks[index][0]):
            raise ValueError(f"Camera {index} is missing/padded; choose a present policy camera")
        sizes = []
        for image in images:
            height, width = image.shape[-2:]
            if height % self.patch_size or width % self.patch_size:
                raise ValueError("Image dimensions must be multiples of vision patch size")
            sizes.append((height // self.patch_size, width // self.patch_size))
        lengths = [h * w for h, w in sizes]
        expected = sum(lengths) + language_mask.shape[1] + self.policy.config.num_vlm_prompt_tokens
        if prefix_length != expected:
            raise ValueError("Unexpected image-token layout; refusing to misalign attention and pixels")
        self.prefix_length = prefix_length
        self.grid = sizes[index]
        self.image_start = sum(lengths[:index])
        self.image_stop = self.image_start + lengths[index]
        # These are the EXACT oriented/resized/padded pixels passed to SigLIP.
        self.image = (
            ((images[index][0].detach().float().cpu().permute(1, 2, 0).numpy() + 1) * 127.5)
            .clip(0, 255)
            .round()
            .astype(np.uint8)
        )
        self.prediction_step = self.step
        self.rows = []
        self.new_prediction = True

    def observe(self, query, key, mask, scaling):
        if not self.new_prediction:
            raise RuntimeError("Attention captured outside an action prediction")
        if self.config.source == "action":
            start = self.policy.config.num_prompt_tokens
            stop = start + min(self.policy.config.n_action_steps, self.policy.config.chunk_size)
            if key.shape[-2] != self.prefix_length + query.shape[-2]:
                raise ValueError("Expected image/language prefix followed by cached action suffix")
        else:
            start = self.prefix_length - self.policy.config.num_vlm_prompt_tokens
            stop = self.prefix_length
        probabilities = selected_attention(query, key, mask, scaling, start, stop)
        # Softmax includes language, prompts, other cameras, and action tokens.
        # Do not renormalize on just the chosen camera.
        self.rows.append(probabilities[0, self.image_start : self.image_stop].cpu().numpy())

    def finish_prediction(self):
        if not self.rows:
            raise RuntimeError("No attention was captured; refusing to generate an empty heatmap video")
        if self.config.denoise == "first":
            values = self.rows[0]
        elif self.config.denoise == "last":
            values = self.rows[-1]
        else:
            values = np.mean(self.rows, axis=0)
        self.latest_map = values.reshape(self.grid).copy()
        self.maps.append(self.latest_map)
        self.steps.append(self.prediction_step)
        self.counts.append(len(self.rows))
        self.new_prediction = False

    def metadata(self):
        return {
            **asdict(self.config),
            "camera_feature": self.camera_feature,
            "resolved_layer_zero_based": self.layer,
            "query_tokens": "executed action-token prefix"
            if self.config.source == "action"
            else "VLM prompts",
            "action_query_count": min(self.policy.config.n_action_steps, self.policy.config.chunk_size),
            "heads": "all, arithmetic mean",
            "denoise_note": "VLM prefix is evaluated once per prediction"
            if self.config.source == "vlm_prompt"
            else "flow denoising passes",
            "normalization": "softmax over all visible keys before camera selection",
            "display_scale": "fixed" if self.config.vmax else "relative to prediction maximum",
            "pixel_alignment": "exact oriented/resized/padded policy input; no additional flip",
            "time_alignment": "map held on its source image during queued actions; live view is separate",
            "interpretation": "qualitative attention routing, not segmentation, causal attribution, or proof of effectiveness",
        }

    def render_frames(self, frames):
        if len(frames) != 1:
            raise ValueError("Attention rendering requires a single environment")
        side, header = 256, 24
        canvas = Image.new("RGB", (side * 3, 320), "white")
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 12)
        except OSError:
            font = ImageFont.load_default()
        live = Image.fromarray(frames[0]).resize((side, side), Image.Resampling.BILINEAR)
        canvas.paste(live, (0, header))
        for column, title in enumerate(
            ("Live environment", "Policy input (source frame)", "Attention overlay")
        ):
            draw.text((column * side + 8, 5), title, fill="#282828", font=font)
        if self.latest_map is None:
            draw.text((side + 12, 130), "Waiting for first prediction", fill="#666666", font=font)
        else:
            overlay = overlay_attention(self.image, self.latest_map, self.config.alpha, self.config.vmax)
            for column, rgb in ((1, self.image), (2, overlay)):
                canvas.paste(
                    Image.fromarray(rgb).resize((side, side), Image.Resampling.BILINEAR),
                    (column * side, header),
                )
            mass = float(self.latest_map.sum())
            maximum = self.config.vmax or float(self.latest_map.max())
            draw.text(
                (8, 284),
                f"{self.config.source} -> image | layer {self.layer} | camera {self.config.camera} | image mass {mass:.3f}",
                fill="#333333",
                font=font,
            )
            scale = "fixed" if self.config.vmax else "relative"
            draw.text(
                (8, 301),
                f"Input step {self.prediction_step}; queued-action age {self.step - self.prediction_step} | {scale} color range 0..{maximum:.4g}",
                fill="#555555",
                font=font,
            )
        return np.asarray(canvas)[None]
