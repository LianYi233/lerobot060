"""Read-only Piper policy checks shared by deployment and offline evaluation.

This module never imports deployment entry points, robot guards, camera helpers,
CAN, Piper SDK or RealSense. ML/checkpoint dependencies are imported on demand.
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np

JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]


def activate_checkout_source():
    """Use the adjacent checkout even if another LeRobot is installed editable."""
    source = Path(__file__).resolve().parent / "src"
    package = source / "lerobot"
    if not (package / "__init__.py").is_file():
        raise RuntimeError(
            "请在完整 piper 分支中运行；piper_policy_utils.py 旁边必须有 src/lerobot/。"
            "不要把入口单独复制到旧版 LeRobot 中。"
        )
    # A caller may already have imported the other editable checkout. Replacing
    # sys.path cannot replace cached modules; fail instead of mixing versions.
    for name, module in tuple(sys.modules.items()):
        if name == "lerobot" or name.startswith("lerobot."):
            origin = getattr(module, "__file__", None)
            if origin is not None and not Path(origin).resolve().is_relative_to(package):
                raise RuntimeError(f"已加载其他目录的 {name}: {origin}；请用新 Python 进程运行此入口")
    sys.path.insert(0, str(source))
    importlib.invalidate_caches()
    return source


def validate_pi05_source(config_class, source):
    """Check source identity and prompt support before reading checkpoint config."""
    module = sys.modules[config_class.__module__]
    actual = Path(module.__file__).resolve()
    expected = source / "lerobot/policies/pi05/configuration_pi05.py"
    if actual != expected.resolve():
        raise RuntimeError(f"PI05 配置来自其他源码: {actual}；应为 {expected}")
    required = {"num_vlm_prompt_tokens", "num_prompt_tokens", "training_stage"}
    missing = required - set(getattr(config_class, "__dataclass_fields__", {}))
    if missing:
        raise RuntimeError(
            f"PI05 源码版本不匹配，缺少 {sorted(missing)}：{actual}。"
            "请拉取完整 piper 分支，不要手动补配置字段。"
        )
    print(f"PI05 config source: {actual}")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def check_checkpoint_files(policy_path):
    """Require all processor state files, including the training normalization stats."""
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (policy_path / name).is_file():
            raise ValueError(f"Checkpoint 文件缺失: {policy_path / name}；请复制完整 pretrained_model 目录")
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        for step in read_json(policy_path / filename)["steps"]:
            if step.get("state_file") and not (policy_path / step["state_file"]).is_file():
                raise ValueError(f"Checkpoint 处理器状态缺失: {step['state_file']}；不能省略归一化统计文件")
    raw_config = read_json(policy_path / "config.json")
    if "action_generation_mode" in raw_config or any("quantile" in key for key in raw_config):
        raise ValueError("这是旧 quantile/regression 配置；请改用 prompt-ablation 训练得到的 flow checkpoint")


def check_prompt_weights(policy_path, policy_cfg):
    # The training loader initializes absent prompts for base models. Inference
    # must reject that case before loading, so a missing learned prompt cannot be random.
    from safetensors import safe_open

    expected = {
        "model.vlm_prompt_tokens.weight": policy_cfg.num_vlm_prompt_tokens,
        "model.prompt_tokens.weight": policy_cfg.num_prompt_tokens,
    }
    with safe_open(str(policy_path / "model.safetensors"), framework="pt", device="cpu") as state:
        saved_keys = set(state.keys())
        for key, count in expected.items():
            if key not in saved_keys:
                raise ValueError(f"Checkpoint 缺少 {key}；需要本分支保存的完整模型，不能使用原始 base 权重")
            shape = state.get_slice(key).get_shape()
            if len(shape) != 2 or shape[0] != count:
                raise ValueError(f"Prompt 形状与 config 不符: {key}={shape}, tokens={count}")
            if not np.isfinite(state.get_tensor(key).float().numpy()).all():
                raise ValueError(f"Prompt 权重包含非有限值: {key}")
            print(f"Loaded prompt check: {key}, shape={shape}")


def deployment_features(policy_cfg, dataset_info=None):
    """Keep saved feature names and the dataset's component order, never overwrite config."""
    if not policy_cfg.input_features or not policy_cfg.output_features or not policy_cfg.image_features:
        raise ValueError("Checkpoint 缺少训练输入/输出特征；请选择真机训练得到的完整 checkpoint")
    expected_names = [f"{name}.pos" for name in JOINT_NAMES] + ["gripper.pos"]
    result = {}
    for key, config_features in (
        ("observation.state", policy_cfg.input_features),
        ("action", policy_cfg.output_features),
    ):
        feature = config_features.get(key)
        if feature is None or tuple(feature.shape) != (7,):
            raise ValueError(f"此 Piper 入口要求 {key} 为 7 维（6 关节+夹爪），checkpoint={feature}")
        if dataset_info is None:
            # Synthetic dry-run only: all zeros, so component order has no hardware effect.
            names = expected_names
        else:
            recorded = dataset_info["features"].get(key, {})
            names = recorded.get("names")
            if tuple(recorded.get("shape", ())) != tuple(feature.shape):
                raise ValueError(f"训练数据与 checkpoint 的 {key} 维度不符")
            if not isinstance(names, list) or len(names) != 7 or set(names) != set(expected_names):
                raise ValueError(
                    f"{key}.names 必须对应原 Piper 驱动的关节与夹爪名称，实际为 {names}；不能猜测顺序"
                )
        result[key] = {"dtype": "float32", "shape": (7,), "names": list(names)}
    for key, feature in policy_cfg.image_features.items():
        if not key.startswith("observation.images.") or len(feature.shape) != 3 or feature.shape[0] != 3:
            raise ValueError(f"不支持的相机特征: {key}={feature.shape}")
        if dataset_info is not None and key not in dataset_info["features"]:
            raise ValueError(f"Checkpoint 相机 {key} 不在指定训练数据中，请检查数据/模型是否属于同一任务")
        _, height, width = feature.shape
        result[key] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return result
