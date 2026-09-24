# Piper 真机部署（PI05 双 prompt）

`piper` 分支从 `prompt-ablation` 的 `51a94e697ae4a3a59b6996c484c68a3137c6594a`
创建，保留完整的配置、模型、注意力、处理器和 CABO 实现。根目录
`deploy_piper_vlaa.py` 使用此版本 PI05 的 flow 推理，支持双 prompt 和 prompt 消融配置。

之前的 `lerobot.processor.core` 导入错误和 `num_vlm_prompt_tokens` 缺失说明部署机混用了
不同版本源码。入口现在优先加载旁边的 `src/`，并核对配置来源及必要字段。不要继续向旧版
`PI05Config` 手动补字段，也不要把旧部署目录的 `policies/` 或 `processor/` 复制进来。

## 1. 获取分支和原 Piper 驱动

首次部署可新建目录，保留原部署目录的修改和相机配置：

```bash
cd /home/drx/DJC
git clone --single-branch --branch piper \
  https://github.com/LianYi233/lerobot060.git lerobot-piper
cd lerobot-piper

# 只复制原部署机上已经使用的完整 Piper 驱动目录。
# 首次复制时目标应不存在；若已存在，先检查内容，不要覆盖自己的修改。
test ! -e src/lerobot/robots/piper && \
  cp -a /home/drx/DJC/lerobot-main/src/lerobot/robots/piper src/lerobot/robots/
```

仓库及此前上传的单个部署脚本均不包含完整 Piper 驱动，因而该目录不能从 GitHub 拉取。
它至少需要提供 `lerobot.robots.piper.PiperConfig` 和
`lerobot.robots.piper.piper.Piper`，以及原有的 SDK、本地依赖和相机配置。
部署沿用原脚本的驱动约定：`joint_1.pos` 至 `joint_6.pos` 为弧度，`gripper.pos` 为 0..1；
使用前确认这与训练数据和本地驱动一致。

新目录后续更新：

```bash
cd /home/drx/DJC/lerobot-piper
git pull --ff-only origin piper
```

如果已在此仓库的其他分支，先保存本地改动，再用 `git fetch origin`、
`git switch --track origin/piper` 切换；若本地 `piper` 已存在，用 `git switch piper`。
本地同名部署脚本可能阻止切换，不要用强制 checkout 覆盖它。

## 2. 环境检查

使用兼容本分支的 Python 环境。`pyproject.toml` 声明 Python >=3.12、
torch >=2.7 且 <2.12、transformers >=5.4 且 <5.6。优先复用训练环境中的依赖版本，
并保留原部署环境使用的 `piper_sdk`、`pyrealsense2` 及驱动其他依赖。

在已满足依赖的环境里直接运行即可；入口自动加载此 checkout 的源码：

```bash
python deploy_piper_vlaa.py --check_env
```

应打印此目录下 `configuration_pi05.py` 的路径、Python/torch/transformers 版本、
CUDA 是否可用、Piper 驱动文件是否存在，并以 `ENV_OK` 结束。
此检查会导入配置、模型和处理器，不加载权重、不导入或连接 Piper 驱动；
`ENV_OK` 仅代表 PI05 导入通过，驱动显示 `exists=False` 时仍需完成上一步复制。

若创建独立环境，需要联网安装本分支依赖时，可在激活的新环境中执行：

```bash
uv pip install --python "$(command -v python)" -e '.[pi,intelrealsense]'
```

此命令不安装自定义 Piper 驱动或其 SDK；它们应沿用原部署机验证过的版本。
没有 `uv` 时可用 `python -m pip install -e '.[pi,intelrealsense]'`。
不要在旧环境中盲目覆盖 torch/CUDA。离线部署应提前准备同版本依赖。

## 3. 准备 checkpoint 和数据元信息

复制完整 `checkpoints/006000/pretrained_model/`，至少包括：

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
处理器 JSON 的 state_file 引用的所有文件（含归一化统计量）
```

入口要求正式 `flow` 阶段的完整 PI05 权重，并核对已保存的 VLM/action prompt 权重形状。
它拒绝缺失 prompt 权重、旧 quantile/regression 模型及仅含 PEFT adapter 的目录。
保留 checkpoint 的 prompt 数量、输入输出特征和 chunk_size，不重新计算部署归一化统计。

`--dataset_info` 指向**该任务训练数据的 `meta/info.json` 副本**。
例如网球任务对应：

```text
May-pick-and-place/3-move_the_tennis_from_yellow_plate_to_blue_plate/meta/info.json
```

可将其复制为 `/home/drx/Downloads/wyn/tennis_info.json`。不需要复制 data/ 或 videos/。
若数据就在本机，也可用 `--dataset_root /路径/May-pick-and-place/3-move_the_tennis_from_yellow_plate_to_blue_plate`
替代 `--dataset_info`。

离线 tokenizer 必须已在本机缓存。如果 checkpoint 记录的是训练服务器上的绝对路径，
为下列命令补充 `--tokenizer_path /本机实际存在的/paligemma-tokenizer目录`。
`HF_HUB_OFFLINE=1` 不会补齐缺失的文件或修复源码版本。

## 4. 不连接硬件的完整推理

```bash
HF_HUB_OFFLINE=1 python deploy_piper_vlaa.py \
  --policy_path /home/drx/Downloads/wyn/piper-move-tennis-pi05-dualprompt-cabo2-fp32/checkpoints/006000/pretrained_model \
  --dataset_info /home/drx/Downloads/wyn/tennis_info.json \
  --task "move the tennis from yellow plate to blue plate" \
  --dry_run
```

`DRY_RUN_OK` 表示权重和处理器加载后完成了一次合成观测推理。此模式不导入 Piper 驱动、
不连接 CAN 或相机，也不代表真机任务成功。单独使用 `--dry_run` 时元信息可省略；
提供元信息能同时校验状态/动作名称和模型相机 key。

## 5. 真机执行

```bash
HF_HUB_OFFLINE=1 python deploy_piper_vlaa.py \
  --policy_path /home/drx/Downloads/wyn/piper-move-tennis-pi05-dualprompt-cabo2-fp32/checkpoints/006000/pretrained_model \
  --dataset_info /home/drx/Downloads/wyn/tennis_info.json \
  --task "move the tennis from yellow plate to blue plate" \
  --num_episodes 5 \
  --steps_per_episode 1000
```

连接动作沿用原脚本：启用机械臂和夹爪，再连接相机。每轮按 Enter 开始；轮次间自行重置
任务场景。Ctrl-C 退出后执行驱动断开流程。未实现自动回初始位或成功率判定。

状态与动作分量按数据元信息各自的 `names` 排序。默认执行频率为训练数据 `fps`，
默认每个 chunk 执行 checkpoint 的 `n_action_steps` 步；可用 `--steps_per_chunk 8`
缩短每次执行段，`--control_hz` 只改变执行频率。等待推理会额外增加 chunk 间的间隔，
脚本在这段时间持续发送当前位置。每次观测只预测一个完整动作段，使用训练时保存的
前后处理器恢复动作单位，随后应用原脚本关节限位和非有限值检查。默认关闭 compile。

相机名称默认去掉模型 key 的 `observation.images.` 前缀，需要改名时显式传入：

```bash
--camera_map '{"observation.images.wrist":"wrist_camera"}'
--camera_serials '{"wrist_camera":"相机的实际序列号"}'
```

示例只说明参数格式；请按 checkpoint 和驱动的实际相机填写所有视角。
图像默认认为驱动输出 RGB；原连接代码请求 RealSense BGR8，驱动的 `get_observation()`
可能已转换为 RGB。若驱动返回原始 BGR，传入 `--camera_color_order bgr`。
不要在未检查驱动输出前重复交换通道。

## 本地代码验证

无需 torch 或机器人 SDK 的回归测试（需要 numpy）：

```bash
uv run --no-project python tests/test_deploy_piper.py
```

覆盖源码加载优先级、旧配置拒绝、训练元信息顺序、相机通道、prompt 权重缺失、
归一化处理器文件、动作段执行、异常传播及连接失败清理。真机 SDK 和实际 checkpoint
需要在部署机另外验证。
