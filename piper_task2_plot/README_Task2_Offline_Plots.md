# Task 2：teleoperation 与 action from model 离线对照图

## 文件与用途

`plot_task2_teleoperation_vs_model.py` 是完整的离线推理和绘图入口。
`run_task2_offline_plot.sh` 是单卡启动器；两个文件放在同一目录。
`test_task2_offline_plot.py` 是 CPU 测试。四个源码和说明文件位于仓库的 `piper_task2_plot/` 目录。

脚本只处理 Task 2。它需要已有 `piper-partial-ft` 仓库和训练环境，不是独立于 LeRobot 的通用模型推理库。
不会导入部署入口、Piper 驱动或 CAN 控制，不会连接机械臂或实时相机。

## 默认输入

代码仓库：

```text
/root/autodl-tmp/lerobot060-piper-partial-ft
```

模型会从下面的运行目录解析唯一的 `020000/pretrained_model`，不会悄悄选择其他步数：

```text
/root/autodl-tmp/piper-partial-ft-work/chkpt/2601-lerobot/piper/task2-fullhead-vlm2-resume10k-to20k/pi05-may-2-remove_the_cuboid_from_blue_plate-dual_prompt_only-seed0/checkpoints/020000/pretrained_model
```

数据：

```text
/root/autodl-fs/datasets/reality/2-remove_the_cuboid_from_blue_plate
```

Tokenizer：

```text
/root/autodl-fs/ckpt/google/paligemma-3b-pt-224
```

直接使用完整 checkpoint，不使用 bundles，也不需要重新 materialize 已存在的 020000 模型。

## 运行

在已成功训练该模型的 conda 环境中，从仓库根目录执行：

```bash
cd /root/autodl-tmp/lerobot060-piper-partial-ft

# 检查路径、配置、Prompt 权重、数据身份和验证集划分；不加载完整模型。
GPU_IDS=0 bash piper_task2_plot/run_task2_offline_plot.sh --preflight-only

# GPU 0 上做实际离线预测和绘图。
GPU_IDS=0 bash piper_task2_plot/run_task2_offline_plot.sh \
  --episodes val \
  --num-episodes 3 \
  --windows-per-episode 2

# CPU 测试（合成数据，不需要真实权重）。
python -m pytest piper_task2_plot/test_task2_offline_plot.py -q
```

启动器优先使用 `PIPER_REPO_ROOT`；未设置时，自动识别其相邻的 LeRobot 仓库。
独立解压到仓库外时，默认使用 `/root/autodl-tmp/lerobot060-piper-partial-ft`。
也可以进入 `piper_task2_plot/` 后运行 `bash run_task2_offline_plot.sh`。
下面的简短示例均假设当前位于 `piper_task2_plot/`。

未指定输出目录时，每次创建时间戳子目录，不覆盖已有结果：

```text
/root/autodl-tmp/piper-partial-ft-work/offline_plots/task2-fullhead-vlm2-20k/<时间戳>/
```

需要指定输出目录时，该目录必须尚不存在：

```bash
GPU_IDS=0 bash run_task2_offline_plot.sh \
  --output-dir /root/autodl-tmp/piper-partial-ft-work/offline_plots/task2-20k-check1
```

每次都用已经空闲的 GPU；不要与仍占用同一卡的训练任务同时运行。
脚本会强制 Hugging Face 离线模式，不会下载模型或 Tokenizer。
缺少绘图依赖时，在当前环境安装：

```bash
python -m pip install matplotlib pillow
```

不要为了本脚本另外替换已经可用的 torch、transformers 或 LeRobot 版本。

## 选择 episode

默认从训练时保留的验证 episodes 中按索引均匀选取最多 3 条，不根据预测效果筛选。
优先读取该 run 的 `dataset_split.json`。缺少它时，仅在 checkpoint 存在可用的
`train_config.json` 时按该仓库原有的每任务末尾留出规则重建；不会默认猜一个 10% 划分。

全部验证 episodes：

```bash
GPU_IDS=0 bash run_task2_offline_plot.sh --episodes val --num-episodes 0
```

指定已有数据集中的 episode（必须属于保存的任务划分）：

```bash
GPU_IDS=0 bash run_task2_offline_plot.sh --episodes 83,84 --num-episodes 0
```

`--num-episodes 0` 表示不对候选 episode 列表做数量限制。
指定列表可能包含训练样本；脚本会在 summary 中记录 val、train 或 mixed，不把它们冒充验证集。

## 输出内容

```text
<输出目录>/
  index.html                         浏览所有图片的静态页面
  run_manifest.json                  配置、来源、随机种子、完成状态
  summary.json                       汇总与逐 episode 指标；成功结束才写入
  metrics.csv                        h1/h8/h50 各维 MAE、RMSE、bias、hold MAE
  predictions.npz                    全部原始 action chunk 预测/标签/有效掩码
  episode_000083/                    实际编号依验证划分而定
    predictions.npz                  本条 episode 的完整动作 chunk
    episode_curves.csv               逐帧拼接曲线的原始数值
    episode/
      joint_1.png, joint_1.svg
      ...
      joint_6.png, joint_6.svg
      gripper.png, gripper.svg
    frame_000000/
      observations/                  当前观测对应的两路已录制 RGB 图像
      chunk50/                       当前观测预测的完整有效 50 步窗口，各维单独保存
      first8/                        同一窗口前 8 步放大图，各维单独保存
```

PNG 用于查看；SVG 用于后期排版。每个关节和夹爪各自成图。
图例只保留两个名称：`teleoperation`、`action from model`。
把整个输出文件夹传回电脑后，打开 `index.html` 可以连续查看所有曲线和记录图像。

成功结束会打印 `TASK2_PLOT_OK: .../index.html`。没有 summary.json 或
`run_manifest.json` 的 completed 为 false 时，表示运行尚未完整结束。

## 图的准确含义

### 完整 episode 曲线

在记录帧 0、8、16……取真实观测，每次预测未来 50 步，只保留当前 chunk 的前 8 步；
末尾只保留有效动作。这样拼接后，每个真实动作标签只对应一个预测值，没有跨 chunk 平均。

这是以示教观测为条件的离线预测，不是让模型自身动作改变后续状态的闭环轨迹。
`teleoperation` 是数据集 action 标签，不是由模型重建的轨迹，也不自动等于 follower 编码器读数。

### 50 步窗口和前 8 步放大图

每条 episode 均匀选取两个观测窗口（可改 --windows-per-episode）。优先使用完整 50 步窗口，
排除末尾 padding；没有完整窗口时仅使用有效长度。不会把 padding 画成静止的真实标签。

offset 0 对应当前帧 action 标签，保持该训练分支的原有对齐，不自行整体平移一帧。
数据按 meta/info.json 中的 fps 解释时间。关节原始数据是弧度，图与关节指标转换成度；
夹爪保留数据集单位，不擅自换算为米或毫米。不会平滑、裁剪或延迟校正曲线。

## 推理与指标

使用 checkpoint 保存的 preprocessor/postprocessor，保留训练后的 state/action 归一化统计。
仅覆盖本机 Tokenizer 路径和设备；不重新拟合统计，不使用 base 模型的处理器替换。
BF16 模型显式使用 BF16 autocast，避免默认 FP16；同时使用 model.eval() 和 inference_mode。
每个全局观测索引使用 seed+index 的固定噪声种子，便于跨 checkpoint 重复同一观测比较。

h8 汇总包括六关节 MAE、保持当前姿态基线、夹爪 MAE，以及逐关节误差。
J2/J3 还报告真实平均运动幅度、预测平均运动幅度、方向一致率和运动增量 MAE。
运动统计只包括相对当前姿态的真实动作变化超过 1 度的位置（可修改 --motion-threshold-deg）。
h50 对 anchor/offset 对统计，同一未来标签可能出现多次；h8 不重复统计帧。

本脚本默认使用完整的所选验证 episodes，与训练日志固定抽样的 action_val/h8 并不相同。
不能仅把这次汇总均值与训练日志 3.3137° 直接相减作为改善幅度。

## 验证范围

本次已完成 Python 语法检查、Bash 语法检查和 18 项 CPU 测试。
测试覆盖 checkpoint 路径解析、验证划分、时间对齐、padding、h8 拼接、误差计算、图例、
图片输出，以及使用模拟 LeRobot/模型接口的端到端文件输出与标签隔离检查。
测试使用合成数据；没有访问 AutoDL 上的实际 checkpoint、视频或 GPU，也没有实际真机测试。
仓库不提交合成测试曲线，避免与模型结果混淆。

## 实现依据

针对以下仓库版本的接口编写。本工具作为独立目录加入，不修改训练器、模型或机器人驱动：

```text
Repository: LianYi233/lerobot060
Branch: piper-partial-ft
Inspected commit: decc30d8cd9572aaebb9502e6ff458ceb5ae7157
Source interfaces: eval_piper_offline.py, piper_policy_utils.py,
                   src/lerobot/datasets/factory.py, src/lerobot/datasets/lerobot_dataset.py
PyTorch reference: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html
```
