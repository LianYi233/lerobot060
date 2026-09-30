# VLM prompt 长度消融：1 / 2 / 8 / 16

在 `prompt-learning` 分支上，从相同基础模型分别训练四个 `vlm_only` 模型。
只改变 VLM prompt 的 token 数，action prompt 为 0，CABO 关闭，无 priming/bridge，
默认均训练 3000 个 flow updates。四组依次执行，适用于当前单卡 A100 环境。

## 一次运行四组

在仓库根目录、已激活的 lerobot 环境中运行：

```bash
GPU_IDS=0 BATCH_SIZE=32 \
  bash examples/training/train_pi05_vlm_token_sweep.sh 0
```

最后的 `0` 是训练 seed。数据集、基础模型与 tokenizer 默认沿用原脚本：

```text
DATASET_ROOT=/root/autodl-tmp/datasets/libero
PRETRAINED_PATH=/root/autodl-tmp/models/pi05_libero_base
TOKENIZER_PATH=/root/autodl-tmp/models/google/paligemma-3b-pt-224
```

可先加 `DRY_RUN=true` 检查四条完整命令；它仍会检查输入目录和输出冲突，但不会训练或创建日志。
`GPU_IDS`（复数）控制训练 GPU。`BATCH_SIZE`、`FLOW_STEPS`、精度、学习率等设置应在四组间保持一致。
默认 FP32、batch size 32 与原始消融脚本一致；不会因 token 数改变而自动增加 batch size。
需要多卡时继续设置 `GPU_IDS=0,1 NUM_PROCESSES=2`，其中 batch size 是每卡的数值。

## 输出与日志

默认 checkpoint 根目录：

```text
/root/autodl-tmp/chkpt/2601-lerobot/prompt-learning/vlm-token-sweep/
```

| Token 数 | 运行目录（seed 0） |
| --- | --- |
| 1 | `pi05-vlm_only-vlm1-seed0` |
| 2 | `pi05-vlm_only-vlm2-seed0` |
| 8 | `pi05-vlm_only-vlm8-seed0` |
| 16 | `pi05-vlm_only-vlm16-seed0` |

各组最终模型位于 `<运行目录>/checkpoints/003000/pretrained_model/`。
日志在 `/root/autodl-tmp/logs/prompt-learning/vlm-token-sweep/<运行目录名>.log`。
`OUTPUT_ROOT`、`LOG_ROOT` 可覆盖根目录，`RUN_PREFIX` 可覆盖默认 `pi05-vlm_only` 前缀。
运行名由 sweep 生成，不应额外设置 `RUN_NAME` 或 `VLM_PROMPT_TOKENS`。

脚本会在第一组训练前检查所有输出目录。已有输出时停止，不覆盖、不自动判断是否训练完成。
任一训练失败会停止后续组；确认前面组的结果后可明确选择剩余组：

```bash
VLM_PROMPT_TOKEN_COUNTS="8 16" \
  bash examples/training/train_pi05_vlm_token_sweep.sh 0
```

若失败组已有输出，请用新的输出根目录或 RUN_PREFIX 重跑该组，或者使用原生训练恢复流程。

## 单独训练指定长度

原入口新增环境变量 `VLM_PROMPT_TOKENS`：

```bash
VLM_PROMPT_TOKENS=8 \
  bash examples/training/train_pi05_prompt_ablation.sh vlm_only 0
```

显式设置长度时，默认运行名带 `-vlm8`，16 也会带 `-vlm16`，避免与原实验混淆。
单次入口仍使用原来的 `prompt-ablation` checkpoint/log 根目录；如需与 sweep 放在一起，
显式指定相同 `OUTPUT_ROOT` 和 `LOG_ROOT`。
不设置此变量时，原命令仍是 16 tokens，默认运行名仍为 `pi05-vlm_only-seed0`。
`action_only` 仍固定为 0 个 VLM prompt；其余含 VLM prompt 的 variant 也可使用这个变量。

四组都应从同一基础 checkpoint 开始，不把之前训练的 16-token 模型作为 1/2/8-token 的预训练路径。
如果 checkpoint 已含 learned prompt tensor，改变长度会产生 shape mismatch，加载器会报错；
本次修改不会裁剪、复制或重置已保存的 prompt 权重。

## 后续评估

训练完成后，一条命令依次评测 1、2、8、16 tokens：

```bash
GPU_ID=0 bash run_eval_libero_vlm_token_sweep.sh all 0
```

默认读取上述四组的 `003000/pretrained_model`，四套 LIBERO suite 各任务 10 个 episode，
即每组 400 个 episode，四组共 1600 个。最后的 `0` 选择训练 seed；评估环境/全局 seed
仍沿用评估器默认 1000。训练使用 `GPU_IDS`，评估使用 `GPU_ID`。

先只检查四组 checkpoint 路径、保存的 prompt 数量和将执行的命令：

```bash
DRY_RUN=true bash run_eval_libero_vlm_token_sweep.sh all 0
```

评估器从 checkpoint config 加载实际 token 数，不在推理时改写 prompt 大小。
所有组开始前会校验 `num_vlm_prompt_tokens` 与组名匹配、`num_prompt_tokens=0`。
路径缺失或数量不符时整批停止；评测中任一组失败也会停止后续组。
重新执行相同命令时沿用各组独立的任务级断点记录，已经保存的完整任务会跳过。
若更换模型、episode 数或评测配置，请使用新的 `BASE_OUTPUT`，避免与旧断点混用。

默认结果根目录：

```text
/root/autodl-tmp/eval/prompt-learning/vlm-token-sweep/
```

结果目录、日志和 summary 文件名都含 `pi05-vlm_only-vlmN-seed0`，不会混用四组结果。
例如 8 tokens：

```text
libero060-all-pi05-vlm_only-vlm8-seed0-003000-libero-all4-resume/eval_info.json
libero060-all-pi05-vlm_only-vlm8-seed0-003000-libero-all4-resume/resume_summary.json
eval_pi05-vlm_only-vlm8-seed0-libero-all4-summary.log
```

`run_eval_libero-full.sh` 也支持单独选择一组：

```bash
VLM_PROMPT_TOKENS=8 GPU_ID=0 bash run_eval_libero-full.sh vlm_only all 0
```

设置 `VLM_PROMPT_TOKENS` 后自动使用 token-sweep 的 checkpoint 根目录和带 token 数的运行名。
不设置时仍查找原 `prompt-ablation/pi05-vlm_only-seed0`，保持旧实验入口兼容。
若使用单次训练入口的旧根目录，可设置 `CKPT_ROOT`，或用 `BASE_CKPT` 指定单组的 checkpoints 目录。
其他含 VLM prompt 的 variant 仍可通过 `BASE_CKPT` 评估，长度选择快捷方式仅用于 `vlm_only`。

只测部分组/一个 suite，或选择其他训练 checkpoint：

```bash
VLM_PROMPT_TOKEN_COUNTS="8 16" GPU_ID=0 \
  bash run_eval_libero_vlm_token_sweep.sh libero_10 0

CKPT_ROOT=/actual/training/output/root \
BASE_OUTPUT=/actual/evaluation/output/root \
CHECKPOINT_STEP=3000 EPISODES_PER_TASK=10 GPU_ID=0 \
  bash run_eval_libero_vlm_token_sweep.sh all 0
```

自定义训练前缀时同步设置 `RUN_PREFIX`。批量入口不要设置仅指向一组的 `BASE_CKPT`、
`RUN_NAME` 或 `VLM_PROMPT_TOKENS`；分别使用 `CKPT_ROOT`、`RUN_PREFIX` 和 `VLM_PROMPT_TOKEN_COUNTS`。

记录各 suite 成功率及 token 数；attention 对比可使用已有 routing 入口，并控制 checkpoint 步数、
环境 seed、输入和去噪设置。单一训练 seed 的差异先作为探索结果，再对有意义的差异补多 seed 验证。
