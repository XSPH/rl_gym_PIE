# 平地盲狗实验实施记录

日期：2026-10-08。仓库：`/home/asuka/Legged/parkour/rl_gym_PIE_native`。
从本地 `025907acd6a8ad1c17fc1aca116133fdbbbff563` 创建 `experiment/pie-blind-flat`，
将工作区已有的 PD 20/0.5 修改纳入该分支。本次只做本地开发及静态/CPU 检查。
远端为 `https://github.com/XSPH/rl_gym_PIE.git`。初版提交 `d7b2dd1` 后已推送并通过 SSH
同步至 `asuka@10.70.205.234:/home/asuka/rl_gym_PIE_native`，服务器 139 项 CPU 检查通过。
随后按用户要求跳过训练时的相机渲染，见 [渲染优化记录](pie_blind_flat_no_render.md)。

## 实验设置与实现

| 项目 | 当前设置 |
| --- | --- |
| 策略视觉 | `camera.input_mode='zero'`，深度 2×60×80 恒为全零 |
| 网络/监督 | 原 CNN、Transformer、GRU、全部预测头与辅助损失；235 维特权 Critic |
| 地形 | `['flat']`、`[1.0]`；10×20 网格，8×8 米地块，课程关闭，初始等级 0 |
| 名义 PD | 刚度 20、阻尼 0.5；原增益随机化仍启用 |
| 相机 | 位置 `[0.25,0,0.06]`、俯角 30°、FOV 87°；视觉随机化关闭 |
| 时间 | 50 Hz 控制、10 Hz 逻辑图像更新、1 帧固定延迟；真实渲染仅用于视觉模式或诊断 |
| 随机化 | 原摩擦、质量、COM、增益、电机、动作延迟、推扰、本体噪声 |
| 奖励/命令 | 与起点配置相同，保留奖励控制 dt 缩放与正奖励裁剪 |
| 训练 | 4096 环境、24 步、5 epochs、4 minibatches、15000 轮、每 500 轮保存 |
| 起点/日志 | `resume=False`，随机初始化第 0 轮；`logs/lite3_pie_blind_flat/` |

`legged_gym/envs/pie/lite3_config.py` 是有效配置的唯一来源。
`Lite3PIE` 初始化历史及延迟缓冲时填零，训练时直接将复用的全零图像写入原有缓冲，
不初始化 Warp 相机、不渲染或编码。局部重置只更新所选行。启用 `--show_depth` 时，
真实 `camera.depth` 供窗口读取，策略输入仍为零。
观测返回、帧池、视觉特征复用、PPO 和网络结构沿用原实现。

相机位置/俯角/FOV 随机化使用新增的独立开关 `domain_rand.randomize_camera`。
原 `randomize_pie` 继续控制 COM、增益、电机及动作延迟。
`rsl_rl/utils/pie_config.py` 统一校验 `depth` / `zero` 模式，非法值明确报错。

version 4 继续保存完整环境、网络、训练配置以及 Adam、LR、计数与 RNG。
视觉模式随 `environment_cfg.camera.input_mode` 保存；加载前检查当前模式与保存模式一致。
旧版 v4 缺少新增字段时解释为 `depth`，不能作为本实验默认配置的续训起点。
play 从保存配置恢复视觉模式，按原有回放流程关闭随机化及课程。

`--show_depth` 显示真实采集画面（米）和两帧实际策略输入。策略输入不换算成米，
归一化图像显示独立的 `[-0.5,0.5]` 色标及单位，盲狗模式标注 `zero mode`。
显示器只读已有缓冲，不推进传感器。

## 本地验证

初版 **139 项 CPU 检查通过，32 条警告，25.61 秒**；警告来自已有 PyTorch
`torch.cpu.amp.autocast` 弃用提示。实现后原有 103 项检查先通过，再新增 36 项覆盖。
静态结果和完整有效配置记录在 [CPU 验证数据](validation/pie_blind_flat_cpu_2026-10-08.json)。
渲染优化后本地 **147 项通过，32 条警告，29.84 秒**，详见渲染优化记录。

新增检查包含初始化、捕获周期、固定延迟、局部/全体重置、观测返回、
相机随机化与非视觉随机化开关独立性、平地网格及起始等级。
固定本体历史与 GRU 状态，改变原始相机画面后动作和下一步 GRU 状态逐值一致。

帧池、缓存前向/梯度与多 epoch PPO 等价检查同时覆盖明确的 `depth` 与 `zero` 配置。
零输入首层卷积权重梯度为零；首层 bias 和 GRU 仍有梯度，所有辅助损失保持启用。
旧视觉地形哈希、默认网络参数及前向输出的原有回归检查继续覆盖。

checkpoint 检查覆盖同模式保存/加载、Adam/LR/计数恢复、盲狗 CPU 更新、play 模式恢复、
旧版 v4 缺字段恢复为 `depth`，以及不同模式在权重/优化器加载之前报错。
深度窗口使用真实 Matplotlib 图像对象和内存窗口替身检查图像值及单位；没有打开 GUI。

CPU 检查没有启动 Isaac Gym 仿真，也不能证明 GPU 行为或基础运动已经学会。

## 训练、续训与 play

本机专用环境，无须重装依赖：

```bash
source /home/asuka/miniconda3/etc/profile.d/conda.sh
conda activate /home/asuka/Legged/parkour/.conda-envs/pie-isaacgym-native
cd /home/asuka/Legged/parkour/rl_gym_PIE_native

# 从头训练，默认设置即为本实验。
python -s legged_gym/scripts/train.py --task=lite3_pie --headless

# 续训本实验的 zero checkpoint；总目标仍为 15000 轮。
python -s legged_gym/scripts/train.py --task=lite3_pie --headless \
  --resume --checkpoint_file /absolute/path/model_500.pt

# 第三人称窗口和深度窗口；自动从模型恢复 zero 模式。
python -s legged_gym/scripts/play.py --task=lite3_pie \
  --checkpoint_file /absolute/path/model_500.pt --num_envs 1 --steps 2000 \
  --show_depth --depth_env 0

# 仅 CPU 检查，不启动仿真。
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  MPLCONFIGDIR=/tmp/pie-blind-flat-mpl \
  python -s -m pytest -q tests --basetemp=/tmp/pie-blind-flat-tests
```

后续正式训练与回放应观察速度跟踪、姿态/碰撞奖励、回合长度、终止原因，
以及站稳、直行、转向表现；这些判断不在本次 CPU 验证范围内。
