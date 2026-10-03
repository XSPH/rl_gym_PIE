# PIE 原版结构分支：4090 短验证

日期：2026-10-04（北京时间）。分支：`refactor/pie-native-training`。
实际 GPU 检查源码：`ff0e38eed7cfd65a0b50f36fa8f865ce0739e002`。
机器可读结果见 [验证数据](validation/pie_native_4090_2026-10-04.json)。

## 部署

- GitHub：`XSPH/rl_gym_PIE` 的 `refactor/pie-native-training` 分支。
- 服务器工作树：`/home/asuka/rl_gym_PIE_native`。
- 专用环境：`/home/asuka/miniconda3/envs/pie-isaacgym-native`，从既有 PIE 环境离线克隆，
  Gym 和 RSL editable 安装均指向新工作树。
- 原 `/home/asuka/rl_gym_PIE` 的 main 工作树及原环境未修改。
- GPU：RTX 4090，24564 MiB，驱动 570.133.07；Python 3.8.20、PyTorch 2.4.1 / CUDA 12.4、
  Warp 1.6.2、既有 Isaac Gym Preview 4 SDK。
- 原始日志及短测试模型保存在服务器 `logs/validation_2026-10-04/`。
  本地日志副本位于 `logs/validation_4090_2026-10-04/`；日志和模型未提交到 Git。

## 检查结果

| 检查 | 结果 |
| --- | --- |
| 本地 CPU 回归 | 103 passed，27 条已知 PyTorch autocast 弃用提示 |
| 服务器 CPU 回归 | 103 passed，27 条相同提示，8.38 秒 |
| Warp 平面深度 | 2.0 米，与预期一致；两相机输出一致且有限 |
| 两环境 GPU 生命周期 | 超时、终止前动作标签、重置后动作及历史缓冲均通过 |
| 正式规模短训练 | 4096 环境 × 24 步、5 epochs、4 minibatches，两轮通过 |
| version 4 续训 | 从第 2 轮继续到第 3 轮，累计 294912 次环境转换 |
| version 4 单环境回放 | 200 控制步通过，奖励有限；此短回放未发生回合重置 |
| 旧格式回放拒绝 | 携带 version 3 标记的检查文件在创建环境前被明确拒绝 |

仅短测试的总轮数与保存间隔通过命令行覆盖为 `2 → 3` 和 `1`。
训练仍使用正式的出生高度 0.3 米、全部地形、噪声、推扰和随机化。
源码默认仍为 **4096/24/5/4/15000/500**。

| 累计轮数 | 采样 / 更新 / 合计秒 | 总损失 | Adam 累计更新 | CNN 复用率 |
| ---: | --- | ---: | ---: | ---: |
| 1 | 1.032 / 2.797 / 3.829 | 0.820331 | 20 | 83.31% |
| 2 | 0.870 / 2.851 / 3.721 | 0.214799 | 40 | 79.17% |
| 3（续训） | 0.989 / 2.951 / 3.940 | 0.229734 | 60 | 79.18% |

- 三轮的模型参数、损失、策略 KL、梯度范数及辅助指标全部有限，非有限状态重置计数均为 0。
  短训练有正常的倾斜/接触失败及超时重置。
- 第 2、3 轮中，CNN 的 6 个参数张量、GRU 的 4 个参数张量、Actor/Critic 各 8 个参数张量
  都发生更新；version 4 保存了模型、优化器、学习率、配置、累计计数和 Torch/CUDA RNG。
- 三个模型的学习率分别为 `0.0004444444 / 0.0006666667 / 0.0015`，与各自 Adam 参数组一致。
  Adam 计数 `20 → 40 → 60` 确认续训延续优化器状态。
- 图像帧池为 300.48 / 449.38 / 374.84 MiB；每轮等价稠密存储为 3600 MiB。

## 时间和显存

`nvidia-smi` 每 200 ms 采样整卡显存，Torch 记录进程内存分配峰值：

| 阶段 | 整卡采样峰值 | Torch allocated 峰值 | Torch reserved 峰值 | 入口总耗时 |
| --- | ---: | ---: | ---: | ---: |
| 两轮训练 | 10941 MiB（10.68 GiB） | 5.34 GiB | 6.77 GiB | 10.37 秒 |
| 续训一轮 | 11035 MiB（10.78 GiB） | 5.73 GiB | 6.98 GiB | 6.84 秒 |

单环境 200 步回放入口耗时 3.81 秒。入口耗时包含环境/模型初始化、保存和关闭；
轮次耗时来自原版 Runner 日志。整卡采样可能漏过短于 200 ms 的尖峰。
测试无 OOM，结束时 GPU 为 95 MiB、0% 利用率，无残留训练/回放进程。

## 检查修正

本次仅修改检查和文档，未改变训练实现：

1. `b21dbcb`：服务器 CPU 网络前向结果与本地基线最大绝对误差为 `4.47e-8`；
   输出比较改用 `atol=1e-7, rtol=1e-6`。网络参数及地形哈希继续严格一致。
2. `f8e4421`：GPU 超时检查按实际 `max_episode_length` 和回合计数执行。
   Gym 将时间步存为 float32，控制周期约 `0.01999999955` 秒；0.04 秒回合的上限为 3，
   原版 `>` 超时条件要求计数达到 4，且 `reset()` 已执行一次零动作步。
3. `ff0e38e`：两环境超时检查固定 Torch/NumPy 种子，出生高度仅在此检查中设为 0.6 米，
   避免随机关节姿态提前碰地或倾斜，干扰超时路径检查。

## 服务器运行指令

```bash
source /home/asuka/miniconda3/etc/profile.d/conda.sh
conda activate pie-isaacgym-native
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd /home/asuka/rl_gym_PIE_native
```

正式训练（本次未启动长训练）：

```bash
python -s legged_gym/scripts/train.py --task=lite3_pie --headless
```

复现短检查，选择新的输出目录；分析包装脚本只记录显存，实际执行以下原版入口：

```bash
python -s legged_gym/scripts/check_pie_backend.py
python -s legged_gym/scripts/train.py --task=lite3_pie --headless \
  --max_iterations 2 --save_interval 1 --output_dir /tmp/pie-native-smoke/train
python -s legged_gym/scripts/train.py --task=lite3_pie --headless \
  --resume --checkpoint_file /tmp/pie-native-smoke/train/model_2.pt \
  --max_iterations 3 --save_interval 1 --output_dir /tmp/pie-native-smoke/resume
python -s legged_gym/scripts/play.py --task=lite3_pie --headless \
  --num_envs 1 --steps 200 --checkpoint_file /tmp/pie-native-smoke/resume/model_3.pt
```

回放本次保存的短测试模型：

```bash
python -s legged_gym/scripts/play.py --task=lite3_pie --headless --num_envs 1 \
  --steps 200 --checkpoint_file logs/validation_2026-10-04/resume/model_3.pt
```

## 验证范围

已验证真实 GPU 后端、完整规模 PPO 更新、保存/续训及有限步回放。
这三轮用于检查流程与资源占用，未验证长期稳定性、训练收敛或跑酷效果；
上述时间和显存是这次短测试的数据。
