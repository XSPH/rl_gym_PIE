# 视觉特征复用与图像帧索引存储

> 历史记录：下面描述的是重构前 `unitree_rl_gym` 的修改与 53 项检查。
> 本分支已将相同视觉优化适配原版训练接口，取消旧模型兼容；当前实现和验证结果
> 以 [原版训练流程重构记录](native_training_refactor.md) 和 [模块收拢记录](pie_module_cleanup.md) 为准。

2026-10-03：仅修改本地 `unitree_rl_gym`。未修改 mjlab、历史回放副本、
GitHub 发布副本或远程服务器；未启动仿真或 GPU 训练。

## 论文依据与工程边界

[PIE 原文](https://arxiv.org/html/2408.13740v3) 第 III-B 节规定：两帧深度沿通道
维堆叠进入 CNN，本体历史进入 MLP，再通过 Transformer 和 GRU 融合。
第 IV-A 节报告真机深度输入 10 Hz、动作输出 50 Hz。本项目已有相同频率设置。
本次保留整组两帧 CNN 输入以及每个控制步的本体编码、Transformer、GRU 和输出。

论文没有披露帧池、缓存或激活重算实现，因此这三者属于工程实现选择，不能称作
作者原版代码。相同输入在同一组权重下经过当前无随机性、无运行统计的 CNN，
输出相同；复用此结果可以减少计算。作为工程参考，
[Extreme Parkour 的视觉训练循环](https://github.com/chengxuxin/extreme-parkour/blob/main/rsl_rl/rsl_rl/runners/on_policy_runner.py#L207)
也在收到新深度时更新视觉特征，并在其他控制步使用已有特征。这里只借鉴其频率
处理，没有引入其教师蒸馏流程。

## 修改内容

| 文件 | 修改 |
| --- | --- |
| `legged_gym/envs/pie/lite3.py`（当前路径） | 为真实捕获的单张图像分配全局唯一 int64 ID，随延迟队列和两帧历史同步移动；局部 reset 仅替换对应行 |
| `rsl_rl/rsl_rl/storage/rollout_storage_pie.py` | 添加 `DepthFramePool`，每个 rollout 只保存唯一单帧图像；逐控制步保存环境×历史的索引；支持按原顺序无损恢复两帧输入 |
| `rsl_rl/rsl_rl/modules/actor_critic_pie.py` | 分离纯深度编码接口，添加按完整两帧 ID 判断的 CNN 特征缓存，通过非原地 `index_copy` 保留计算图 |
| `rsl_rl/rsl_rl/algorithms/ppo_pie.py` | collect、每个逻辑 PPO minibatch、更新后的 hidden 重放分别创建独立缓存；新增实际编码数量、复用比例和帧池大小指标 |
| `rsl_rl/rsl_rl/runners/on_policy_runner_pie.py` | 独立验证与 inference policy 接入缓存；保留原版日志及奖励项，追加视觉统计到终端、TensorBoard 和 JSON |
| `tests/test_pie_visual_reuse.py` | CPU 无损存储、动作／概率／hidden／梯度、多个 PPO 更新、更新权重后的重放和保存张量对照 |
| `tests/test_pie_observation_and_push.py` | 增加真实相机队列方法的 CPU 替身检查，覆盖延迟、局部重置和存储独立性 |

图像保持原浮点精度；正常训练仍是 FP32。没有 uint8 量化、额外噪声或图像预处理。
帧池使用设备端查找，不把逐环境图像或 ID 搬到 CPU 字典。
环境返回的临时深度快照仍存在；消除的是 rollout 中逐步保留的重复图像。
帧池最终合并时可能短暂同时持有分块与连续池，不能把池大小等同于程序峰值。

训练缓存只覆盖一次逻辑 minibatch。在其完整 24 步中，所有使用位置的梯度汇总
到同一 CNN 计算图；没有 detach 视觉特征或提前截断 GRU。每次 optimizer 更新后
丢弃缓存，下一批次按新权重计算。采样、hidden 重放和回放使用各自无梯度缓存。
无帧 ID 的旧调用者保留 raw-depth 存储和直接编码路径。

`initial_state()` 显式沿用模型 std 的 dtype；正式 FP32 行为不变，也使 CPU
float64 对照正确创建 hidden。没有修改网络参数名、参数尺寸、`ModelConfig`
字段或 Adam 参数顺序；帧池和特征缓存不进入模型 checkpoint。

## 保留的配置

4096 环境、24 步 rollout、5 epochs、4 个逻辑 minibatches、每轮 20 次 optimizer
更新、15000 轮、每 500 轮保存，以及学习率／KL 调度、奖励、随机化、相机频率
和延迟、物理配置、网络宽度、损失及原有 GRU 时序边界均保持原值。
没有混合精度；没有启用梯度累积或增加逻辑 minibatch 数。

## CPU 验证

专用环境：Python 3.8.10、PyTorch 2.4.1。显式隐藏 CUDA，未创建仿真器：

```bash
cd /home/asuka/Legged/parkour/unitree_rl_gym
PYTHONPATH="$PWD/rsl_rl:$PWD" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  ../.conda-envs/pie-isaacgym/bin/python -s -m pytest -q tests \
  --basetemp=/tmp/pie-visual-reuse-all
```

结果：53 项通过。包含旧 checkpoint／optimizer 续训、PPO 调度、奖励日志及
原有标签／终止路径回归。新对照覆盖完整 24 步和多环境异步相机变化，CNN、GRU
梯度均非零；多个 epoch 的更新及更新权重后 hidden 重放与旧路径在指定容差内一致。
另外与本地发布基线 `7911fff` 的模型源码做 CPU 对照：`ModelConfig`、全部
state dict 键、相同种子下的初始权重、参数顺序均一致，旧基线权重严格加载通过。
这不是实际加载并回放 `model_14000.pt` 的仿真验证。

FP32 多次 Adam 更新的注意力 key-bias 切片出现约 `5.03e-6` 参数差异，其原始
梯度幅值约 `2.92e-11`；浮点累加残差相对 Adam epsilon 被放大。仅此切片使用
独立容差，其他参数保留原检查。统一 float64 的 CPU 对照以 `rtol=1e-9`、
`atol=1e-11` 检查全部参数和梯度通过；正式训练精度未改变。

## 梯度累积能否替代重算

使用完整 `ModelConfig()`，4 个环境、24 步、两帧 60×80 图像，含异步 reset。
通过 `saved_tensors_hooks` 按底层存储去重，排除参数存储，测量 encoder／
Transformer／GRU 为反向传播保存的张量；同时实际反向传播确认计算图有效。

| 方案 | 保存张量约 MiB | 前向 CNN 输入栈数 |
| --- | ---: | ---: |
| 原逐步 CNN＋激活重算 | 15.83 | 96 |
| 索引＋复用＋激活重算（当前默认） | 13.26 | 22 |
| 索引＋复用＋直接 CNN | 24.65 | 22 |
| 只取 1/4 环境：索引＋复用＋激活重算 | 3.14 | 6 |
| 只取 1/4 环境：索引＋复用＋直接 CNN | 6.24 | 6 |

这个样例的 CNN 前向输入栈减少约 77%，保存张量减少约 16.2%。**不是整轮耗时
减少 77%，也不是 4090 实测显存减少 16.2%。** 测量不含 actor／critic／decoder
完整联合损失、全部 rollout 存储、CUDA 临时工作空间、PhysX 和 Warp 内存。
局部 reset 的频率也会影响真实复用比例。

结论：当前保留激活重算。直接取消重算仍增加保存张量；按环境拆微批有进一步
降低激活存储的空间，但该行仅测小批图，并没有实现或证明梯度累积更新等价。
若后续实现累积，需保持每个微批完整 24 步、按照整个逻辑批的有效样本数归一化
各个 masked auxiliary loss、控制 VAE 随机样本，并且只在逻辑批结束时计算整体
KL／调整学习率／裁剪梯度／执行一次 Adam。直接平均微批损失或增加 minibatches
不能保证等价。是否值得替代当前重算，需要真实 GPU 峰值和更新时间证据。

新增日志：`Depth frame pool (MiB)`、`Equivalent dense depth (MiB)`、
`CNN feature reuse fraction`、`Unique depth frames`、`CNN encoded stacks` 和
`Equivalent dense CNN stacks`。CNN 数量统计前向图像栈，不计 backward 的重算调用。
