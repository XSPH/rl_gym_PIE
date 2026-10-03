# 原版训练流程重构记录

日期：2026-10-03。分支：`refactor/pie-native-training`。
开发 worktree：`/home/asuka/Legged/parkour/rl_gym_PIE_native`。

## 授权与基线

用户要求在新分支内尽可能复用原版训练流程；不兼容旧模型；奖励计算和注册完整沿用原版。
命令选择原版角速度范围采样，地形选择原版网格/中心出生点和距离课程。
Gym 单独开发，不修改 mjlab、旧 worktree 或远程正在运行的训练。

- 发布基线：`7911fff5479aeb295f2a2349c6ec9f9a0cd17ac1`。
- Unitree 原始基线：`276801e46c5d433564f24658bac64f254b7d2d4b`。
- RSL-RL 基线：v1.0.2，`2ad79cf0caa85b91721abfe358105f869a784121`。
- 导入此前只在本地实现的帧池/视觉复用，随后适配原版接口；未导入旧模型加载转换。

## 环境与接口

`Lite3PIE` 使用原版 `LeggedRobot.step`、`post_physics_step`、奖励函数注册/累计和
`BaseTask.reset`。任务提供观测、传感器、随机化、PD 延迟和课程扩展。
删除 PIE 内另一套 simulate/reset/reward 主循环与重复核心状态。

原版环境五元组保持：`obs45, critic235, rewards, dones, extras`。
`get_pie_observations()` 增加历史/深度/帧ID/当前辅助标签；
`extras['pie']` 记录重置前本体观测与 critic、终止原因。
重置前回调在计算奖励之后、改写机器人状态之前保存监督。
超时使用真实终止状态价值 bootstrap 一次；真实失败停止 bootstrap，GAE 不跨回合。

奖励权重移至原版 `cfg.rewards.scales`，保留原版 dt 和非负总奖励规则；
只增加功率和二阶动作变化函数。全局视觉附件设置恢复原版，Lite3 单独设置。

## 地形与配置来源

地形采用 Terrain 原版网格/原点组织；PIE 地形生成器提供五类障碍。
最终地形网格同时供 PhysX 与 Warp 使用，高度监督对应同一坐标和地表。
中心出生平台覆盖原版 ±1 米出生位置随机化及机器人足部。
障碍采用围绕中心平台的方形环带，避免直线赛道的空白侧边允许转向绕过障碍。
最终网格只保留合并地表、外露高度差墙面与地图外部闭合面，不保留地下重复接触面。
默认地图有 44,916 个顶点和 22,458 个三角形，CPU 检查未发现零面积面。
距离课程迁自 [legged_gym](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)，
Unitree 基线虽然有 Terrain 工具，但原版 base 未接入粗糙地形与等级更新。

| 设置 | 来源 / 决定 |
| --- | --- |
| 4096、H1=10、H2=2、CNN/MLP→Transformer→GRU、五项估计器损失 | [PIE III-B/III-C](https://arxiv.org/html/2408.13740v3) |
| 十项奖励及权重、随机化范围、障碍上限 | PIE 表 I、表 II、III-C |
| 24步、5epochs、4minibatches、1e-3、adaptive、KL0.01、std1 | Unitree 任务配置默认，不称为论文精确超参数 |
| gamma0.99、lambda0.95、clip0.2、entropy0.01、value1、gradclip1 | Unitree 原版配置 |
| 15000轮、保存500轮 | 用户指定 |
| 8×8米、10×20网格、最高初始等级5、距离课程 | 用户选择沿用开源框架；不是论文披露的完整课程 |
| 五类等比例、地形随机宽度/深度、非零最低难度 | 保留现有 PIE 复现选择；布局为原版中心出生点适配 |
| 网络宽度/token128/GRU128/map32/VAE16、策略使用均值 | 保留已有复现选择，来源与理由见 PIE_NETWORK.md |
| 80×60相机、现有安装位姿、100ms固定延迟、帧池/CNN复用/重算 | 工程选择，非作者原版实现 |
| 本体噪声、随机推扰、PhysX缓冲2**23/5 | 原版配置 |

不增加教师蒸馏、导航目标、AMP、图像量化、梯度累积或减小正式训练规模。

## 训练框架与模型

runner 使用原版 `OnPolicyRunner.learn`，PIE 通过输入/生命周期回调扩展。
PPO 复用原版采样接口、损失/KL调度/优化器更新与 RolloutStorage GAE。
PIE storage 补充视觉索引、标签和 actor-only GRU 的时间批次。
Critic 为前馈网络；Actor 仅用本体/视觉及估计值，不能读取真实速度/高程标签。

保持完整24步 recurrent训练，图像仍FP32；训练缓存每个逻辑minibatch独立，
保留CNN/GRU梯度，不缓存detach的采样latent来训练策略。
新模型格式 version3 保存 native states、实际训练/环境配置、累计轮数和随机状态。
不支持旧version1/2模型；同分支续训重新建立仿真回合与GRU记忆。
保存的是 PyTorch/CUDA 随机状态，不是完整仿真状态快照。

整合审查额外修复：原版推扰只更新被选中机器人的缓存；日志上下文不再递归引用
上一轮 rollout；NumPy 运行时参数转为普通 Python 值安全保存；回放重新计算
推扰间隔，并允许保存的四 minibatches 配置在单环境推理时初始化。
实际训练更新仍检查环境数量，未静默减少 minibatches。

相机随 torso 根位姿更新，射线只包含静态地形；当前没有机器人自身或其他动态物体遮挡。
相机安装位姿和粗糙地形布局属于复现选择，不宣称是作者未公开的原版实现。

## 验证记录

本阶段执行静态/CPU检查，不启动仿真或GPU训练。
专用环境由既有 PIE 环境离线克隆，Python 3.8.10、PyTorch 2.4.1；editable 的
`legged_gym` 和 `rsl_rl` 均已核实指向本 worktree，未安装到 base。

CPU 检查结果：**68 passed**。包括真实原版步进方法与奖励注册、dt 缩放、局部重置、
终止前监督、超时 bootstrap 一次、原版 GAE、Actor 信息隔离、GRU 状态与梯度、
FP32 帧池与视觉复用输出/梯度、全部地形墙面法向与高度跨度、原版 policy 参数生效、
新模型保存/续训、安全配置恢复及完整默认网络的单环境推理。
Isaac Gym 接口使用 CPU 替身，仅替代模拟器接口；这些检查没有创建真实仿真器。
GPU峰值显存、4096稳定性、训练速度和行走性能仍待验证。
此前视觉复用文档中的53项结果属于历史源码检查，不能当成本分支的结果。
