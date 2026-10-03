# Lite3 PIE：原版训练流程分支

分支：`refactor/pie-native-training`。本分支以 Unitree RL Gym 和随项目保存的
rsl_rl **v1.0.2** 实现 PIE，从头训练，仅加载本分支的 version-4 模型。
网络参数来源见 [PIE_NETWORK.md](PIE_NETWORK.md)，改动与验证边界见
[训练流程记录](docs/native_training_refactor.md)与 [模块收拢记录](docs/pie_module_cleanup.md)。
旧审查文档记录历史版本，不是本分支的运行证据。

## 专用环境

本机开发环境已从既有 PIE 环境离线克隆，依赖版本未调整，未安装到 base：

```bash
conda activate /home/asuka/Legged/parkour/.conda-envs/pie-isaacgym-native
cd /home/asuka/Legged/parkour/rl_gym_PIE_native
```

环境中项目和 RSL 都 editable 指向这个 worktree。其他机器可使用
`environment.pie.yml` 创建专用环境；Isaac Gym Preview 4 SDK 需另行安装。
在项目目录安装本地源码：

```bash
python -s -m pip install --no-deps --no-build-isolation -e ./rsl_rl -e .
```

当前依赖为 Python 3.8、PyTorch 2.4.1、Warp 1.6.2；安装要求的
rsl_rl 版本见 [宇树原版说明](doc/setup_zh.md#23-安装-rsl_rl)。

## 正式训练

从项目根目录执行：

```bash
python -s legged_gym/scripts/train.py --task=lite3_pie --headless
```

也可进入 `legged_gym/scripts` 后执行 `python -s train.py --task=lite3_pie --headless`。
默认 4096 个环境，24 步 rollout，5 epochs，4 minibatches，学习率
1e-3、adaptive、目标 KL 0.01，初始 action std 为 1.0。
训练总目标 15000 轮，每完成 500 轮保存一次，默认输出到
`logs/lite3_pie_native/<时间>_<run_name>/`。

本分支模型续训可直接给出文件：

```bash
python -s legged_gym/scripts/train.py --task=lite3_pie --headless \
  --resume --checkpoint_file /absolute/path/model_500.pt
```

恢复模型、Adam、学习率和累计轮数，仿真回合与 GRU 记忆重新开始。
加载第 500 轮后，默认再训练 14500 轮。version 1/2/3 与未知格式均明确拒绝。
version 4 保存统一环境配置、有效网络和训练配置、优化器、当前学习率、累计轮数、
时间/步数及 PyTorch/CUDA 随机状态；仿真器、相机队列和 GRU 不作为状态快照保存。

## 回放本分支模型

```bash
python -s legged_gym/scripts/play.py --task=lite3_pie \
  --checkpoint_file /absolute/path/model_500.pt --num_envs 1 --steps 2000
```

默认一个机器人、2000 个控制步（40 秒仿真时间）。`--headless` 可用于无窗口回放。
回放从模型恢复环境与网络配置，关闭噪声、随机推扰及课程升级。
策略仍通过原版 `env.step()` 执行，回合结束后清除对应机器人的 GRU 状态；
前五次重置打印 `[PIE reset]` 原因。
原版按环境编号分配地形列，单环境默认落在平地列；需要同时观察五类地形时使用
`--num_envs 5`。这属于回放环境数量选择，不改变训练配置。

## 原版流程与 PIE 扩展

最终目录中任务与工具位于原版对应位置：

```text
legged_gym/
  envs/pie/
    lite3_config.py       # 唯一环境配置：env/asset/control/terrain/domain_rand/camera
    lite3.py              # LeggedRobot 子类，含传感器与任务生命周期扩展
  utils/
    terrain.py            # 原版 Terrain 和 PIE 地形/地图/高度采样扩展
    warp_camera.py        # 传感器初始化时惰性导入
    kinematics.py         # URDF FK
    math.py               # yaw 与轴角
    helpers.py            # 原版通用工具与回放配置恢复
rsl_rl/rsl_rl/
  modules/actor_critic_pie.py
  algorithms/ppo_pie.py
  storage/rollout_storage_pie.py
  runners/on_policy_runner_pie.py
```

配置直接修改 `Lite3PIECfg` 的嵌套类。URDF 唯一入口为 `asset.file`。
`domain_rand.randomize_pie` 控制 COM、增益、电机、动作延迟和相机安装随机化；
原版摩擦、附加质量和推扰由各自开关控制。正式规模仍为 4096/24/5/4/15000/500。

`train.py → task_registry → OnPolicyRunner.learn → PPO.act → LeggedRobot.step`
使用原版接口。环境返回 `obs, privileged_obs, rewards, dones, extras`。

- `LeggedRobot.step`、奖励计算/注册/episode 累计和 `BaseTask.reset` 直接复用。
- Lite3 子类负责 45 维观测、235 维 Critic、PD 延迟与随机化、传感器和任务重置扩展。
- `get_pie_observations()` 提供本体历史、深度帧与索引；`extras['pie']` 提供重置前标签和诊断。
- 原版 runner 的采样/记账/日志流程与 PPO 的损失、KL 调度、优化器和 GAE 复用；PIE 扩展 recurrent 多模态批次及辅助损失。
- 本体 10 帧、深度 2 帧经 MLP/CNN → Transformer → GRU，估计速度、四足离地高度、地图潜变量及 VAE 潜变量。Actor 不读取特权真值。
- 相机跟随 torso 根位姿，保留安装位姿、10 Hz 捕获、100 ms 固定延迟、50 Hz 控制；深度无额外噪声/滤波。
- 射线当前只查询静态地形，不包含机器人自身或其他动态物体的遮挡。
- 图像帧池保存唯一 FP32 图像，控制步存索引；同一次逻辑 minibatch 内复用 CNN 计算图，保留梯度及当前激活重算。未启用梯度累积。

## 奖励、地形与命令

奖励由原版 `_prepare_reward_function()` 注册、`compute_reward()` 计算，
权重唯一来源是 `Lite3PIECfg.rewards.scales`。原版已有八项直接复用，
只新增 `joint_power` 和 `smoothness` 两项函数。所有项按控制 `dt` 缩放一次，
保留原版 `only_positive_rewards=True`。

| 奖励配置名 | 权重 |
| --- | ---: |
| tracking_lin_vel | 1.5 |
| tracking_ang_vel | 0.5 |
| lin_vel_z | -1 |
| ang_vel_xy | -0.05 |
| orientation | -1 |
| dof_acc | -2.5e-7 |
| joint_power | -2e-5 |
| collision | -10 |
| action_rate | -0.01 |
| smoothness | -0.01 |

地形采用原版 Terrain 网格：10 行、20 列、8×8 米地块、中心出生点，
五类等比例的平地/沟壑/高台/障碍/楼梯。障碍上限保留沟壑 1 米、
高台/障碍 0.75 米、楼梯单阶 0.25 米。Warp 和 PhysX 共用最终网格。
使用 legged_gym 距离课程，初始化最高等级为 5；删除旧窄赛道侧边越界和固定 x 终点重置。
地形等级是距离课程的统计，不等同于全部障碍成功率。

原版命令采样，`heading_command=False`：前向 `[0,1.5]` m/s、侧向 0、
偏航角速度 `[-1.2,1.2]` rad/s，每 10 秒重采样。保留原版小速度归零逻辑。
噪声/推扰启用；PhysX 缓冲沿用原版 `2**23` 和倍率 5。

## 日志与 CPU 检查

保留原版终端布局、episode 奖励、吞吐与时间，追加 PIE 辅助损失、策略 KL、
梯度范数、学习率、实时地形等级、重置原因、图像帧池和 CNN 复用统计。
同时写 TensorBoard 与 `metrics.jsonl`。`Mean reward` 是完成回合累计奖励，
单步平均奖励单独记录；VAE KL 与策略 KL 分开记录。

仅 CPU 检查，不启动 Isaac Gym 仿真：

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  python -s -m pytest -q tests --basetemp=/tmp/pie-native-tests
```

本分支的具体检查结果见改动记录。GPU 仿真、4096 环境峰值显存、训练速度和
行走/跑酷效果需要另行验证，CPU 结果不能证明这些效果。
