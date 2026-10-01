# Lite3 PIE 扩展

网络超参数已按 PIE、DreamWaQ、LocoTransformer 与 Extreme Parkour 的公开资料确定，具体数值、选择理由及工程假设见 [PIE_NETWORK.md](PIE_NETWORK.md)。网络配置保存在本地 rsl_rl 的 `ModelConfig`，随 checkpoint 保存。

直接基于 `unitreerobotics/unitree_rl_gym` 修改，基线 `276801e46c5d433564f24658bac64f254b7d2d4b`。新增 `lite3_pie`，使用原 `legged_gym/scripts/train.py`、`play.py` 和 `task_registry`。`Lite3PIE` 继承原 `LeggedRobot`，保留上游仿真和 actor 创建。

## rsl_rl 源码与独立环境

[中文安装说明](doc/setup_zh.md)和[英文安装说明](doc/setup_en.md)均指定官方 rsl_rl **v1.0.2**，源码基线为 `2ad79cf0caa85b91721abfe358105f869a784121`。PIE 扩展直接修改这份源码；发布仓库将完整源码保存在 `rsl_rl/` 普通目录，克隆 `rl_gym_PIE` 即包含算法扩展。

新环境已改为 editable 安装该源码，原先的 `rsl-rl-lib==2.2.4` 已从新环境卸载。从零配置时，在项目目录执行：

```bash
python -s -m pip install --no-deps --no-build-isolation -e ./rsl_rl
python -s -m pip install --no-deps --no-build-isolation -e .
```

请在专用 Conda 环境中执行安装。现有 `rsl_rl/` 已含扩展，无需重新克隆。父项目只打包 `legged_gym`，rsl_rl 独立安装。

## 最小运行命令（本次未执行）

```bash
conda activate /home/asuka/Legged/parkour/.conda-envs/pie-isaacgym
cd /home/asuka/Legged/parkour/unitree_rl_gym
export TORCH_EXTENSIONS_DIR=/tmp/pie-gymtorch-check
python -s -m pytest -q tests
python -s legged_gym/scripts/check_pie_backend.py
python -s legged_gym/scripts/train.py --task lite3_pie --headless --num_envs 2 --max_iterations 1 --rollout_steps 16 --output_dir runs/minimal
python -s legged_gym/scripts/play.py --task lite3_pie --headless --num_envs 2 --steps 4 --checkpoint_file runs/minimal/checkpoint.pt
```

默认 4096 环境、8 步 rollout、15000 iterations，每完成 500 轮保存 `model_500.pt`、`model_1000.pt` 等，训练结束另存 `checkpoint.pt`。可以通过 `--num_envs`、`--max_iterations`、`--rollout_steps`、`--save_interval` 覆盖。以上最小验证命令显式使用 2 环境、1 轮和 16 步，使 10 Hz / 100 ms 延迟的视觉帧真正进入 rollout。检查脚本不训练。`runs/minimal` 保存的是此前 2.2.4 实现的 checkpoint、指标和日志，不能作为本次 v1.0.2 修改的运行证据；再次在同目录训练会覆盖 checkpoint。

启动时打印 Actor、Critic、本体 MLP、深度 CNN、Transformer、GRU、各估计头及解码器。训练终端调用原版 RSL-RL `OnPolicyRunner.log()`，保留原版速度、采集/更新时间、噪声标准差、回合奖励/长度，以及全部 10 个 PIE 奖励项的 `Mean episode rew_*`。同一张表中追加总损失、速度/足高估计损失、高程图/后继重构损失、VAE KL、裁剪前梯度范数、学习率、单步奖励和本轮 transition 数；同时保留 `metrics.jsonl` 并写入 TensorBoard。JSON 的 `episode_rewards` 保存本轮分项奖励统计。原版 logger 只有一个可选追加文本入口，原任务不传该参数时输出不变。`Mean reward` / `Train/mean_reward` 是最近 100 个完成回合的累计奖励；`Mean step reward` / `Train/mean_step_reward` 和 JSON 的 `mean_reward` 是 rollout 中的单步平均奖励，不能直接比较。各项奖励按原 `LeggedRobot.reset_idx()` 的口径，累计已加权、按控制 dt 缩放的奖励，结束回合时除以配置的最大回合秒数，写入 `Episode/rew_*`。只有完成回合后才显示分项，未结束的回合累计值跨 rollout 保留。`rew_tracking_lin_vel`、`rew_tracking_ang_vel`、`rew_lin_vel_z`、`rew_ang_vel_xy`、`rew_dof_acc` 使用原仓库熟悉的命名，其余保留 PIE 奖励名。`PIE/vae_kl_loss` 是潜变量正则项。随机初始超时计数不计入已采集的回合长度。

当前实际奖励由 `legged_gym/pie/sensors_and_rollout.py::_reward` 计算，以下系数在 `reward_scale_dt=True` 时还乘以控制 dt；日志并未更改这些公式或系数。

| 奖励项 | 日志名 | 系数 |
| --- | --- | --- |
| 平面线速度跟踪 | `rew_tracking_lin_vel` | 1.5 |
| 偏航角速度跟踪 | `rew_tracking_ang_vel` | 0.5 |
| 垂直速度平方 | `rew_lin_vel_z` | -1.0 |
| 横滚/俯仰角速度平方和 | `rew_ang_vel_xy` | -0.05 |
| 姿态偏离平方和 | `rew_orientation` | -1.0 |
| 关节加速度平方和 | `rew_dof_acc` | -2.5e-7 |
| 关节绝对功率和 | `rew_joint_power` | -2e-5 |
| 非足部碰撞数量 | `rew_collision` | -10.0 |
| 动作一阶变化平方和 | `rew_action_rate` | -0.01 |
| 动作二阶变化平方和 | `rew_smoothness` | -0.01 |

仅 Lite3 PIE 的 PhysX 配置预留 `max_gpu_contact_pairs=2**25`、`default_buffer_size_multiplier=10.0`，用于 4096 环境共享地形时的 GPU 碰撞缓冲区；其他机器人任务继续使用原配置。该设置增加物理缓冲区显存占用，与网络超参数无关。

2026-10-01 完成静态代码审查并修复原生配置/时间步长同步、动作延迟上限、非有限奖励和异常清理等问题。此次未执行上述运行命令。[完整审查记录](docs/code_review_2026-10-01.md)。`Lite3PIE` 在创建 actor 前将 `cfg.pie` 的最终 URDF、机器人控制、摩擦/质量随机化和物理步长写入父类使用的配置。

环境为 Python 3.8、Torch 2.4.1、Isaac Gym Preview 4、Warp 1.6.2、项目内 rsl_rl v1.0.2。新增依赖只装在新环境。`environment.pie.yml` 是配方，SDK 需从 NVIDIA 单独安装。当前 SDK 为 `/home/asuka/isaacgym/python`。rsl_rl 必须先于父项目 editable 安装。Isaac Gym native binding 必须先于 Torch 导入。

## 修改位置

| 路径 | 内容 |
| --- | --- |
| `legged_gym/envs/pie/lite3.py` | 原 `LeggedRobot` 的 Lite3 PIE 扩展 |
| `legged_gym/envs/pie/lite3_config.py` | 原 `LeggedRobotCfg` / `LeggedRobotCfgPPO` 配置 |
| `legged_gym/pie/config.py` | 相机、地形、随机化及控制默认值 |
| `legged_gym/pie/warp_camera.py` | Warp 批量环境/像素射线，输出轴向深度 |
| `legged_gym/pie/sensors_and_rollout.py` | 历史、标签、PD、奖励与 reset |
| `rsl_rl/rsl_rl/modules/actor_critic_pie.py` | 继承 v1.0.2 `ActorCritic`，添加 PIE 编码器、GRU、多头估计与解码器 |
| `rsl_rl/rsl_rl/algorithms/ppo_pie.py` | 继承原 `PPO`，使用其优化器与配置，联合循环 PPO 和辅助损失 |
| `rsl_rl/rsl_rl/storage/rollout_storage_pie.py` | 扩展原 `RolloutStorage`，保存视觉历史、标签与真正的终止前状态 |
| `rsl_rl/rsl_rl/runners/on_policy_runner_pie.py` | 继承原 `OnPolicyRunner`，通过原工厂创建模型、算法和存储 |
| `legged_gym/pie/models.py`, `learner.py` | 兼容导入入口，算法实现均在本地 rsl_rl |
| `resources/robots/lite3` | 独立官方 Lite3 URDF、meshes、许可证 |

本体历史 MLP + 两帧深度 CNN → Transformer → GRU，输出速度、四足离地高度、地图 latent 和 VAE latent。actor 使用 posterior mean，辅助解码使用采样 latent，避免 PPO 重算时策略 latent 重采样。单优化器联合更新 PPO 和速度/足高/高程图/后继/KL 损失。

默认本体 45 维、历史 10 帧、深度 2×60×80、critic 235 维、50 Hz 控制、10 Hz 图像。图像无噪声、无滤波；当前默认位姿随机化、固定 100 ms 延迟。地形包括平地、沟壑、高台、栏杆、楼梯。未披露参数是工程默认，不代表作者精确配置。

修改本体历史长度或高程图网格时，需要同步模型的 history、heightmap/critic 维数。原生 critic 观测大小随网格更新，训练与回放入口会检查模型配置是否匹配实际观测。

本次 v1.0.2 迁移仅核查代码、AST 语法、安装元数据与 diff 格式，未运行测试、仿真、回放或训练。此前短更新/回放结果只适用于旧实现。网络尺寸和损失权重保留；动作噪声改用 v1.0.2 原生 `std` 参数，使用正值边界。旧 `policy.actor/critic/log_std` checkpoint 权重有转换入口，但本次未验证加载。PIE optimizer resume、ONNX 导出和 GUI 可视化未实现；原真机部署脚本不适用于新增网络。详细接口见 [rsl_rl PIE 扩展说明](rsl_rl/PIE_EXTENSION.md)。
