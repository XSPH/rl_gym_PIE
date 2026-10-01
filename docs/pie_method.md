# PIE 复刻方法与证据边界

依据 [PIE 论文 v3](https://arxiv.org/html/2408.13740v3) 第 III 节、Fig. 2 和 Table I–III；已检查原始 PDF 图表。作者公开材料未提供全部工程参数，因此这里区分论文约束、作者补充和复刻选择。两个平台应分别保存配置、代码、依赖、checkpoint 和实验记录。

## 论文约束

| 项目 | 约束 |
|---|---|
| 原始机器人 | DEEP Robotics Lite3，12 个关节 |
| 本体观测 | `o = [omega(3), projected_gravity(3), command(3), q(12), qd(12), previous_action(12)]`，45 维 |
| 历史 | 本体 10 帧拼接；深度 2 帧沿通道堆叠 |
| 频率 | 相机 10 Hz；动作 50 Hz |
| Actor 输入 | `[o, velocity_hat, foot_clearance_hat, map_latent, latent]` |
| Critic 输入 | `[o, true_velocity, true_height_scan]` |
| 动作 | 12 维关节目标偏移，`q_target = q_stand + action` |
| 编码器 | 深度 CNN、本体 MLP → 共享 Transformer → 拼接 → GRU |
| 输出 | 速度、足离地高度、地图 latent、VAE latent |
| 解码器 | 全部估计向量 → 下一时刻本体；仅地图 latent → 高度扫描 |
| 优化 | 同阶段 PPO actor/critic 与 estimator 回归 |
| 原始训练 | Isaac Gym，4096 环境，Warp；10000 iterations；RTX 4090 上少于 20 h |
| 地形上限 | 缺口 1 m；台阶/栏杆 0.75 m；楼梯单级 0.25 m |

估计器必须具备以下独立损失，不得用接触分类或摩擦回归代替：

```text
L_est = KL(q(z | proprio_history, depth_history) || N(0,I))
      + MSE(next_proprio_hat, next_proprio)
      + MSE(height_scan_hat, height_scan)
      + MSE(velocity_hat, velocity)
      + MSE(foot_clearance_hat, foot_clearance)
```

公式 (4) 未给各项额外权重。`next_proprio` 是实际下一步，不能误用当前帧或自动 reset 后的观测。Fig. 2 未注明网络宽度、latent 维数、深度分辨率或扫描点数。[论文公式与架构](https://arxiv.org/html/2408.13740v3#S3)。

## 奖励与随机化

| 奖励项 | 实现表达式 | 权重 |
|---|---|---:|
| 平面速度跟踪 | `exp(-4 * sum((cmd_xy - vel_xy)^2))` | 1.5 |
| yaw 速度跟踪 | `exp(-4 * (cmd_yaw - omega_yaw)^2)` | 0.5 |
| 垂直速度 | `vel_z^2` | -1 |
| roll/pitch 角速度 | `sum(omega_xy^2)` | -0.05 |
| 倾斜 | `sum(projected_gravity_xy^2)` † | -1 |
| 关节加速度 | `sum(qdd^2)` | -2.5e-7 |
| 关节功率 | `sum(abs(torque) * abs(qd))` | -2e-5 |
| 非足碰撞 | `nonfoot_collision_count` † | -10 |
| 动作变化 | `sum((a_t-a_previous)^2)` | -0.01 |
| 动作平滑 | `sum((a_t-2*a_previous+a_previous_previous)^2)` | -0.01 |

† Table I 把碰撞项和权重均写为负号，又把倾斜写成完整单位重力向量范数；前者会奖励碰撞，后者为常数。这里按惩罚意图明确修正符号，并取重力 xy 分量。这是复刻解释，不能标记成作者代码。[Table I](https://arxiv.org/html/2408.13740v3#S3.T1)。

| 参数 | Table II 范围 |
|---|---|
| payload | `[-1,2] kg` |
| Kp/Kd/电机强度倍率 | `[0.9,1.1]` |
| CoM 位移 | `[-0.05,0.05] m` |
| 摩擦系数 | `[0.2,1.2]` |
| 初始关节位置 | `[0.5,1.5] rad` † |
| 系统延迟 | `[0,15] ms` |
| 相机 xyz 偏移 | 每轴 `[-0.01,0.01] m` |
| 相机 pitch 偏移 | `[-1,1] degree` |
| 相机水平视场角 | `[86,88] degree` |

† 初始关节位置是绝对角度还是对默认姿态的倍率未解释；Lite3 各关节符号不同，不能把所有关节直接设为同一正角度。命令文字同样未明确平面两个轴分别如何采样，应在配置中记录解释。[Table II 与训练课程](https://arxiv.org/html/2408.13740v3#S3.SS3)。

## 作者补充与资料可访问性

用户提供的 [飞书 PIE 资料](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87) 摘录补充：Warp 深度训练不加图像噪声、不进行滤波或抗锯齿，仅加入延迟与相机位姿随机化；Transformer 后接 GRU 改善步态；显式估计四个抬脚高度并用 latent 编码高程图。直接网页获取重定向至登录页，本文采用用户提供的摘录，未声称从登录页读取全文。

飞书教学伪代码应视作结构示意：足离地高度和高度地图是两个独立标签；实现仍需论文 KL 项、真正的 PPO 概率比/clip/GAE 和 recurrent rollout。公开论文只链接演示视频，本次检索未核验到作者发布的 PIE 训练仓库；其他项目属于参考实现，不能称为 PIE 官方代码。

## 复刻选择与验收

当前实现直接修改 `unitree_rl_gym` 和 `unitree_rl_mjlab` 两个实际上游仓库，保留原训练入口；每套代码与资产独立。Isaac Gym 使用原 `LeggedRobot` 的初始化和 actor 创建流程，mjlab 扩展宇树原速度环境工厂并使用原生 `ManagerBasedRlEnv`。方法实现与最小验收见各项目 `README_PIE.md`。

以下是工程选择，必须随 checkpoint 保存，不能归为论文原始参数：CNN/MLP/Transformer/GRU 宽度、token 网格、latent 大小、深度分辨率/量程/轴深度语义、相机外参、地图扫描范围/坐标系/点数、PD 增益、动作缩放/裁剪、PPO 超参数、课程升级条件和终止规则。建议两平台采用相同物理单位和标签定义，再分别调接触参数。

2026-10-01 已根据用户授权确定网络基线：策略512/256/128、token128、4×4视觉网格、单层4头Transformer、GRU128、动力学latent16、地图latent32。原始来源、每项取舍和解析参数量见两个项目各自的 [Gym网络决策](../PIE_NETWORK.md) / [mjlab网络决策](https://github.com/XSPH/rl_mjlab_PIE/blob/main/PIE_NETWORK.md)。此前隐含在网络代码中的宽度、CNN核/步幅/padding和网格现已进入独立的ModelConfig并随checkpoint保存；默认层尺寸保持一致，本轮仅静态审查，未运行测试或训练。

每个 episode 重置本体/图像历史及 GRU 状态；10 Hz 相机在两次采集之间保持上一帧，不在每个 50 Hz 动作步重复推进图像历史。监督标签和相机历史需与同一 transition 对齐；下一本体使用 reset 前的真实后继帧，仅屏蔽无效标签，各当前标签独立屏蔽，timeout 的 value bootstrap 与普通终止分开处理。序列 minibatch 保留起始隐藏状态与 done mask，避免打散 GRU 时间顺序。PPO 更新后重放本次 rollout 刷新隐藏状态，只重置结束的 episode；rollout 开始的 detached 状态作为截断边界。训练时若采样 VAE latent，PPO 重算应重放同一噪声或明确采用确定性 posterior mean，避免潜变量重采样破坏概率比。

验收分别覆盖实际物理步进、相机可见性与深度平面几何、reset 屏蔽、监督损失下降、短程 PPO 更新、完整地形课程和多随机种子跑酷表现。正式评估应分别报告 gap/stairs/step 的 10 级通过能力及五类感知扰动；论文采用每地形 40 组、每组 100 台机器人。无需把原论文训练耗时当成本实现测量。[论文评估协议](https://arxiv.org/html/2408.13740v3#S4.SS2)。

本轮用户仅要求最小验证：创建两个新的 conda 环境，分别安装依赖、验证模型/观测/深度和少量更新；不向主环境安装，不启动大规模训练。完整课程和正式评估留作后续实验，不能把最小检查表述为已复现论文性能。

两边默认 80×60 轴向深度、0.1–3 m 量程、100 ms 图像延迟和 187 个高程图点均是工程选择。足高标签为脚底相对局部地面的离地高度，不是整幅高度图；两者由不同头/解码器处理。地图标签采用机体高度减地面高度再减 0.5 m 并裁剪的表示。mjlab 电机强度随机化当前修改 effort cap，Isaac Gym 修改 PD 力矩倍率；这一差异没有通过完整训练对照验证。

## Lite3 官方模型接入

[DeepRoboticsLab 模型仓库](https://github.com/DeepRoboticsLab/deep_robotics_model/tree/main/Lite3) 提供 URDF/MJCF，许可证为 BSD-3-Clause，复制模型需保留其 LICENSE。原始文件：`Lite3/urdf/Lite3.urdf`、`Lite3/mjcf/Lite3.xml`；各自引用同目录下 `meshes/` 中的 `torso.STL`、`hip.STL`、`thigh.STL`、`shank.STL`。

关节按 `FL, FR, HL, HR` 各自 `HipX, HipY, Knee` 组织，完整名称如 `FL_HipX_joint`、`FL_HipY_joint`、`FL_Knee_joint`；根 body 为 `TORSO`，足 body 为 `{leg}_FOOT`。HipY/Knee 的轴是负 y，关节范围分别为 `[-2.67,0.314]` 和 `[0.524,2.792]`，不能照搬 Unitree 站姿符号。`[0,-0.8,1.6]` 每腿可作起始站姿候选，须先做静态站立检查，属于复刻默认。[官方 URDF](https://github.com/DeepRoboticsLab/deep_robotics_model/blob/main/Lite3/urdf/Lite3.urdf)。

MJCF 自带 floor/light 和两台旁观相机，应在任务组合时移除重复 floor 并新增 egocentric depth 相机。模型 URDF 力矩上限是髋 24 / 膝 36 Nm，MJCF 电机默认 ±30 Nm；二者也不同于论文报告的膝峰值 30.5 Nm，需统一配置并记录。当前 URDF 各 link 质量相加为 11.9376 kg，论文整机质量为 12.7 kg；相机/计算单元等附载并未独立列入模型。是否补载及补载位置必须显式记录，不能假定当前官方模型就是作者训练资产。这些模型差异不等于 PIE 算法差异。[官方 MJCF](https://github.com/DeepRoboticsLab/deep_robotics_model/blob/main/Lite3/mjcf/Lite3.xml)。
