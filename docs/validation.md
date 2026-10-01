# 最小验证结果

2026-10-01：两个项目将网络超参数显式配置化并补充来源说明，此次仅做静态审查。以下历史运行记录不等于当前源码已重新运行验证。

随后对两个项目与 Gym 内 rsl_rl 完成 60 个 Python 文件的静态审查，并修复仿真配置同步、非有限奖励、连续训练状态和异常资源清理等问题。当前两套源码均未针对这些修复运行测试、仿真或训练；以下通过记录继续作为历史结果保存。[代码审查与修复清单](code_review_2026-10-01.md)。

验证日期：2026-09-30。机器人 Lite3；RTX 4060 Laptop GPU，8 GB。只执行少量步进和每个平台一次 PPO 更新，没有大规模训练。[机器可读结果](validation_results.json)，[环境版本](environment_versions.json)。

以下 Isaac Gym 运行结果属于此前 rsl-rl-lib 2.2.4 实现。随后已按宇树安装说明改为本地 rsl_rl v1.0.2；迁移按用户要求只做静态审查，没有运行测试、仿真、训练或回放，当前实现的运行结果尚未验证。旧日志与 checkpoint 原样保留。[迁移记录](rsl_rl_v1_pie_migration.md)。该迁移仅修改 Gym；两平台后续改动的验证状态见本页顶部。

| 验证 | unitree_rl_gym | unitree_rl_mjlab |
| --- | --- | --- |
| CPU 检查 | 10 项通过 | 12 项通过 |
| 新环境依赖检查 | 通过 | 通过 |
| GPU 相机 / reset | 通过 | 通过 |
| 原 train.py | 2 环境 × 16 步 × 1 iteration，退出码 0 | 2 环境 × 16 步 × 1 iteration，退出码 0 |
| PPO / 估计损失 | 全部有限 | 全部有限 |
| 原 play.py | 加载 checkpoint，4 步，退出码 0 | 加载 checkpoint，4 步，退出码 0 |

CPU 检查覆盖：actor 不读取特权输入、循环概率重算、GRU reset 与更新后重放、timeout bootstrap、真实后继标签、辅助标签有效性及联合梯度。mjlab 另核查历史隔离、时序约束、配置读取和 CLI 相机/执行器参数传递。

Isaac Gym GPU 检查以解析平面验证轴向深度：期望 2.0 m，实际 2.0 m；批量两相机一致。实际原生仿真还验证两环境的 timeout 前 action/后继标签被保存，reset 后动作与历史清理。

mjlab GPU 检查使用两环境：前 16 步仅渲染 4 次（reset、step 5/10/15），第 10 步交付第 5 步采集帧，历史帧 ID 为 `[[5,10],[5,10]]`；深度范围约 `[-0.4444,0.5]` 且有限。随后强制临近 time limit，验证 pre-reset 标签、历史与 pending 图像清理。该检查隔离了未训练策略的摔倒终止，只验证传感器和 reset 时序。

训练更新的总 loss 分别约 1.7481 和 3.5041，每边 32 条 transition。这些数值仅说明计算和梯度链路有限，不能用于比较两平台训练质量。梯度范数日志记录裁剪前数值；优化配置执行梯度裁剪。

## 保存的结果

| 项目 | 日志与 checkpoint |
| --- | --- |
| Isaac Gym | [runs/minimal](../runs/minimal)：`train.log`、`play.log`、`backend_check.log`、`metrics.jsonl`、`checkpoint.pt` |
| mjlab | [本次运行目录](https://github.com/XSPH/rl_mjlab_PIE/blob/main/logs/rsl_rl/lite3_pie/2026-09-30_21-53-53_minimal)：上述文件，以及 `cpu_checks.log`、原入口保存的参数 YAML 和上游 commit |

源码基线及操作入口分别见 [Isaac Gym README](../README_PIE.md) 和 [mjlab README](https://github.com/XSPH/rl_mjlab_PIE/blob/main/README_PIE.md)。源码为本地未提交修改，上游 origin 保留；未推送任何仓库。早期独立原型及其旧日志位于归档目录，不作为本次宇树入口验证的证据。

## 未验证范围

checkpoint 是最小更新产物，尚不能行走或跑酷。完整地形课程、多种子收敛、论文成功率、性能加速倍数、真机迁移均未验证。PIE optimizer resume、部署导出、视频和交互 viewer 尚未实现。mjlab 原任务只验证注册导入与兼容配置，没有重新训练所有其他机器人。

新环境使用独立 prefix 和 `PYTHONNOUSERSITE=1`；新增依赖没有安装进主环境或原有环境。Warp / Torch 原生编译缓存属于运行缓存，与环境依赖安装分开。mjlab 的相机降频使用固定 1.6.0 内部接口，升级时需核查。
