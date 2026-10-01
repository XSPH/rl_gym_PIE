# PIE 实施记录

用户范围：直接修改 `unitree_rl_gym` 与 `unitree_rl_mjlab`，各自独立开发；新建两个 Conda 环境；保留论文 Lite3；仅做最小验证。

1. 已核查 [PIE 原论文](https://arxiv.org/html/2408.13740v3)，结合用户提供的作者摘录，记录论文约束、作者补充和工程默认值。
2. 已克隆并修改真实 [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)（基线 `276801e46c5d433564f24658bac64f254b7d2d4b`）和 [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)（基线 `1425b15f73bd4095f0df53709d7c389c3eb9e790`）。工作区以外的源仓库未修改。
3. Isaac Gym 新增 `Lite3PIE(LeggedRobot)`，沿用上游 `BaseTask`/`LeggedRobot` 的仿真和 actor 创建、`task_registry` 与 `train.py/play.py`，扩展地形、Warp 深度和 rollout。
4. mjlab 新增 `src/tasks/pie/config/lite3`，扩展宇树原 `make_velocity_env_cfg()`，沿用原生注册与 `ManagerBasedRlEnv`，由原训练入口选择 PIE runner。
5. 两边各自保存 Lite3 官方模型、CNN/MLP → Transformer → GRU、多头估计、解码器和循环 PPO。速度、足高、地图、真实后继与 KL 五项损失同阶段更新。
6. 已创建两个新 Conda 环境并分别安装；主环境及原有环境未修改。mjlab 新环境更新至 1.6.0，增加必要的上游资产和碰撞配置兼容修改。
7. 此前已验证 actor 输入无特权信息、终止前后继标签、timeout bootstrap、GRU 重置/重放、相机时延、深度几何及一次短训练和 checkpoint 回放。[验证记录](validation.md)。
8. 按用户后续要求阅读 Isaac Gym 安装说明，克隆官方 rsl_rl v1.0.2 至 `unitree_rl_gym/rsl_rl`，基于原 ActorCritic、PPO、RolloutStorage 和 OnPolicyRunner 扩展 PIE。独立 Isaac Gym 环境改为 editable 安装该源码；此次仅做静态审查，没有重新运行第 7 项检查。[迁移记录](rsl_rl_v1_pie_migration.md)。

范围不包含完整课程评估、多随机种子、论文跑酷成功率、训练速度倍数、部署导出或零样本真机迁移。初期独立原型已归档，后续以两个实际宇树仓库为准。
