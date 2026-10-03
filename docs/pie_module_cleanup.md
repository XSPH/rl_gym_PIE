# PIE 模块收拢记录

日期：2026-10-03；工作树 `rl_gym_PIE_native`；分支 `refactor/pie-native-training`。
本次起点为 `09c2122`。仅修改此 Gym 工作树，使用专用环境 `pie-isaacgym-native`。

## 第一阶段：配置唯一来源

提交：`e6bc709`。

- `Lite3PIECfg` 改为原版嵌套配置类。本体历史在 `env`；关节/足部/根链接在 `asset`；
  高程偏移、障碍类型和上限在 `terrain`；相机参数在 `camera`；附加随机化在 `domain_rand`。
- 删除 `cfg.pie`、环境 `self.config` 和构造时两份配置的同步。任务初始化执行配置校验。
- `asset.file` 同时提供 PhysX 与 URDF FK 的资产入口；原版摩擦、质量、推扰开关与
  `domain_rand.randomize_pie` 独立。
- 原版站立位姿、PD、命令、奖励和地形默认值保持；无效旧配置字段将在目录清理时删除。
- 修改前现有 **68 项 CPU 测试通过**；配置迁移后原有 **68 项仍通过**。
- `tests/fixtures/pie_before_cleanup.json` 从起点源码的 CPU 运行记录有效配置、
  seed=4 地形数组哈希、FK/相机/监督值、seed=101 完整默认网络参数哈希与输出。
  资产路径规范化为原版根目录占位符；删除的无效字段不作为有效配置参与对照。
- 新增 **16 项对照/配置校验/开关独立性检查通过**。地形数组和网络初始权重逐字节相同；
  默认网络连续两步输出逐值相同；FK、相机和辅助标签对照通过。

## 第二阶段：环境与工具合并

提交：`e67765a`。

- 传感器方法合入 `envs/pie/lite3.py`，`Lite3PIE` 只继承 `LeggedRobot`；
  `step/reset/post_physics_step/compute_reward/_prepare_reward_function` 与父类是同一方法。
- `PIETerrain/TerrainAtlas/TerrainSampler` 合入 `utils/terrain.py`，原版 `Terrain` 生成逻辑未改。
  Warp 相机和 URDF FK 移至 `utils/warp_camera.py`、`utils/kinematics.py`。
- yaw、轴角函数合入 `utils/math.py`；乘法、旋转、共轭复用 Isaac Gym 的 xyzw Torch 工具。
  CPU 检查加载 SDK 的纯 Torch 工具源码，不导入 Gym 原生绑定。
- 回放恢复移至 `utils/helpers.py`；脚本引用同步迁移；Warp 仍在传感器初始化时惰性导入。
- **87 项 CPU 检查通过**。地形哈希和默认网络权重/输出继续相同，FK/标签/相机对照通过。
  Isaac Gym 的四元数乘法改变 FP32 运算顺序；90° yaw 相机测试的零分量误差为约
  `2.98e-7`，该断言绝对容差由 `2e-7` 调整至 `5e-7`。

## 第三阶段：删除冗余入口与 version 4

提交：`5f9c33c`。

- 删除整个 `legged_gym/pie/`，包括 JSON 配置加载器、`PIE_ROBOT_URDF` 资产入口与未引用的
  `models.py/learner.py`。任务名、命令行选项、原版环境五元组和 PIE 观测/监督接口保留。
- 删除 RSL `PIERunnerCfg/PPOConfig`、独立 `train/evaluate/seed_everything`；
  测试直接读取原版任务的 `algorithm` 参数字典。`ModelConfig` 保留，
  原版 policy 控制 std、actor/critic 宽度和 activation。
- version 4 保存统一 `environment_cfg`、有效 `model_config`、原版 `train_config`、
  模型/Adam 状态、当前学习率、累计轮数、时间/步数和 PyTorch/CUDA 随机状态。
  续训与回放拒绝 version 1/2/3、缺失和未知版本；没有旧模型转换层。
- 使用保存的当前学习率恢复 Adam；仿真回合、相机队列与 GRU 重新建立，沿用此前续训行为。
- `README_PIE.md/PIE_NETWORK.md/rsl_rl/PIE_EXTENSION.md` 已更新；
  历史审查的源码链接指向发布快照，历史发布清单不作为本分支当前文件哈希。

## 最终检查与运行

- 专用环境：`/home/asuka/Legged/parkour/.conda-envs/pie-isaacgym-native`。
- 最终 **103 passed**：原有 68 项行为覆盖保留，新增配置/基线对照、真实 Isaac Gym
  四元数形状与语义、继承关系、随机化独立性、唯一 URDF 入口、version 4/RNG/旧格式检查。
- 57 个 Python 文件在 Python 3.8 下语法编译通过；运行代码清除旧模块引用。
- 禁止导入 `legged_gym/isaacgym` 时 RSL 仍可加载；确认 PIE Runner 的 `learn`、
  PPO 的 `update`、Storage 的 `compute_returns` 与原版方法是同一对象。
- 原版 `Terrain` 类的 AST 与起点 `09c2122` 完全相同。
- 固定种子地形数组和网络参数哈希一致，连续网络输出相同；FK、关节排序、相机和标签对照通过。
- 检查期间存在 PyTorch CPU autocast 弃用提示，不影响通过结果。

从工作树根目录执行（完整目录图与参数说明见 [README_PIE](../README_PIE.md)）：

```bash
conda activate /home/asuka/Legged/parkour/.conda-envs/pie-isaacgym-native
python -s legged_gym/scripts/train.py --task=lite3_pie --headless
python -s legged_gym/scripts/train.py --task=lite3_pie --headless \
  --resume --checkpoint_file /absolute/path/model_500.pt
python -s legged_gym/scripts/play.py --task=lite3_pie \
  --checkpoint_file /absolute/path/model_500.pt --num_envs 1 --steps 2000
```

CPU 检查命令：

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  python -s -m pytest -q tests --basetemp=/tmp/pie-native-final-tests
```

正式训练规模为 4096 环境、24 步、5 epochs、4 minibatches、15000 轮、每 500 轮保存。

## 本地收拢阶段的验证边界

仅静态/CPU 检查；CPU 替身替代 Gym 原生接口，不创建仿真器。
GPU 仿真、4096 环境性能与显存、服务器同步不在本次范围。

## 后续服务器检查

服务器首次 CPU 检查为 102 passed、1 failed：默认网络参数哈希相同，
前向输出与本地基线最大绝对误差 `4.47e-8`。为允许不同 CPU 内核的 FP32 舍入差异，
输出对照使用 `atol=1e-7, rtol=1e-6`；参数和地形哈希仍严格一致。
GPU 与部署结果另记于服务器验证记录。

GPU 后端检查按环境实际 `max_episode_length` 和计数推导超时步数。
Gym 的 `SimParams.dt` 为 float32，实际控制周期约 `0.01999999955` 秒，
`ceil(0.04 / dt)` 为 3；原版超时条件为严格 `>`，检查需走到计数 4。
此次仅修正检查脚本的固定步数假设，训练环境的时间和终止逻辑未改。
该超时检查固定 Torch/NumPy 种子，并使用检查专用的 0.6 米出生高度，
避免随机关节姿态在几步内碰地/倾斜、提前走入失败终止路径。
正式配置仍为 0.3 米出生高度；短训练使用完整正式随机化与地形。

后续已完成 GitHub/4090 部署、真实 GPU 后端检查、4096 环境两轮更新、续训一轮和
单环境 200 步回放；结果及运行命令见 [4090 验证记录](native_4090_validation_2026-10-04.md)。
