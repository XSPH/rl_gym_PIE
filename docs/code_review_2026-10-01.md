# PIE 两个宇树项目：代码审查记录

> 历史记录：对应模块收拢前的源码。当前目录、配置与 version 4 检查见 [模块收拢记录](pie_module_cleanup.md)。

日期：2026-10-01。审查覆盖相对宇树源码新增、修改的代码，以及 Gym 项目内的 rsl_rl 扩展。按配置加载、环境创建、观测、动作、奖励、rollout、PPO 更新、checkpoint 的运行顺序阅读，并修复下述问题。

本轮只做静态审查：未执行测试、模型 forward、仿真、训练或回放，未安装依赖。报告中的“已修复”表示源码已修改并经过静态复核；运行行为仍需后续验证。此前的短训练日志属于历史记录。

## 范围与源码基线

| 仓库 | Git HEAD 基线 | 新增或修改的 Python 文件 |
| --- | --- | --- |
| `unitree_rl_gym` | `276801e46c5d433564f24658bac64f254b7d2d4b` | 21 |
| `unitree_rl_gym/rsl_rl` | v1.0.2，`2ad79cf0caa85b91721abfe358105f869a784121` | 9 |
| `unitree_rl_mjlab` | `1425b15f73bd4095f0df53709d7c389c3eb9e790` | 30 |

共 60 个 Python 文件，包含配置、注册入口、核心实现、兼容层、检查脚本及现有测试源码。另核查了安装配方、README、网络说明、Lite3 XML/URDF 的资源路径、关节配置和资产来源。用户的 `.vscode/` 未修改；归档原型不属于当前宇树实现。

## 发现并修复的问题

P1 表示会破坏训练数据、模型状态或关键仿真语义；P2 表示配置、边界行为或资源管理错误；P3 表示入口校验和接口一致性问题。各条均已修改源码。

| 编号 / 优先级 | 触发条件与影响 | 修复位置与行为 |
| --- | --- | --- |
| R01 / P1 | Gym 设置自定义 URDF 时，FK 使用最终解析路径，父类可能仍加载默认 `cfg.asset.file`；物理模型与监督标签可能不一致。机器人初始姿态、控制配置也存在两份来源。 | [Lite3 初始化](https://github.com/XSPH/rl_gym_PIE/blob/7911fff5479aeb295f2a2349c6ec9f9a0cd17ac1/legged_gym/envs/pie/lite3.py#L24)：在父类创建 actor 前同步最终 URDF、站姿、基座高度、PD 与控制参数。 |
| R02 / P1 | 修改 Gym `pie.physics_dt` 后，PD/时延/奖励使用新步长，PhysX 的 `SimParams.dt` 仍可能保持旧值。 | [仿真时间步长](https://github.com/XSPH/rl_gym_PIE/blob/7911fff5479aeb295f2a2349c6ec9f9a0cd17ac1/legged_gym/envs/pie/lite3.py#L37)：创建仿真前写入最终 PIE 时间步长。 |
| R03 / P2 | Gym PIE 的摩擦、附加质量范围或随机化开关改变后，父类仍可能按原生 `domain_rand` 值创建 actor。关闭随机化后再 reset，也可能保留旧增益、时延和相机扰动。 | [原生随机化同步](https://github.com/XSPH/rl_gym_PIE/blob/7911fff5479aeb295f2a2349c6ec9f9a0cd17ac1/legged_gym/envs/pie/lite3.py#L32)与[reset](https://github.com/XSPH/rl_gym_PIE/blob/7911fff5479aeb295f2a2349c6ec9f9a0cd17ac1/legged_gym/pie/sensors_and_rollout.py#L161)：同步摩擦/质量配置；关闭时恢复增益、电机倍率、动作时延、相机位姿/FOV，并取消随机横向出生偏移。摩擦/质量仍是创建 actor 时的随机化。 |
| R04 / P2 | Gym 动作延迟队列用向上取整。上限 15 ms、物理步长 6 ms 时，原实现可采样 18 ms，超过配置上限。 | [动作队列容量](https://github.com/XSPH/rl_gym_PIE/blob/7911fff5479aeb295f2a2349c6ec9f9a0cd17ac1/legged_gym/pie/sensors_and_rollout.py#L75)：使用向下取整，本例最大离散延迟为 12 ms；负延迟及逆序随机化范围由配置校验拒绝。 |
| R05 / P1 | 非有限物理状态已经触发终止时，NaN/Inf 奖励仍可能进入 episode return、GAE 和 PPO。mjlab 的旧终止检查也未覆盖速度、加速度和执行器力。 | [Gym step](https://github.com/XSPH/rl_gym_PIE/blob/7911fff5479aeb295f2a2349c6ec9f9a0cd17ac1/legged_gym/pie/sensors_and_rollout.py#L283)、[mjlab step](https://github.com/XSPH/rl_mjlab_PIE/blob/main/src/tasks/pie/backend.py#L162)、[mjlab fallen](https://github.com/XSPH/rl_mjlab_PIE/blob/main/src/tasks/pie/mdp.py#L79)：隔离非有限奖励，将相关 transition 标为真正终止，取消该 transition 的 timeout bootstrap；补全状态检查。正常 transition 的奖励公式与权重保持原值。 |
| R06 / P1 | mjlab 同一 runner 连续调用两次 `learn()`，第二次会重新初始化模型、Adam、环境历史及迭代计数，丢失先前训练结果。 | [mjlab runner / train](https://github.com/XSPH/rl_mjlab_PIE/blob/main/src/tasks/pie/rl/learner.py#L56)：保留模型、优化器、观测、GRU 状态、reset mask，累计 iteration；standalone `train()` 默认仍创建一次新运行。 |
| R07 / P2 | mjlab 的分布计算裁剪 `log_std`，但参数本身可能被优化到区间之外，导致裁剪后的梯度持续为零。 | [PPO 更新](https://github.com/XSPH/rl_mjlab_PIE/blob/main/src/tasks/pie/rl/learner.py#L208)：每次 Adam 更新后将参数投影回 `[-5, 2]`。Gym 原生 `std` 已有相应正值边界。 |
| R08 / P2 | 环境创建成功后，wrapper/runner 构造、日志写入、更新或 checkpoint 加载抛错，部分入口无法关闭环境。mjlab adapter 的 `sense` hook 也未恢复。 | [Gym train](https://github.com/XSPH/rl_gym_PIE/blob/7911fff5479aeb295f2a2349c6ec9f9a0cd17ac1/legged_gym/scripts/train.py#L11)、[Gym 构造](https://github.com/XSPH/rl_gym_PIE/blob/7911fff5479aeb295f2a2349c6ec9f9a0cd17ac1/legged_gym/envs/pie/lite3.py#L48)、[mjlab train](https://github.com/XSPH/rl_mjlab_PIE/blob/main/scripts/train.py#L98)、[mjlab play](https://github.com/XSPH/rl_mjlab_PIE/blob/main/scripts/play.py#L70)、[mjlab 检查脚本](https://github.com/XSPH/rl_mjlab_PIE/blob/main/scripts/check_pie_backend.py#L16)、[adapter 生命周期](https://github.com/XSPH/rl_mjlab_PIE/blob/main/src/tasks/pie/backend.py#L67)：扩展 `try/finally` 覆盖范围；构造失败或关闭时恢复原 `sense`，自建环境由 adapter 清理，传入环境由调用方清理。 |
| R09 / P2 | Gym Warp 相机接收 CPU `env_ids` 时，只转换 int32，随后将 CPU 数组传给 CUDA kernel，设备不匹配。默认训练路径的 IDs 原本在 GPU 上。 | [Warp render](https://github.com/XSPH/rl_gym_PIE/blob/7911fff5479aeb295f2a2349c6ec9f9a0cd17ac1/legged_gym/pie/warp_camera.py#L61)：同时转换设备和 dtype。 |
| R10 / P2 | mjlab 选择可选 Go1 profile 时，非足碰撞名单遗漏四个 hip，碰撞惩罚覆盖不完整。 | [机器人 profile](https://github.com/XSPH/rl_mjlab_PIE/blob/main/src/tasks/pie/robots.py#L49)：补齐 XML 中实际存在的四个 hip collision；默认 Lite3 路径不受该项影响。 |
| R11 / P2 | 指定不同 RL/仿真设备时，Gym 会在后续运算中失败，mjlab 参数此前被忽略。当前 adapter 未实现跨 GPU 观测搬运。 | [Gym runner](../rsl_rl/rsl_rl/runners/on_policy_runner_pie.py#L28)、[mjlab runner](https://github.com/XSPH/rl_mjlab_PIE/blob/main/src/tasks/pie/rl/learner.py#L42)：在构造时明确检查两者相同，并解析 RL 参数中省略编号的 `cuda`。 |
| R12 / P3 | `rollout_steps=0`、play `num_envs=0` 被 `or` 默认值悄悄覆盖；Gym `--resume` 会进入不兼容的原 checkpoint 查找流程；mjlab PIE play 对部分 viewer/video/dummy-agent 参数静默忽略。 | [Gym 注册入口](https://github.com/XSPH/rl_gym_PIE/blob/7911fff5479aeb295f2a2349c6ec9f9a0cd17ac1/legged_gym/utils/task_registry.py#L118)、[mjlab play](https://github.com/XSPH/rl_mjlab_PIE/blob/main/scripts/play.py#L58)：用 `None` 判断默认值，让非法零值进入校验；明确拒绝未实现的功能；正常传递 `no_terminations`。 |
| R13 / P2 | 修改 Gym 高程图扫描网格后，原生 privileged-observation 维数仍固定为 235，即使同步了模型配置，也无法正确构建 storage/model。两平台的本体历史或地图大小与模型配置不匹配时，也缺少统一的入口校验。 | [Gym 观测维数](https://github.com/XSPH/rl_gym_PIE/blob/7911fff5479aeb295f2a2349c6ec9f9a0cd17ac1/legged_gym/envs/pie/lite3.py#L38)、[模型维度校验](../rsl_rl/rsl_rl/modules/actor_critic_pie.py#L66)及 [mjlab 对应校验](https://github.com/XSPH/rl_mjlab_PIE/blob/main/src/tasks/pie/rl/models.py#L67)：Gym 原生 critic 维数改为 `48 + map_size`；训练与回放入口在网络执行前检查实际本体、历史、深度 history、地图、critic 和动作维数。默认保持 `proprio=45 / critic=235 / map=187 / action=12`。 |

另补齐 mjlab `log_dir=None` 的无文件输出分支，以及 Lite3 的[资产来源说明](https://github.com/XSPH/rl_mjlab_PIE/blob/main/src/assets/robots/lite3/SOURCE.md)。这些修改没有增加网络层、改变奖励权重或替换 PPO 目标。

## 按训练顺序的复核结论

| 阶段 | 核查内容 | 静态结论 |
| --- | --- | --- |
| 1. 注册与配置 | 两个原 train/play 入口、任务注册、CLI 覆盖、独立包安装、rsl_rl 版本 | Gym 继续使用项目内 v1.0.2；mjlab 使用独立环境中的 5.4.2 MLP。两个项目无相互 Python 导入。修复 R01–R03、R11–R13。 |
| 2. 资产与环境初始化 | 12 关节 canonical/native 映射、站姿、关节/力矩限幅、FK、地形实例及出生坐标 | 未发现新的确定错误。两套 Lite3 资产的关节名称匹配，引用的四种 mesh 均存在。Gym 父类创建 actor 时使用 `env_origins` 覆盖初始位置，原来的 base-height 偏移应保留，不能按“双重加高”删除。 |
| 3. reset 与随机化 | 按 env ID 重置、关节/动作/GRU 历史隔离、增益/摩擦/质量/相机扰动与动作时延 | 修复 R03–R04。Gym 的创建期摩擦/质量与 reset 期随机化作用时机不同，动态关闭 reset 期随机化不会还原已创建 actor 的质量或摩擦。 |
| 4. 视觉和本体观测 | 相机坐标、轴向深度、归一化、图像降频/时延、history、地图和足高标签 | 除 R09 外未发现新的确定错误。mjlab 先交付旧 pending frame，再覆盖新 frame；局部 reset 只重置指定环境的 history。原生 raycast 监督保持控制频率。默认深度 history 是 `2×60×80`，本体 history 是 `10×45`。 |
| 5. 估计器与策略 | MLP/CNN/Transformer/GRU、各 head、两个 decoder、actor/critic 输入与梯度 | 两个 `ModelConfig` 的 25 个字段和方法 AST 一致，R13 加入模型/环境尺寸检查。actor 只编码本体/历史/深度及估计输出，特权速度/地图进入 critic 和辅助监督。足高 head/标签均为 4 维，地图 decoder/标签默认 187 维。策略使用 posterior mean、Transformer dropout 为 0，避免 PPO 重算引入 latent/dropout 随机变化。 |
| 6. 动作与奖励 | 原始采样动作与执行动作、PD/执行器、clip、action rate/二阶平滑、dt 缩放、终止 | 修复 R02、R04、R05、R10。存储原始采样动作与其概率，环境执行裁剪动作；两者用途明确。正常奖励没有因本轮审查而改权重。 |
| 7. rollout 与 GAE | 当前状态监督与真实后继、auto-reset 前 snapshot、terminated/truncated 区别、标准化 | 未发现新的确定错误。后继标签在 reset 前复制；timeout 使用终止前 critic 值 bootstrap，真正终止不 bootstrap；二者均截断跨 episode 的 GAE trace。 |
| 8. recurrent PPO 与辅助更新 | 概率 ratio/clipping、value clipping、完整轨迹 minibatch、reset mask、联合 Adam、梯度裁剪、VAE/KL | 除 R07 外未发现新的确定错误。保留时间顺序，不随机打散 GRU 单步；每个 label 独立判定有效性，坏后继不会丢弃有效的当前速度/地图/足高监督。 |
| 9. 下一轮训练 | 更新权重后重放 rollout、GRU 边界 detach、继续 episode、重复 learn | 两套实现都在更新后重算 rollout 的 hidden，并按最后一步 done mask 清理。修复 R06；每次 PPO 更新无需重置整个环境。 |
| 10. 保存、回放与退出 | checkpoint 配置与权重 key、旧 Gym 权重迁移、加载路径、异常资源清理 | 修复 R08、R12。mjlab checkpoint 的累计 iteration 与持续训练状态一致。optimizer/episode 的磁盘 resume 仍未实现；旧 checkpoint 兼容入口本轮未实际加载。 |

## 覆盖文件清单

下面只列相对基线新增或修改的 Python 文件；原生库源码也作为 API/调用顺序的阅读依据。

Gym（21）：

```text
legged_gym/envs/__init__.py
legged_gym/envs/pie/{__init__,lite3,lite3_config}.py
legged_gym/pie/{__init__,config,kinematics,learner,math_utils,models,sensors_and_rollout,terrain,warp_camera}.py
legged_gym/scripts/{check_pie_backend,play,train}.py
legged_gym/utils/{helpers,task_registry}.py
setup.py
tests/{test_learning,test_terrain_and_assets}.py
```

项目内 rsl_rl（9）：

```text
rsl_rl/algorithms/{__init__,ppo_pie}.py
rsl_rl/modules/{__init__,actor_critic_pie}.py
rsl_rl/runners/{__init__,on_policy_runner,on_policy_runner_pie}.py
rsl_rl/storage/{__init__,rollout_storage_pie}.py
```

mjlab（30）：

```text
scripts/{check_pie_backend,play,train}.py
setup.py
src/assets/robots/compat.py
src/assets/robots/unitree_a2/a2_constants.py
src/assets/robots/unitree_as2/as2_constants.py
src/assets/robots/unitree_g1/{g1_23dof_constants,g1_constants}.py
src/assets/robots/unitree_go2/go2_constants.py
src/assets/robots/unitree_h1_2/h1_2_constants.py
src/assets/robots/unitree_h2/h2_constants.py
src/assets/robots/unitree_r1/r1_constants.py
src/tasks/pie/{__init__,backend,env_config,history,mdp,reproduction,robots,terrains}.py
src/tasks/pie/config/__init__.py
src/tasks/pie/config/lite3/{__init__,env_cfgs,rl_cfg}.py
src/tasks/pie/rl/{__init__,learner,models}.py
tests/{test_backend_contract,test_learning}.py
```

原有八组机器人 constants 的兼容导入、相关 collision mask 与 `compat.update_assets` 也已阅读；它们不等于所有原机器人任务已运行验证。测试源码只阅读和解析，没有执行，也未新增测试。

## 实际完成的静态检查

- 使用专用 Gym 环境 Python **3.8.10** 对 21+9 个新增/修改 Python 文件执行 `ast.parse`；全部成功。
- 使用专用 mjlab 环境 Python **3.11.16** 对 30 个新增/修改 Python 文件执行 `ast.parse`；全部成功。
- 两个 `ModelConfig` 声明的 AST 比较一致，共 25 个配置字段。
- 标准库 XML 解析成功；两份 Lite3 模型分别有 12 个匹配策略命名的活动关节；所有 mesh 引用可在对应项目内解析。
- 三个独立 Git 仓库的 `git diff --check` 均通过。
- 对 mjlab 1.6.0 的 environment/simulation/entity/action API 做源码复核，确认 adapter 使用的属性、reset 与 sense 调用顺序；没有导入执行这些组件。

AST 和 diff 检查不能证明仿真稳定性、CUDA kernel 行为、运行时加载、梯度数值或收敛性能。完整课程、大规模训练、真机、跨 GPU 和交互视频仍未验证或实现。训练抛错后入口会关闭环境；本轮没有实现复用失败 runner 的自动恢复。

论文未披露的尺寸、梯度边界等仍是已说明的复现选择，参见[Gym 网络说明](../PIE_NETWORK.md)、[mjlab 网络说明](https://github.com/XSPH/rl_mjlab_PIE/blob/main/PIE_NETWORK.md)和[方法差异](pie_method.md)。两平台在加速度取样、电机强度实现、地形/课程、终止阈值及图像时延控制上的工程差异沿用现有设计。作者未公开实现的细节仍无法逐行核实。
