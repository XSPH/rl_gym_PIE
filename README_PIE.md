# Lite3 PIE：平地视觉训练

当前分支：`refactor/pie-rsl-native-style`。本分支以 Unitree RL Gym 和随项目保存的
rsl_rl **v1.0.2** 实现 PIE，在全平地上从第 0 轮随机初始化训练，默认恢复 Warp 深度渲染
和真实视觉输入，名义 PD 为 20/0.5。CNN、Transformer、GRU、全部辅助头/损失与
Critic 特权观测保留。相机安装位置、俯角和 FOV 随机化启用，深度噪声仍为 0。
平地、奖励、命令和非视觉随机化保留当前设置。仅加载 version-4 模型，续训必须匹配
视觉模式。恢复改动见 [视觉管线恢复记录](docs/pie_visual_restore_2026-10-10.md)，
历史盲狗实验见 [盲狗实验记录](docs/pie_blind_flat.md)。
网络参数来源见 [PIE_NETWORK.md](PIE_NETWORK.md)，改动与验证边界见
[训练流程记录](docs/native_training_refactor.md)与 [模块收拢记录](docs/pie_module_cleanup.md)。
原训练分支的 GitHub 部署和真实 GPU 短测试见 [4090 验证记录](docs/native_4090_validation_2026-10-04.md)。
这些历史结果不作为本次恢复的 GPU 验证证据。本次只做本地开发和 CPU 检查。

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

## 直接同步到 4090

图形界面可直接启动，会自动打开本地网页：

```bash
./sync_4090_web.py
```

默认地址为 `http://127.0.0.1:8765`。网页提供连接设置、预览/同步按钮、文件变化列表、
实时日志、备份路径和运行记录。修改连接设置会保存在当前浏览器；运行记录在本次
网页服务运行期间保留，最多 20 条。同一时间执行一个任务，页面刷新可恢复最近结果。
关闭服务按 `Ctrl+C`；端口被占用时用 `--port 8766`，只启动服务用 `--no-browser`。
前后端均随仓库提供，服务只监听本机，无需安装前端框架或额外 Python 包。

在本地项目目录运行，不需要 Conda，也不需要先提交或推送 GitHub：

```bash
./sync_4090.py --dry-run   # 先预览
./sync_4090.py             # 正式同步
```

默认连接 `asuka@192.168.1.121`，目标目录 `/home/asuka/rl_gym_PIE_native`。
也可以从其他目录用脚本的绝对路径运行；本地源目录默认是脚本所在的仓库。
换地址或目录时：

```bash
./sync_4090.py --host asuka@10.70.205.234 --remote-dir /home/asuka/rl_gym_PIE_native
```

需要本地 Git、OpenSSH、rsync，远端需要 Git、rsync 和现有项目目录；SSH 使用免密登录。
同步包含受 Git 跟踪的文件及未被忽略的新文件，跳过 `docs/`、`.git/`、日志、模型、
环境和缓存。使用内容校验识别变化，覆盖前的远端文件保存在
`.sync_4090/backups/<时间>/`，同步后再次校验文件内容。
本地当前分支领先 4090 时，工具会先打包并导入这些提交，再同步工作区文件；
因此作者、提交说明和提交历史会一并出现在 4090 的当前分支。两端分支必须同名，
远端工作区须干净且历史须为本地历史的祖先；遇到远端独有提交或历史分叉时会停止，
不会覆盖远端提交。未提交的本地修改仍会直接同步，所以远端 `git status` 可能显示修改。
此过程只在本地与 4090 之间传输，不会推送 GitHub。
它不会自动删除远端文件，本地删除的文件需自行在远端处理；不会重启正在运行的训练。

## 正式训练

从项目根目录执行：

```bash
python -s legged_gym/scripts/train.py --task=lite3_pie --headless
```

也可进入 `legged_gym/scripts` 后执行 `python -s train.py --task=lite3_pie --headless`。
默认 4096 个环境，24 步 rollout，5 epochs，4 minibatches，学习率
1e-3、adaptive、目标 KL 0.01，初始 action std 为 1.0。
训练总目标 15000 轮，每完成 500 轮保存一次，默认输出到
`logs/lite3_pie/<时间>_<run_name>/`，默认 `resume=False`。

本分支模型续训可直接给出文件：

```bash
python -s legged_gym/scripts/train.py --task=lite3_pie --headless \
  --resume --checkpoint_file /absolute/path/model_500.pt
```

恢复模型、Adam、学习率和累计轮数，仿真回合与 GRU 记忆重新开始。
加载第 500 轮后，默认再训练 14500 轮。version 1/2/3 与未知格式均明确拒绝。
默认使用 `depth` checkpoint 续训；旧视觉 checkpoint 缺少 `camera.input_mode`
时解释为 `depth`。`zero` checkpoint 与默认视觉配置不匹配，权重和优化器都不会加载；
如需继续盲狗实验，先将 `camera.input_mode` 设为 `zero`，并使用独立实验日志目录。
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
盲狗实验保存的所有地形列均为平地。play 自动从 checkpoint 恢复 `depth` 或 `zero`
模式，旧版 v4 模型仍恢复原视觉行为；回放旧模型时可用 `--num_envs 5` 观察五类地形。

回放时增加 `--show_depth`，可同时打开机器人深度窗口：

```bash
python -s legged_gym/scripts/play.py --task=lite3_pie \
  --checkpoint_file /absolute/path/model_500.pt --num_envs 1 --steps 5000 \
  --show_depth --depth_env 0
```

窗口并排显示最新真实采集帧、策略历史的较旧帧和最新帧。真实采集帧按米显示；
策略历史直接显示送入网络的数值，归一化输入标注 `normalized input`，使用独立色标。
盲狗策略历史全零，标注 `zero mode`；零值不代表真实距离。
窗口内用左右方向键或 `P/N` 切换机器人，`Esc/Q` 或关闭按钮只关闭深度窗口。
默认选择编号 0；`--depth_env` 可指定初始编号。
显示只读取当前缓冲，不额外捕获、推进相机队列或更改模型输入，刷新上限跟随
配置中的相机捕获频率。此窗口需要 Matplotlib、Tk 和桌面显示。
`--headless --show_depth` 可以只显示深度窗口，关闭 Isaac Gym 第三人称窗口。

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
    depth_viewer.py       # 回放时可选的深度图窗口
rsl_rl/rsl_rl/
  modules/actor_critic_pie.py
  algorithms/ppo_pie.py
  storage/rollout_storage_pie.py
  runners/on_policy_runner_pie.py
```

配置直接修改 `Lite3PIECfg` 的嵌套类。URDF 唯一入口为 `asset.file`。
`domain_rand.randomize_pie` 控制 COM、增益、电机、动作延迟；
独立的 `domain_rand.randomize_camera` 控制相机安装位置、俯角和 FOV，训练默认启用，回放关闭。
`camera.input_mode` 支持 `depth`（当前默认）和 `zero`，非法值初始化时报错。
`depth` 模式自动初始化 Warp 相机、渲染真实深度并编码，经过延迟队列和两帧历史
送入策略 CNN；不需要启用 `render_for_debug` 或 `--show_depth`。
`zero` 模式默认跳过 Warp 相机初始化、渲染和编码，直接使用全零图像，
历史、固定延迟和逻辑帧编号保持原节奏。CNN 和全部辅助损失仍正常计算。
play 的 `--show_depth` 设置 `camera.render_for_debug=True`；在 `zero` 模式下也会
启用真实相机画面，但该模式的策略输入继续为零。此诊断开关不参与视觉模式匹配。
原版摩擦、附加质量和推扰由各自开关控制。正式规模仍为 4096/24/5/4/15000/500。

`train.py → task_registry → PIEOnPolicyRunner.learn → PIEPPO.act → LeggedRobot.step`
使用原版接口。环境返回 `obs, privileged_obs, rewards, dones, extras`。

- `LeggedRobot.step`、奖励计算/注册/episode 累计和 `BaseTask.reset` 直接复用。
- Lite3 子类负责 45 维观测、235 维 Critic、PD 延迟与随机化、传感器和任务重置扩展。
- `get_pie_observations()` 提供本体历史、深度帧与索引；`extras['pie']` 提供重置前标签和诊断。
- 公共 `ppo.py` 和 `on_policy_runner.py` 已完整还原为 v1.0.2 上游文件。PIE 子类自己管理循环和日志，保留原有 PPO 公式、KL 调度和优化顺序；GAE 继续继承原版实现。
- 网络参数直接写在 `Lite3PIECfgPPO.policy`，传入显式构造参数。旧 v4 的 `policy.model_config` 只在 runner/play 入口转换，已有模型与 Adam 状态保持兼容。
- `PIERolloutStorage.Transition` 和时间×环境张量保存采样数据；完整环境轨迹参与 GRU 重放。详见 [RSL 扩展说明](rsl_rl/PIE_EXTENSION.md)。
- 本体 10 帧、深度 2 帧经 MLP/CNN → Transformer → GRU，估计速度、四足离地高度、地图潜变量及 VAE 潜变量。Actor 不读取特权真值。
- 逻辑深度更新为 10 Hz，固定延迟 100 ms、控制 50 Hz。启用相机时跟随 torso，固定安装位置 `[0.25,0,0.06]`、俯角 30°、FOV 87°，深度噪声及椒盐噪声为 0。
- 射线当前只查询静态地形，不包含机器人自身或其他动态物体的遮挡。
- 图像帧池保存唯一 FP32 图像，控制步存索引；同一次逻辑 minibatch 内复用 CNN 计算图，保留梯度及当前激活重算。未启用梯度累积。

## 奖励、地形与命令

奖励由原版 `_prepare_reward_function()` 注册、`compute_reward()` 计算，
权重唯一来源是 `Lite3PIECfg.rewards.scales`。原版已有奖励函数直接复用，
额外函数包括 `joint_power`、`smoothness`、`hip_default` 和 `feet_regulation`。所有项按控制 `dt` 缩放一次，
沿用当前实验的 `only_positive_rewards=False`、目标机身高度 `0.3` 米和初始高度 `0.31` 米。

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
| torques | -1e-4 |
| base_height | -1 |
| hip_default | -0.5 |
| feet_regulation | -0.05 |

`feet_regulation` 使用 [CTS 论文式 (9)](https://arxiv.org/html/2405.10830v2#S3.SS1)：
各脚水平世界速度的平方乘以 `exp(-脚离地高度 / (0.025 * base_height_target))` 后求和，
由负权重惩罚贴地快速运动。奖励计算前刷新刚体状态，脚高度用 PIE 地形采样器查询
每只脚当前位置的地面，再减去 `foot_radius=0.022` 米，得到脚底离地净空；
避免用机身下方地面近似台阶上的脚高度，也不混用世界位移和机身坐标重力。
本项使用训练环境真实状态，不读取相机或网络足高预测，足高定义与辅助足高目标一致。

地形采用原版 Terrain 网格：10 行、20 列、8×8 米地块、中心出生点，
全部地块使用 `kinds=['flat']`、比例 `[1.0]`，关闭地形课程，初始等级为 0。
Warp 和 PhysX 共用最终网格。盲狗实验没有沟壑、高台、障碍或楼梯。

原版命令采样，`heading_command=False`：前向 `[0,1.5]` m/s、侧向 0、
偏航角速度 `[-1.2,1.2]` rad/s，每 10 秒重采样。保留原版小速度归零逻辑。
噪声/推扰启用；PhysX 缓冲沿用原版 `2**23` 和倍率 5。

## 日志与 CPU 检查

保留原版终端布局、episode 奖励、吞吐与时间，追加 PIE 辅助损失、策略 KL、
梯度范数、学习率和一行失败/超时重置汇总。
终端省略单步平均奖励、地形等级、细分重置原因、图像帧池和 CNN 复用统计，
完整指标仍写 TensorBoard 与 `metrics.jsonl`。`Mean reward` 是完成回合累计奖励；
VAE KL 与策略 KL 分开记录。

仅 CPU 检查，不启动 Isaac Gym 仿真：

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  MPLCONFIGDIR=/tmp/pie-visual-mpl \
  python -s -m pytest -q tests --basetemp=/tmp/pie-visual-tests
```

历史盲狗模式的渲染跳过优化见 [不渲染相机的零输入路径](docs/pie_blind_flat_no_render.md)，
该优化仅在显式选择 `zero` 模式时使用。
本次实现检查没有启动 Isaac Gym 仿真或 GPU 训练；CPU 检查只确认数据流、网络更新与恢复逻辑。
基础运动是否学会和单轮耗时降低多少，需要后续训练和回放判断。
