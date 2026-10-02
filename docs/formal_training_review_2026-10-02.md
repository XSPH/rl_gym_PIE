# PIE 正式训练配置与参数来源审查

日期：2026-10-02。两仓库独立实现。本次恢复原版训练参数并做CPU检查，没有启动正式GPU训练、仿真或真机测试。用户已要求先不连接服务器；本次成果在本地与GitHub，服务器源码/正在运行的进程不因本次修改自动更新。

比较基线：unitree_rl_gym 276801e46c5d433564f24658bac64f254b7d2d4b；Gym项目内RSL-RL官方v1.0.2 2ad79cf0caa85b91721abfe358105f869a784121；unitree_rl_mjlab 1425b15f73bd4095f0df53709d7c389c3eb9e790。mjlab环境仍使用已接入相机的1.6.0/RSL-RL5.4.2，与原上游依赖版本不同；本次没有重装依赖或降级。

“恢复原版”表示恢复对应上游项目的训练基线，不表示PIE作者使用了这些数字。“参考其他论文”只表示存在可核查的选择依据，不表示证明了它适用于PIE。“常用默认”限于核实到的项目，不能称为所有开源项目的共识。

## 1. 本次恢复的参数与功能

| 项目 | 之前的最小配置 | 本次正式配置 | 实际生效路径 |
| --- | --- | --- | --- |
| 新训练动作std | 0.5 | 1.0，两平台均直接优化逐动作std | Gym原生ActorCritic.std；mjlab原生GaussianDistribution scalar |
| PPO学习率 | 3e-4 | 1e-3 | Adam实际参数组 |
| PPO epoch | 2 | 5 | 循环更新实际次数 |
| minibatch | 2 | 4 | 按完整环境轨迹分组，保护GRU时间顺序 |
| rollout步数 | 8 | 24 | 原训练入口和独立helper均使用24默认 |
| 学习率调度/策略KL目标 | fixed或未实现；None | adaptive；0.01 | 使用旧/新Gaussian实际KL调整Adam LR；独立记录policy_kl |
| 本体噪声 | 关闭；Gym开启分支幅度也被改 | 开启；恢复各原版IMU/关节幅度 | Gym45-D向量；mjlab原生ObservationTerm噪声 |
| 随机推扰 | 关闭/删除/无调用 | Gym每15s最大平面速度1m/s；mjlab原事件每5–6s | Gym索引root setter；mjlab原生interval event |
| 初始地形最高等级 | 1 | 5 | 首次初始化可采样0–5，后续课程仍0–9 |
| 优势标准化 | 总体std，下限1e-6 | 原生样本std+1e-8 | 单transition显式smoke例外用零方差有限回退 |
| 任意动作std边界 | exp(-5)..exp(2) | Gym取消上界，仅dtype epsilon正值保护；mjlab原生边界 | 不再人为限制探索范围到旧值 |
| Gym命令刷新 | 5s | 10s | 命令实际resample |
| Gym命令观察缩放 | 原样拼接 | [2,2,0.25] | 当前本体和历史 |
| Gym动作/观测裁剪 | 动作±4；无通用观测clip | 动作±100；本体/critic±100 | 原版normalization默认；监督真值不截成100 |
| Gym总奖励截断 | 可为负 | only_positive_rewards=True，总和最低0 | 10项权重保持；负奖励分项日志仍保留 |
| Gym出生base速度 | 清零 | 线/角速度各分量[-0.5,0.5] | reset root tensor，randomization禁用时仍确定性 |
| mjlab actor/critic归一化 | 缺失 | 原生EmpiricalNormalization | checkpoint保存统计；原始监督标签不被归一化 |
| mjlab本体单位缩放 | omega×0.25，qd×0.05 | 原生物理单位，即×1/×1 | 新训练输入；旧模型评估读取保存的旧缩放 |
| mjlab编码器偏置 | 被移除 | [-0.015,0.015] | 原生startup事件 |
| mjlab命令刷新/站立/直行比例 | [4,8]s/0.1/0.3 | [3,8]s/0.05/0.0 | 原生velocity command |
| mjlab策略/关节目标裁剪 | backend±4且target clip | 原生clip_actions=None，移除额外target clip | 不删除正常力矩/执行器限幅 |
| mjlab求解器/容量/摩擦锥 | 50/64/256/elliptic | 原环境工厂10/35/1500/pyramidal | iterations/nconmax/njmax/cone |
| mjlab高程扫描距离 | 4m | 5m | height_scan；新增足高传感器仍4m |
| mjlab种子/随机回合进度 | seed0/False | seed42/True | 原训练入口 |
| mjlab环境数/总轮次/保存 | 2/1/仅结束保存 | 4096/15000/500轮 | 用户要求，同时应用两仓库 |
| checkpoint续训 | 明确不支持 | 权重、Adam/LR、累计轮次；mjlab含归一化统计 | 当前配置决定epochs等；旧checkpoint不能悄悄恢复最小PPO配置 |
| 相机rollout生命周期 | 旧轨迹可能在新采集期间被额外引用 | 更新/记录后释放调用方batch | 防止两整段图像轨迹同时存在，不降低采样量 |

Gym的PPO配置现在由原policy/algorithm字段同步到PIE实际模型/优化器，避免“配置显示原值但实际用独立小配置”。两边训练入口将max_iterations作为总目标，model_500续训到15000只再运行14500轮。原子保存保留500/1000等编号模型；结束另存checkpoint.pt。续训开始新仿真回合和GRU记忆，不宣称精确重放未完成回合。新训练std=1.0；续训恢复已学习std，不把探索参数强行重置。

mjlab归一化层/统计公式来自固定的原生RSL版本；更新统计的调用时机仍是PIE适配选择：在rollout边界更新，采集与PPO重算期间冻结，以保护old/new log likelihood一致性。原生PPO通常每次env step更新统计；不能说两条管线完全相同。

## 2. 沿用的原版值与用户明确设置

| 参数 | 当前值 | 来源/是否PIE原文数值 |
| --- | --- | --- |
| gamma / GAE lambda | 0.99 / 0.95 | 原版RSL/Unitree；DreamWaQ也明确采用 |
| PPO clip | 0.2 | 原版RSL/Unitree；DreamWaQ有数值依据 |
| entropy / value系数 | 0.01 / 1.0 | 上游默认；PIE未给这两个PPO系数 |
| 梯度norm上限 | 1.0 | 上游默认，PIE未披露 |
| clipped value loss | True | 上游默认，PIE未披露 |
| Adam | 默认betas(0.9,0.999)、eps1e-8、weight_decay0 | PyTorch默认，上游无额外覆盖，不等于PIE披露 |
| actor/critic隐藏层 | [512,256,128]，ELU | 两上游任务原值；DreamWaQ图1/IV-A同样支持 |
| physics dt/decimation | 0.005s / 4 | 上游值；产生50Hz policy控制 |
| 回合长度 | 20s | 上游原值，PIE未披露 |
| action scale | 0.25 | 两平台基线动作项默认；PIE公式未给归一化动作比例 |
| Gym PhysX contact pairs/buffer multiplier | 2**23 / 5 | 已继承原版；先前增大到2**25/10的覆盖保持撤销 |
| Gym原版solver/contact设置 | 原LeggedRobotCfg.sim.physx全部继承 | 无新倍率/内存配置 |
| 4096环境 | 4096 | PIE有此数值，用户也明确指定 |
| 总训练轮次 | 15000 | 用户明确要求；PIE报告10000，不声称两者相同 |
| 保存间隔 | 500 | 用户明确要求；不是PIE论文值 |

不得把Lite3的PD30/0.8、站姿与模型机械恢复为Go2的20/0.5，因为上游无Lite3对应任务。这些保留的机器人参数另列在工程选择中。

## 3. PIE明确给出而保留的内容

依据[PIE正文与表I/II](https://arxiv.org/html/2408.13740v3)：本体45维、动作12维、本体历史10帧、深度2帧；CNN/MLP→跨模态Transformer→拼接→GRU；速度3维、足高4维、地图latent与动力学VAE；高程图与后继本体重构。相机10Hz、策略50Hz；渐进地形课程、gap1m、step/hurdle0.75m、stairs0.25m、前向命令0..1.5m/s与yaw±1.2rad/s保留。该结构不意味着所有模块层宽都在论文中公布。

| 奖励 | 当前权重（两平台） |
| --- | --- |
| 平面速度 / yaw速度跟踪 | 1.5 / 0.5 |
| 垂直速度 / 横滚俯仰角速度 | -1 / -0.05 |
| 倾斜 / 关节加速度 | -1 / -2.5e-7 |
| 关节功率 / 非足碰撞 | -2e-5 / -10 |
| 动作变化 / 二阶平滑 | -0.01 / -0.01 |

随机化范围保留payload[-1,2]kg、PD/电机倍率[0.9,1.1]、CoM±0.05m、摩擦[0.2,1.2]、系统延迟0..15ms、相机位置±0.01m、pitch±1°、HFOV86..88°。初始关节[0.5,1.5]在论文表中带rad单位，当前沿用native按站姿乘倍率的实现，是解释选择；不能声称量纲与分布已经完全核对。论文Eq4辅助项无额外显式系数，当前相对系数为1；它们与PPO合成后的整体estimation_weight=1、MSE各维度取mean等仍属实现选择。

深度图不加噪声/滤波参考用户提供的作者说明；这是作者答复，不能冒充Table II的内容。当前延迟固定100ms，不是论文披露值。30.5Nm可由论文机器人硬件说明支持峰值膝力矩，但模型内对不同关节的执行器上限安排仍需要机器人资产适配。

奖励表权重一致，计算不宣称与作者未公开代码完全一致。重力惩罚解释为xy分量；碰撞按非足刚体/geom接触而非枚举全部接触点；Gym阈值1N；策略动作有0.25缩放；两边奖励按control dt缩放。Gym已恢复负总和截为0，mjlab原生RewardManager不采用这个Gym选项。这些会改变最终奖励，不能用权重表一致掩盖。原论文中的orientation和collision符号存在歧义，代码按惩罚意图解释。

## 4. 参考其他论文或作者代码保留的参数

| 参数 | 本实现 | 可核查依据与限度 |
| --- | --- | --- |
| actor/critic层宽与ELU | 512/256/128，ELU | [DreamWaQ图1、IV-A](https://arxiv.org/pdf/2301.10602)；也等于Unitree原值 |
| 动力学latent | 16 | DreamWaQ图2直接给16；不是PIE披露的latent维数 |
| token维数 | 128 | [LocoTransformer附录B.5](https://arxiv.org/html/2107.03996#A2.SS5)明确128 |
| 视觉token网格 | 4×4 | LocoTransformer实现说明有4×4特征网格；本实现用adaptive average pooling达到该网格 |
| CNN前两层通道/核 | 32/64，5/3 | [Extreme Parkour作者depth_backbone](https://github.com/chengxuxin/extreme-parkour/blob/main/rsl_rl/rsl_rl/modules/depth_backbone.py)直接给出；其max-pool/层数与本实现不同 |
| 地图latent | 32 | [Extreme Parkour配置scan_encoder_dims](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/envs/base/legged_robot_config.py)末层32；本项目用于PIE高程图表示是迁移选择 |
| LR/gamma/lambda/PPO clip | 1e-3/.99/.95/.2 | DreamWaQ IV-B明确给值；本次恢复依据首先仍是Unitree原版 |

这不等于整个网络复用了某篇论文：LocoTransformer的Transformer是2层，本实现是1层；Extreme Parkour的视觉GRU输入32、隐藏512，本实现输入2176、隐藏128；DreamWaQ的CENet encoder是128/64，后继decoder64/128，本实现另作视觉融合结构。

## 5. 哪些是常用默认，哪些不能称为共同默认

在两Unitree任务、RSL-RL及所核实Extreme Parkour配置中可看到gamma=.99、lambda=.95、clip=.2、5epochs/4minibatches、adaptive/KL=.01、rollout24、std1、actor/critic512/256/128、ELU、梯度上限1等重复设置。只能称为这一系列腿足RL实现的常用起点。

学习率并非统一：Unitree与DreamWaQ是1e-3，Extreme Parkour的PPO配置是2e-4，LocoTransformer给1e-4和3epochs。噪声、动作clip、GRU宽度、相机参数、保存周期、最大迭代、RewardManager和负总奖励处理也不统一。例如Extreme Parkour的noise开关False且action clip1.2，不能为了恢复Unitree默认而同时声称与所有视觉项目一致。

## 6. 保留的工程选择及与参考项目的区别

| 项目 | 当前自行确定的值/实现 | 为什么保留；哪些未经证明 |
| --- | --- | --- |
| 本体encoder | 450→512→256→128 | 历史450维压缩到token128；不同于LocoTransformer256/256或DreamWaQ128/64 |
| CNN整体结构 | 2→32→64→128；kernel5/3/3，stride2/2/2，padding2/1/1，ELU | 第三层/步幅/padding自行设计；Extreme Parkour为两卷积+max-pool |
| Transformer | 1层、4heads、FFN256、GELU、dropout0、post-norm/eps1e-5 | 1层控制容量；dropout0保证PPO重算不引入随机mask；其他值未核实作者配置。LocoTransformer使用2层 |
| 位置编码/feature pooling | 可学习17×128位置编码、4×4 average pool、全token拼接 | 17=16视觉+1本体；聚合/位置编码细节由本实现确定 |
| GRU | GRUCell、hidden128，input2176 | PIE只给GRU结构未给宽度；参数量约885504，与EP input32/hidden512约838656同量级，不证明性能等效 |
| 后继/地图decoder | 两层128/128 | 非线性重构预算，PIE未给层宽；不同于DreamWaQ64/128后继decoder |
| 地图表示/critic尺寸 | latent32，height scan17×11=187，critic235 | scan网格继承Gym的常见测点，map latent参考EP；PIE只规定height map，不规定这些维数 |
| 深度输入 | 80×60，near0.1m/far3m，归一化为[-.5,.5] | 工程图像与距离范围；EP resized87×58、near0/far2，LocoTransformer为64×64 |
| 相机名义位姿 | torso相对(0.25,0,0.06)m、向下pitch30°；名义HFOV87° | 原论文未给名义安装位姿；HFOV87取86..88中点，也与EP87相同。EP位置(0.27,0,0.03)、angle[-5,5] |
| 图像延迟 | 固定100ms（5个控制tick） | 作者强调时延但无精确值，当前自行选择；不与15ms执行器delay混淆 |
| VAE进入策略的方式 | actor用posterior均值；decoder采样；logvar[-10,5] | 保证PPO概率重算一致；PIE未披露采样重放/裁剪细节 |
| 联合梯度与总loss | 一个Adam；PPO梯度经过估计器；estimation_weight=1 | PIE明确并行学习但未给优化器/梯度边界；总loss跨模块尺度未经收敛调优 |
| Lite3机器人配置 | PD30/.8、stand[0,-.8,1.6]×4、base0.30m、foot radius.022m | Lite3适配，不照搬Go220/.5；PD/站姿/机体出生高度无PIE明确数值 |
| 跑道与课程离散 | 长8m/宽3m，10级；Gym5类×2变体10列/步长.05m/spacing2m；mjlab5类5列 | PIE给障碍上限与渐进原则，未给此完整布局。Gym rough字段宽8m且原任务实际平地网格，不能直接等同 |
| 出生位姿 | Gym固定起点x、y±.1；mjlabx/y/yaw±.1 | 前向跑道适配；不同于原native reset幅度。没有作者披露的精确范围 |
| 课程升降规则/终止 | 成功升、早期失败降；base接触/倾斜/坑底/越界/完成/非有限状态 | 本任务规则与阈值自行实现，未恢复为Go2平地判定 |
| 标签单位/裁剪 | 高程图=(base_z-terrain_z-.5)clip[-1,1]；足高clip[0,2] | 兼顾label幅度的工程约定；PIE未给这些单位处理 |
| 执行器实现差异 | Gym完整torque乘motor factor且target做URDF限幅；mjlab随机effort cap且不加target clip | Table II范围相同不等于动作执行机制完全相同 |
| 数值保护 | Gymstd≥dtype eps；非有限reward/标签处理 | 本实现保护，不是论文参数；不能取消NaN隔离来模拟未知作者实现 |
| 平台/API版本 | Gym Torch2.4.1/Warp1.6.2；mjlab1.6/MuJoCoWarp3.11/RSL5.4.2 | 相机/API兼容选择，本次不更换专用环境，不装主环境 |

## 7. 审查、验证与尚未验证的范围

按环境/传感器初始化→采集→GRU replay→GAE→策略KL/adaptive LR→联合梯度→统计与checkpoint→resume顺序检查。Gym CPU 41项通过；mjlab CPU20项通过。内容涵盖特权信息隔离、true terminal与timeout、继续回合GRU、Gaussian KL/调度、原子保存/Adam恢复、日志、噪声观测缩放、未到期actor不被push改动、critic裁剪不污染标签、mjlab归一化与15k总目标。CPU检查使用张量/边界mock，不作为GPU物理正确性证据。

未进行4096×24 rollout的显存/吞吐/收敛验证。此前Gym broad-phase foundLostAggregatePairsCapacity警告仍属于未解决的已知问题；恢复PhysX原值不等于该警告已经修复，本次没有改容量或跑道布局来假称解决。没有验证论文成功率或真机效果。

最小检查脚本和headless bounded play仍显式使用少量环境/有限步数，它们是诊断工具，不限制正式train默认。Gym保留原RSL控制台日志、10项真实奖励、所有PIE辅助loss与terrain等级；mjlab保留JSON终端指标并补齐TensorBoard真实奖励分项与PIE loss。两边区分VAE kl与policy_kl。mjlab视频/多GPU/导出/交互viewer与非TensorBoard后端仍未实现，服务器mjlab依赖仍按用户先前要求暂停下载。
