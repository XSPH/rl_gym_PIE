# PIE 网络超参数决策

日期：2026-10-01。两平台采用相同网络尺寸，各自在项目内保存实现与配置。本文确定复现基线，不声称恢复了 PIE 作者未公开的配置，也不声称经过收敛或速度验证。

## 核实到的依据

| 来源 | 可以直接核实的内容 | 本实现的使用方式 |
| --- | --- | --- |
| [PIE §III-B](https://arxiv.org/html/2408.13740v3#S3.SS2) | 本体历史10帧、深度2帧；CNN/MLP跨模态Transformer，输出拼接后进GRU；速度、足高、地图latent、VAE latent及两类重构 | 保持结构和观测约束 |
| [DreamWaQ Fig.1/2、§IV-A](https://arxiv.org/pdf/2301.10602) | actor/critic隐藏512/256/128；动力学latent16、速度3；CENet编码128/64、解码64/128；隐藏层ELU | 采用策略层宽、动力学latent维数与ELU；PIE视觉编码器和解码器另行选择 |
| [LocoTransformer §4.2、Appendix B.5](https://arxiv.org/html/2107.03996) | 视觉网格4×4、通道128；token维数128；2层Transformer；本体MLP与projection head的隐藏层256/256 | 采用128维token和4×4空间网格；本复现用1层注意力，并按PIE接GRU |
| [Extreme Parkour 官方 depth_backbone.py](https://github.com/chengxuxin/extreme-parkour/blob/main/rsl_rl/rsl_rl/modules/depth_backbone.py) | CNN通道32/64、卷积核5/3，中间max-pool；视觉压缩经128维FC到32维；GRU输入32、隐藏512 | 参考通道规模和紧凑地图表示；步幅、第三层CNN与GRU宽度自行确定 |
| [Extreme Parkour 官方配置](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/envs/base/legged_robot_config.py) | scan encoder输出32维，actor/critic为512/256/128，ELU；深度87×58 | 交叉验证策略层宽与地图latent32的起点 |

LocoTransformer正文另有“hidden feature dimension 256”的描述；附录明确token为128，不能把256误写成token维数。本文的4个注意力头、FFN=256是复现选择，不宣称这些数字已从该论文的官方实现核实。Extreme Parkour公开代码按检索当天的main分支阅读；它的多阶段训练目标也不同于PIE，因此不能整套照搬。以上均来自论文或作者仓库，第三方DreamWaQ复刻没有被当作作者配置。

## 最终基线与理由

| 模块 | 确定值 | 理由与取舍 |
| --- | --- | --- |
| 输入 | 本体10×45；深度2×60×80 | 历史帧数遵循PIE；80×60是工程分辨率，与参考视觉项目的像素量同量级。更高分辨率是否改善细窄边缘需实验确定 |
| 本体编码 | 450→512→256→128，隐藏层ELU、末层线性 | 450维历史先作非线性压缩，输出与视觉token对齐；第一层比LocoTransformer的256更宽，是面对更长拼接历史的容量选择 |
| 深度编码 | 通道2→32→64→128；kernel=5/3/3，stride=2/2/2，padding=2/1/1；各层ELU | 前两层规模参考Extreme Parkour；三次降采样限制后续特征规模；第三层将视觉通道对齐到128 |
| 空间token | CNN特征8×10→自适应平均池化4×4；16个视觉token+1个本体token；可学习位置编码 | 参考LocoTransformer，保留粗空间布局；平均池化是特征压缩，不是对原始深度图做传感器滤波。细窄边缘可能损失，是需要关注的代价 |
| Transformer | d_model=128，4 heads，每头32；1层；FFN=256，GELU；dropout=0；post-norm，LayerNorm eps=1e-5 | 一层即可让所有token交换信息，GRU继续整合时间信息；选择一层是控制容量的起点，未经消融证明足够。dropout关闭，避免PPO重算概率时额外随机mask造成比值变化 |
| 时间记忆 | 单层GRUCell，input=17×128=2176，hidden=128；按50 Hz策略步更新 | 遵循PIE拼接后进GRU；128维是容量预算选择。隐藏状态跨rollout保留，仅按环境done重置；它的记忆时长不是固定128步 |
| 输出头 | 速度3、四足离地高度4、地图latent32、VAE均值/对数方差各16；均线性输出 | 前两项由物理量定义；z16参考DreamWaQ；地图32参考Extreme Parkour并压缩187个扫描标签，但不保证足够表达所有地形 |
| Actor | 100→512→256→128→12，隐藏ELU、末层线性 | 100=当前本体45+速度3+足高4+地图32+动力学16；层宽由DreamWaQ与Extreme Parkour共同支持 |
| Critic | 235→512→256→128→1，隐藏ELU、末层线性 | 235=本体45+真实速度3+地形187；估值独立使用特权信息 |
| 后继解码器 | 55→128→128→45，隐藏ELU、末层线性 | 55包含全部估计向量；两层128是工程选择，让辅助任务具备非线性重构能力，控制解码器规模 |
| 地图解码器 | 32→128→128→187，隐藏ELU、末层线性 | 只读取地图latent，与PIE信息路径一致；不向actor提供真实高程图 |
| 潜变量/动作随机性 | actor用VAE均值；后继重构用重参数化采样；logvar裁剪[-10,5]；动作初始std=0.5 | 均值供策略重算避免latent重采样改变概率比；裁剪为数值范围选择；std=0.5是保留的探索起点，非论文披露值 |

128维GRU并非简单地把Extreme Parkour网络缩小四倍。采用PyTorch GRU的参数公式
`3H(I+H+2)`（含两组bias），本实现 `I=2176,H=128` 有 **885,504** 个参数；
参考网络 `I=32,H=512` 有 **838,656** 个参数。
如果在本实现的2176维输入上直接改成512隐藏维，将有 **4,131,840** 个GRU参数。
这解释了128作为初始预算的合理性，但参数量相近不等于表达能力相同。

完整默认训练网络按层尺寸解析计算约 **2,095,464** 个参数，含critic、辅助解码器和12个动作噪声参数；不是运行模型测得的结果，也不代表显存占用或推理速度。循环展开时的激活、优化器状态和图像缓存仍需另计。

## 配置与复现边界

网络宽度、CNN核/步幅/padding、token网格、Transformer FFN比例和dropout都已纳入各项目的 `ModelConfig`，由现有checkpoint的 `model_config=asdict(cfg)` 保存。默认层形状保持上述基线，旧checkpoint缺失的新字段补用这些默认值；加载兼容性本次未执行验证。模型配置会检查token/head整除、CNN层数、正维数和零dropout，避免形成不一致的网络。

Isaac Gym在本地rsl_rl v1.0.2的ActorCritic上扩展，使用原生std参数；mjlab沿用独立rsl_rl 5.4.2 MLP实现和log_std参数。两者初始化分布相同，但动作噪声的优化参数化仍不同，这一点应在后续平台对照中记录。

当前PPO设置仍是最小验证配置：学习率3e-4、gamma=0.99、lambda=0.95、clip=0.2、2个epoch和2个minibatch；单次短rollout不是经过确认的正式训练方案。网络配置可作为后续复现起点；本轮未启动训练、测试、仿真或回放，不能据此断言网络已收敛、视觉有效或达到论文成功率。

以后如果获准训练，优先观察地图重构误差、后继误差、KL及各头梯度，再依据证据逐项考虑GRU128→256、Transformer1→2层、地图latent32→64；这些是候选消融，不是当前已应用的改动。
