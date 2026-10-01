# Isaac Gym PIE：迁移至安装文档指定的 rsl_rl

依据：`unitree_rl_gym/doc/setup_zh.md` 的 2.3 节和英文版对应章节，均要求 `git checkout v1.0.2`。
本地官方克隆：`unitree_rl_gym/rsl_rl`。
基线：`v1.0.2` / `2ad79cf0caa85b91721abfe358105f869a784121`。
源码为独立嵌套 Git 仓库，保留 origin；没有提交或推送。

实现路径：

| 扩展 | 使用的 v1.0.2 基础实现 |
| --- | --- |
| `modules/actor_critic_pie.py` | 继承 ActorCritic，直接使用原生 actor/critic/std，增加视觉、本体、Transformer、GRU、估计头和辅助解码器 |
| `algorithms/ppo_pie.py` | 继承 PPO，复用原生 Adam 和 PPO 配置字段，扩展完整轨迹重放及联合估计损失 |
| `storage/rollout_storage_pie.py` | 继承 RolloutStorage，通过 add_transitions 存原生 PPO 字段，补充深度历史、标签和 timeout 前值 |
| `runners/on_policy_runner_pie.py` | 继承 OnPolicyRunner，调用父构造器的原生模型/算法工厂及存储初始化，扩展字典观测、循环状态、日志和 checkpoint |

原 runner 新增 reset 钩子，PIE 覆盖字典 reset；非 PIE reset 仍保持原双返回值逻辑。原包导出新类。宇树 task_registry 直接选择本地 rsl_rl 的 PIE runner；原 `legged_gym/pie/models.py` 和 `learner.py` 仅保留兼容导入。

保留此前网络尺寸、历史长度、PPO 超参数和联合损失设置；此次没有更改环境奖励或物理随机化。版本差异中实际改变的是动作噪声优化参数：由 2.2.4 的 log_std 改为 v1.0.2 的 std，初始标准差仍为 0.5。旧 checkpoint 权重提供键名及 log_std→std 转换，未验证加载；不支持旧优化器恢复。网络的工程默认值仍不能称为论文作者未公开参数的精确复现。

依赖仅调整新环境 `.conda-envs/pie-isaacgym`：卸载 `rsl-rl-lib==2.2.4`，editable 安装本地 `rsl_rl==1.0.2` 并重新安装父项目。安装元数据确认路径指向上述克隆，pip check 无损坏依赖。主环境、原环境及 mjlab 环境未修改。

静态审查：Python 3.8 AST 解析 Gym 项目、测试源码及 rsl_rl 共 56 个 Python 文件；父仓库与 rsl_rl 仓库的 git diff --check 通过；核查导入关系、工厂参数、原生存储写入、GRU 序列重放、timeout bootstrap 和 actor 特权信息隔离。
未执行任何模型、测试、仿真、训练或 checkpoint 回放。此前验证日志只对应旧算法实现。
