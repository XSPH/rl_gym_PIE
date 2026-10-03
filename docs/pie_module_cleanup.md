# PIE 模块收拢记录

日期：2026-10-03；工作树 `rl_gym_PIE_native`；分支 `refactor/pie-native-training`。
本次起点为 `09c2122`。仅修改此 Gym 工作树，使用专用环境 `pie-isaacgym-native`。

## 第一阶段：配置唯一来源

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

## 验证边界

仅静态/CPU 检查；CPU 替身替代 Gym 原生接口，不创建仿真器。
GPU 仿真、4096 环境性能与显存、服务器同步不在本次范围。
