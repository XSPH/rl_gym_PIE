# mjlab 视觉强化学习组件调研

调研日期：2026-09-30。**mjlab 原生支持 GPU RGB 和深度图。** `CameraSensorCfg` 可关联 MJCF 相机，也可创建相机；渲染使用 MuJoCo Warp 的射线管线，直接提供批量 GPU 图像。深度是到相机平面的轴向深度，不能直接当作射线距离。[原生 RGB-D 文档](https://mujocolab.github.io/mjlab/main/source/sensors/rgbd_camera.html)。

| 开源项目 | 感知组件 | 策略 / 学习组件 | 与本任务的关系 |
| --- | --- | --- | --- |
| [mjlab 官方 Yam 视觉抓取](https://github.com/mujocolab/mjlab/tree/v1.6.0/src/mjlab/tasks/manipulation/config/yam) | 原生 `CameraSensorCfg`，RGB/Depth 进入策略观测 | `SpatialSoftmaxCNNModel`、rsl_rl PPO、`MjlabOnPolicyRunner` | 原生视觉 RL 训练示例，机器人为机械臂 |
| [InstinctMJ](https://github.com/project-instinct/InstinctMJ) | G1 parkour 的 `NoisyGroupedRayCasterCameraCfg`，pinhole 射线深度、图像历史及延迟 | 视觉编码器和 `instinct_rl` 训练配置 | 腿足视觉任务参考；相机是自定义扩展 |
| [mjdrone](https://github.com/oorischubert/mjdrone) | mjlab 相机 RGB + IMU/state | ResNet18、cross-attention 融合与 PPO | 多模态融合参考，任务为无人机 |

官方 Yam 明确把 `camera` 纳入 actor/critic 观测组，并给出 CNN 与 PPO 配置，属于策略使用视觉的训练任务。[视觉 runner 源码](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/manipulation/config/yam/rl_cfg.py)，[环境源码](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/manipulation/config/yam/env_cfgs.py)。

InstinctMJ 的 G1 parkour 配置使用 64×36 射线相机、`distance_to_image_plane`、延迟观测及自定义噪声链。本项目参考传感器组织方式，不复制噪声设置，以遵循用户提供的 PIE 作者说明。[G1 感知配置](https://github.com/project-instinct/InstinctMJ/blob/main/src/instinct_mj/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py)。

mjdrone 的项目说明给出 RGB 输入、预训练 ResNet18、视觉 token 与状态 cross-attention，并给出 PPO 训练入口。任务类型与四足跑酷不同，不能作为 PIE 成功率证据。[项目与组件说明](https://github.com/oorischubert/mjdrone)。

本次 `unitree_rl_mjlab` 固定使用 mjlab **1.6.0** 原生 `CameraSensorCfg(data_types=("depth",))`。相机 10 Hz、策略 50 Hz，两帧深度历史；无图像噪声或滤波。额外实现相机姿态/FOV 随机化、图像延迟队列、同步高程图/足高标签和 reset 清理。

非相机帧只更新 BVH 和地形监督射线，避免每个控制步渲染。目前这部分访问固定版本 `Simulation._sensor_context` 内部接口，升级 mjlab 需重新核查。actor 只读取本体和深度，真实速度/高程图进入 critic 和辅助监督。

原宇树任务未提供 PIE 联合估计器训练，因此新增 `PIEOnPolicyRunner`。本次未把 ONNX 回放或 viewer 渲染视作视觉 RL 训练，未将参考项目称为 PIE 官方实现。
