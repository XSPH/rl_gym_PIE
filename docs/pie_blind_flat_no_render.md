# 零视觉训练跳过相机渲染

日期：2026-10-08；分支 `experiment/pie-blind-flat`；起点 `d7b2dd1`。
用户确认零输入实验可以省掉相机渲染。此前先渲染/编码再归零，仍承担相机开销。

## 当前实现

- `camera.input_mode='zero'` 且 `render_for_debug=False` 时，相机对象为 `None`，
  不导入/初始化 Warp，不建立相机 mesh、姿态或真实深度缓冲。
- `_init_pie_buffers()` 分配一张全零图像并扩展为原环境数量视图，逻辑捕获时复用，
  不重复分配与编码。历史、固定延迟、周期更新、局部/全体重置及帧编号沿用原流程。
- `depth` 模式继续构造、渲染和编码真实相机。
- play 将 `--show_depth` 传给配置恢复函数，明确设置 `render_for_debug`。
  零输入模型打开诊断窗口时才构造真实相机，显示原始米制深度，策略输入仍然为零。
- CNN、Transformer、GRU、全部预测头、PPO 和辅助损失没有修改。
  本次优化减少相机初始化和采样阶段开销；CNN、帧池及更新计算仍保留。

修改文件为 `lite3.py`、`lite3_config.py`、`helpers.py`、`play.py` 和相应 CPU 检查与文档。
新增字段只是诊断渲染开关，默认关闭。checkpoint 仍为 version 4，以 `input_mode`
检查视觉兼容性；已有 `d7b2dd1` 的 `zero` 模型可以续训，不要求新增诊断字段。
play 不继承模型中保存的诊断渲染状态，按本次 CLI 是否带 `--show_depth` 决定。

## 验证

本地完整 CPU 回归：**147 passed，32 条已有 PyTorch autocast 警告，29.84 秒**。
结果和当前有效配置见 [验证数据](validation/pie_blind_flat_no_render_cpu_2026-10-08.json)。

检查包括：零输入训练拒绝任何 Warp 导入；visual/debug 四种组合的相机初始化选择；
诊断相机渲染真实图像时不调用编码；无相机的周期更新、局部及全体重置严格全零；
渲染不同图像和完全不渲染时，固定本体观测/GRU 状态的策略动作和下一状态逐值一致；
checkpoint 的视觉模式、诊断开关恢复；原有视觉、梯度、缓存及 PPO 回归继续通过。

没有启动仿真或 GPU 训练，也没有测量提速比例。是否明显变快应比较重新启动后的
采样耗时与总单轮耗时，PPO 更新耗时可能保持接近。

## 在 4090 使用

服务器仓库为 `/home/asuka/rl_gym_PIE_native`，环境为 `pie-isaacgym-native`。
代码更新不会替换已经启动的 Python 进程中的类；运行中的旧训练保持旧行为。
本次不终止或重启已有训练。下次启动使用原命令即可跳过相机渲染：

```bash
source /home/asuka/miniconda3/etc/profile.d/conda.sh
conda activate pie-isaacgym-native
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd /home/asuka/rl_gym_PIE_native
python -s legged_gym/scripts/train.py --task=lite3_pie --headless
```

保留已经学到的权重时，在停止旧训练后给出该实验已保存的 checkpoint：

```bash
python -s legged_gym/scripts/train.py --task=lite3_pie --headless \
  --resume --checkpoint_file /absolute/path/model_500.pt
```
