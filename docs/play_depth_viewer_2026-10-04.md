# 回放深度窗口

日期：2026-10-04；工作树 `rl_gym_PIE_native`；分支 `refactor/pie-native-training`。

## 改动

- `play.py` 在 `lite3_pie` 回放中支持 `--show_depth` 和 `--depth_env`。
- `utils/depth_viewer.py` 使用已有 Matplotlib/Tk，按相机捕获频率上限刷新，
  仅复制所选机器人的图像到 CPU 显示。
- 并排显示 `camera.depth` 最新采集帧及 `depth_history` 的两帧。
  显示在策略推理前更新；归一化历史帧按原公式逆变换为米。
- 左右方向键或 `P/N` 切换机器人；`Esc/Q` 和窗口关闭按钮关闭深度窗口，
  回放继续。脚本退出或环境初始化失败时清理窗口及环境。
- Tk/Matplotlib 仅在启用显示时导入，不额外调用 `camera.render/encode`，
  不改动相机采集时序、延迟、策略输入、训练及模型格式。
- `--headless --show_depth` 仍需要桌面显示，但仅打开深度窗口。

## 本地运行

使用已下载的第 7000 轮模型，在专用环境中执行：

```bash
python -s legged_gym/scripts/play.py --task=lite3_pie \
  --checkpoint_file logs/from_4090/Oct04_00-37-49_/model_7000.pt \
  --num_envs 5 --steps 5000 --show_depth --depth_env 0
```

当前深度源为静态地形，机器人自身及其他动态物体不参与深度遮挡。

## 检查范围

本次语法及差异检查通过；未启动本地仿真，也未运行测试套件。
