# rl_gym_PIE 发布范围

GitHub：<https://github.com/XSPH/rl_gym_PIE>。

发布内容是当前宇树项目的源码快照，保留上游许可证、原任务源码和机器人模型资产。上游基线为 `unitreerobotics/unitree_rl_gym@276801e46c5d433564f24658bac64f254b7d2d4b`，此前本地 Git 历史仍保留在原工作区。

`rsl_rl/` 提供完整的 v1.0.2 基线和 PIE 扩展，包含 `setup.py`、许可证、modules、algorithms、storage、runners；它是普通源码目录，克隆本仓库即可获得，不需要额外的私有仓库或 submodule。安装仍使用 `python -s -m pip install -e ./rsl_rl`。Isaac Gym SDK 需单独安装，未包含在源码中。

`.gitignore` 排除 Conda/venv、缓存、日志、checkpoint、导出策略、数据集、压缩包和预编译库。本次发布不携带上游演示权重 `deploy/pre_train/*/motion.pt`；原部署示例若需这些权重，应从宇树上游获取，新增 PIE 网络也不适用原部署脚本。URDF、XML、mesh 和源代码完整保留。

`PUBLICATION_MANIFEST.json` 记录此前发布快照的上传文件；本地重构分支已改变源码，当前变更与检查见 [模块收拢记录](docs/pie_module_cleanup.md)。历史清单列出上传文件的字节数、权限模式和 SHA-256，以及有意排除的文件。单文件大小门限为 25 MiB；超过门限会中止准备，不能静默漏传。发布后重新克隆远端，逐文件检查清单、内容哈希、权限及 Python 源码语法，同时核对 GitHub `main` 的 commit。

本次仅检查发布完整性，没有运行强化学习测试、仿真、训练或回放。`docs/validation*` 保存的是此前历史记录，相关日志与 checkpoint 已排除。仓库里的原工作区命令含本地路径；在新机器上需要按实际克隆目录和独立环境路径调整。
