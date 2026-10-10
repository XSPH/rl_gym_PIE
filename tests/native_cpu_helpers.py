"""Load real task methods with narrow CPU substitutes for native Gym bindings.

These substitutes implement imports and tensor setters, never a simulator. Tests
exercise the actual task/native methods rather than reimplementing their logic.
"""
import importlib.util
from dataclasses import asdict, is_dataclass
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import rsl_rl.runners  # Initialize TensorBoard before the temporary module cache patch.


ROOT = Path(__file__).parents[1]

# The SDK's torch_utils.py contains pure Torch functions and imports no native
# Gym bindings. Load those actual quaternion operations while keeping all
# simulator interfaces below as CPU substitutes.
_sdk = importlib.util.find_spec('isaacgym')
_utils_spec = importlib.util.spec_from_file_location(
    'pie_cpu_isaacgym_torch_utils', Path(_sdk.origin).parent / 'torch_utils.py')
_torch_utils = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_torch_utils)


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


def package(name, relative):
    result = module(name)
    result.__path__ = [str(ROOT / relative)]
    return result


def class_to_dict(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: class_to_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [class_to_dict(item) for item in value]
    if not hasattr(value, "__dict__"):
        return value
    return {key: class_to_dict(getattr(value, key)) for key in dir(value)
            if not key.startswith("_") and not callable(getattr(value, key))}


def original_visual_config(classes):
    """Explicit pre-experiment settings for original visual/terrain regressions."""
    cfg = classes.config()
    cfg.camera.input_mode = 'depth'
    cfg.domain_rand.randomize_camera = True
    cfg.control.stiffness, cfg.control.damping = {'joint': 30.0}, {'joint': 0.8}
    cfg.terrain.curriculum = True
    cfg.terrain.kinds = ['flat', 'gap', 'step', 'hurdle', 'stairs']
    cfg.terrain.terrain_proportions = [0.2] * 5
    cfg.terrain.max_init_terrain_level = 5
    return cfg


def load_native_classes():
    """Return task/config classes while keeping fake imports local to this call."""
    gymapi = module("isaacgym.gymapi", UP_AXIS_Z=2, DOMAIN_SIM=0,
                    Vec3=lambda *args: SimpleNamespace(x=args[0], y=args[1], z=args[2]))
    gymtorch = module("isaacgym.gymtorch", wrap_tensor=lambda tensor: tensor,
                      unwrap_tensor=lambda tensor: tensor)
    utils = _torch_utils
    gymutil = module("isaacgym.gymutil")
    terrain_utils = module("isaacgym.terrain_utils")
    gym = module("isaacgym", gymapi=gymapi, gymtorch=gymtorch, gymutil=gymutil,
                 torch_utils=utils, terrain_utils=terrain_utils)
    imported = {
        "isaacgym": gym, "isaacgym.gymapi": gymapi,
        "isaacgym.gymtorch": gymtorch, "isaacgym.gymutil": gymutil,
        "isaacgym.torch_utils": utils,
        "isaacgym.terrain_utils": terrain_utils,
        "legged_gym.envs": package("legged_gym.envs", "legged_gym/envs"),
        "legged_gym.envs.base": package("legged_gym.envs.base", "legged_gym/envs/base"),
        "legged_gym.envs.pie": package("legged_gym.envs.pie", "legged_gym/envs/pie"),
        "legged_gym.utils": package("legged_gym.utils", "legged_gym/utils"),
        "legged_gym.utils.helpers": module("legged_gym.utils.helpers", class_to_dict=class_to_dict),
        "legged_gym.utils.isaacgym_utils": module("legged_gym.utils.isaacgym_utils",
             get_euler_xyz=lambda quat: torch.zeros(quat.shape[0], 3)),
    }
    loaded = {}
    # The module cache is restored when the context exits. Keeping returned
    # class objects is safe: their global namespace holds the imported helpers.
    with patch.dict(sys.modules, imported):
        # The terrain builders/converter are pure NumPy/SciPy SDK code. Load
        # the actual module against the temporary Gym imports, not a geometry
        # substitute, now that PIE uses WMP's native generators.
        terrain_spec = importlib.util.spec_from_file_location(
            'isaacgym.terrain_utils', Path(_sdk.origin).parent / 'terrain_utils.py')
        terrain_spec.loader.exec_module(terrain_utils)
        for name, path in (
                ("legged_gym.envs.base.base_config", "legged_gym/envs/base/base_config.py"),
                ("legged_gym.envs.base.legged_robot_config", "legged_gym/envs/base/legged_robot_config.py"),
                ("legged_gym.envs.base.base_task", "legged_gym/envs/base/base_task.py"),
                ("legged_gym.utils.helpers", "legged_gym/utils/helpers.py"),
                ("legged_gym.utils.math", "legged_gym/utils/math.py"),
                ("legged_gym.utils.kinematics", "legged_gym/utils/kinematics.py"),
                ("legged_gym.utils.terrain", "legged_gym/utils/terrain.py"),
                ("legged_gym.envs.base.legged_robot", "legged_gym/envs/base/legged_robot.py"),
                ("legged_gym.envs.pie.lite3_config", "legged_gym/envs/pie/lite3_config.py"),
                ("legged_gym.envs.pie.lite3", "legged_gym/envs/pie/lite3.py"),
                ("legged_gym.utils.task_registry", "legged_gym/utils/task_registry.py")):
            spec = importlib.util.spec_from_file_location(name, ROOT / path)
            obj = importlib.util.module_from_spec(spec)
            sys.modules[name] = obj
            spec.loader.exec_module(obj)
            loaded[name] = obj
        terrain_module = loaded["legged_gym.utils.terrain"]
    return SimpleNamespace(
        base=loaded["legged_gym.envs.base.legged_robot"].LeggedRobot,
        task=loaded["legged_gym.envs.pie.lite3"].Lite3PIE,
        terrain=terrain_module.PIETerrain,
        sampler=terrain_module.TerrainSampler,
        kinematics=loaded["legged_gym.utils.kinematics"],
        math=loaded["legged_gym.utils.math"],
        torch_utils=utils,
        helpers=loaded["legged_gym.utils.helpers"],
        registry=loaded["legged_gym.utils.task_registry"].TaskRegistry,
        config=loaded["legged_gym.envs.pie.lite3_config"].Lite3PIECfg,
        train_config=loaded["legged_gym.envs.pie.lite3_config"].Lite3PIECfgPPO)
