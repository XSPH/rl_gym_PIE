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


def quat_rotate_inverse(quat, vector):
    # Isaac Gym xyzw rotation, implemented only so native state-refresh code can
    # run on CPU. No task behavior (rewards, resets, sensors) is substituted.
    xyz = quat[:, :3]
    return (vector * (2 * quat[:, 3].square() - 1)[:, None]
            - 2 * quat[:, 3, None] * torch.cross(xyz, vector, dim=-1)
            + 2 * xyz * (xyz * vector).sum(-1, keepdim=True))


def load_native_classes():
    """Return task/config classes while keeping fake imports local to this call."""
    gymapi = module("isaacgym.gymapi", UP_AXIS_Z=2, DOMAIN_SIM=0,
                    Vec3=lambda *args: SimpleNamespace(x=args[0], y=args[1], z=args[2]))
    gymtorch = module("isaacgym.gymtorch", wrap_tensor=lambda tensor: tensor,
                      unwrap_tensor=lambda tensor: tensor)
    utils = module("isaacgym.torch_utils", quat_rotate_inverse=quat_rotate_inverse,
                   quat_apply=lambda quat, vector: quat_rotate_inverse(
                       torch.cat((-quat[:, :3], quat[:, 3:]), -1), vector),
                   torch_rand_float=lambda low, high, shape, device:
                       low + (high-low) * torch.rand(*shape, device=device),
                   to_torch=lambda value, **kwargs: torch.as_tensor(value, **kwargs),
                   get_axis_params=lambda value, axis: [value if index == axis else 0
                                                       for index in range(3)])
    gymutil = module("isaacgym.gymutil")
    class SubTerrain:
        def __init__(self, name, width, length, vertical_scale, horizontal_scale):
            self.width, self.length = width, length
            self.vertical_scale, self.horizontal_scale = vertical_scale, horizontal_scale
            self.height_field_raw = np.zeros((width, length), dtype=np.int16)
    terrain_utils = module("isaacgym.terrain_utils", SubTerrain=SubTerrain)
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
        "legged_gym.utils.math": module("legged_gym.utils.math",
             wrap_to_pi=lambda angle: (angle + torch.pi) % (2 * torch.pi) - torch.pi),
        "legged_gym.utils.isaacgym_utils": module("legged_gym.utils.isaacgym_utils",
             get_euler_xyz=lambda quat: torch.zeros(quat.shape[0], 3)),
    }
    loaded = {}
    # The module cache is restored when the context exits. Keeping returned
    # class objects is safe: their global namespace holds the imported helpers.
    with patch.dict(sys.modules, imported):
        for name, path in (
                ("legged_gym.envs.base.base_config", "legged_gym/envs/base/base_config.py"),
                ("legged_gym.envs.base.legged_robot_config", "legged_gym/envs/base/legged_robot_config.py"),
                ("legged_gym.envs.base.base_task", "legged_gym/envs/base/base_task.py"),
                ("legged_gym.utils.helpers", "legged_gym/utils/helpers.py"),
                ("legged_gym.envs.base.legged_robot", "legged_gym/envs/base/legged_robot.py"),
                ("legged_gym.envs.pie.lite3_config", "legged_gym/envs/pie/lite3_config.py"),
                ("legged_gym.envs.pie.lite3", "legged_gym/envs/pie/lite3.py"),
                ("legged_gym.utils.task_registry", "legged_gym/utils/task_registry.py")):
            spec = importlib.util.spec_from_file_location(name, ROOT / path)
            obj = importlib.util.module_from_spec(spec)
            sys.modules[name] = obj
            spec.loader.exec_module(obj)
            loaded[name] = obj
        terrain_module = sys.modules["legged_gym.pie.terrain"]
    return SimpleNamespace(
        base=loaded["legged_gym.envs.base.legged_robot"].LeggedRobot,
        task=loaded["legged_gym.envs.pie.lite3"].Lite3PIE,
        terrain=terrain_module.PIETerrain,
        sampler=terrain_module.TerrainSampler,
        helpers=loaded["legged_gym.utils.helpers"],
        registry=loaded["legged_gym.utils.task_registry"].TaskRegistry,
        config=loaded["legged_gym.envs.pie.lite3_config"].Lite3PIECfg,
        train_config=loaded["legged_gym.envs.pie.lite3_config"].Lite3PIECfgPPO)
