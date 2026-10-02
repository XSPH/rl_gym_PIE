"""Two tiny GPU integration checks; no policy training.

Run with the dedicated environment: python -s legged_gym/scripts/check_pie_backend.py
"""
from isaacgym import gymapi  # Native bindings must precede torch.

import json
import math

import numpy as np
import torch

from legged_gym.envs.pie.lite3 import Lite3PIE
from legged_gym.envs.pie.lite3_config import Lite3PIECfg
from legged_gym.pie.config import CameraConfig
from legged_gym.pie.terrain import TerrainAtlas
from legged_gym.pie.warp_camera import WarpDepthCamera


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU checks require CUDA; no checks were run")
    vertices = np.array([[-20, -20, 0], [20, -20, 0], [20, 20, 0], [-20, 20, 0]], dtype=np.float32)
    atlas = TerrainAtlas(vertices, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
                         None, None, None, [], None)
    camera = WarpDepthCamera(atlas, 2, CameraConfig(height=21, width=31, far=4.0, normalize=False), "cuda:0")
    camera.positions[:, 2] = 1.0
    camera.orientations[:, 1] = math.sin(math.pi / 12)
    camera.orientations[:, 3] = math.cos(math.pi / 12)
    depth = camera.render()
    optical_depth = depth[0, 10, 15].item()
    assert abs(optical_depth - 2.0) < 1e-4
    assert torch.equal(depth[0], depth[1]) and torch.isfinite(depth).all()
    cfg = Lite3PIECfg()
    cfg.seed = 7
    cfg.env.num_envs = 2
    cfg.env.episode_length_s = 0.04
    cfg.pie.terrain.kinds = ["flat"]
    cfg.pie.terrain.levels = cfg.pie.terrain.variants = 1
    cfg.pie.randomization.enabled = False
    cfg.pie.observation_noise = False
    cfg.domain_rand.randomize_friction = cfg.domain_rand.randomize_base_mass = False
    sim = gymapi.SimParams()
    sim.dt = cfg.pie.physics_dt
    env = Lite3PIE(cfg, sim, gymapi.SIM_PHYSX, "cuda:0", True)
    try:
        obs = env.reset()
        assert tuple(obs["critic"].shape) == (2, 235)
        assert tuple(obs["depth"].shape) == (2, 2, 60, 80)
        actions = torch.full((2, 12), 0.1, device=env.device)
        env.step(actions)
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert truncated.all() and not terminated.any()
        assert torch.allclose(info["terminal_proprio"][:, -12:], actions)
        assert torch.equal(info["terminal_proprio"], info["terminal_observation"]["proprio"])
        assert torch.equal(obs["proprio"][:, -12:], torch.zeros_like(actions))
        assert torch.equal(obs["proprio_history"][:, 0], obs["proprio_history"][:, -1])
        assert torch.isfinite(rewards).all()
    finally:
        env.close()
    print(json.dumps({"checks": 2, "passed": True, "optical_depth_m": optical_depth,
                      "expected_depth_m": 2.0, "pre_reset_labels": True, "history_reset": True}))


if __name__ == "__main__":
    main()
