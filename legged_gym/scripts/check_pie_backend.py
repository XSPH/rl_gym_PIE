"""Two tiny GPU integration checks; no policy training.

Run with the dedicated environment: python -s legged_gym/scripts/check_pie_backend.py
"""
from isaacgym import gymapi  # Native bindings must precede torch.

import json
import math
from types import SimpleNamespace

import numpy as np
import torch

from legged_gym.envs.pie.lite3 import Lite3PIE
from legged_gym.envs.pie.lite3_config import Lite3PIECfg
from legged_gym.utils.warp_camera import WarpDepthCamera


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU checks require CUDA; no checks were run")
    vertices = np.array([[-20, -20, 0], [20, -20, 0], [20, 20, 0], [-20, 20, 0]], dtype=np.float32)
    atlas = SimpleNamespace(vertices=vertices,
                            triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32))
    camera_cfg = Lite3PIECfg().camera
    camera_cfg.height, camera_cfg.width = 21, 31
    camera_cfg.far, camera_cfg.normalize = 4.0, False
    camera = WarpDepthCamera(atlas, 2, camera_cfg, "cuda:0")
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
    cfg.terrain.kinds = ["flat"]
    cfg.terrain.num_rows = cfg.terrain.num_cols = 1
    cfg.terrain.max_init_terrain_level = 0
    cfg.terrain.terrain_proportions = [1.0]
    cfg.terrain.curriculum = False
    cfg.domain_rand.randomize_pie = False
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    sim = gymapi.SimParams()
    sim.dt = cfg.sim.dt
    sim.use_gpu_pipeline = True
    sim.physx.use_gpu = True
    env = Lite3PIE(cfg, sim, gymapi.SIM_PHYSX, "cuda:0", True)
    try:
        env.reset()
        obs = env.get_pie_observations()
        assert tuple(obs["critic"].shape) == (2, 235)
        assert tuple(obs["depth"].shape) == (2, 2, 60, 80)
        actions = torch.full((2, 12), 0.1, device=env.device)
        # Gym stores sim.dt as float32: ceil(0.04 / env.dt) may be 3,
        # rather than 2. Follow the native strict '>' timeout boundary and
        # include the zero-action step already performed by BaseTask.reset().
        assert torch.equal(env.episode_length_buf, env.episode_length_buf[:1].expand(2))
        remaining = int(env.max_episode_length) + 1 - int(env.episode_length_buf[0])
        for step in range(remaining):
            _, _, rewards, dones, info = env.step(actions)
            if step < remaining - 1:
                assert not dones.any(), "Unexpected early termination in the flat-course check"
        assert dones.all() and info["time_outs"].all()
        assert torch.allclose(info["pie"]["terminal_proprio"][:, -12:], actions)
        obs = env.get_pie_observations()
        assert torch.equal(obs["proprio"][:, -12:], torch.zeros_like(actions))
        assert torch.equal(obs["proprio_history"][:, 0], obs["proprio_history"][:, -1])
        assert torch.isfinite(rewards).all()
    finally:
        env.close()
    print(json.dumps({"checks": 2, "passed": True, "optical_depth_m": optical_depth,
                      "expected_depth_m": 2.0, "pre_reset_labels": True, "history_reset": True}))


if __name__ == "__main__":
    main()
