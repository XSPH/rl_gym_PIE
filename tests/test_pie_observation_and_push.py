"""CPU contracts for sensor corruption, auxiliary labels and indexed impulses.

Native tensor setters are recorded; this does not validate PhysX execution.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from legged_gym.pie.config import EnvConfig


@pytest.fixture
def env():
    tensors = [torch.zeros(3, 13), torch.zeros(3, 12, 2),
               torch.zeros(3, 1, 13), torch.zeros(3, 1, 3)]
    setters = []
    gym = SimpleNamespace(
        acquire_actor_root_state_tensor=lambda sim: tensors[0],
        acquire_dof_state_tensor=lambda sim: tensors[1],
        acquire_rigid_body_state_tensor=lambda sim: tensors[2],
        acquire_net_contact_force_tensor=lambda sim: tensors[3],
        refresh_actor_root_state_tensor=lambda sim: None,
        refresh_dof_state_tensor=lambda sim: None,
        refresh_rigid_body_state_tensor=lambda sim: None,
        refresh_net_contact_force_tensor=lambda sim: None,
        set_actor_root_state_tensor_indexed=lambda sim, root, ids, count:
            setters.append((root.clone(), ids.clone(), count)))
    bindings = SimpleNamespace(gymapi=SimpleNamespace(), gymtorch=SimpleNamespace(
        wrap_tensor=lambda tensor: tensor, unwrap_tensor=lambda tensor: tensor))
    path = Path(__file__).parents[1] / "legged_gym/pie/sensors_and_rollout.py"
    spec = importlib.util.spec_from_file_location("legged_gym.pie._cpu_sensor_contract", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"isaacgym": bindings}):
        spec.loader.exec_module(module)
    task = module.PIESensorsAndRollout()
    task.num_envs, task.num_bodies, task.device = 3, 1, "cpu"
    task.gym, task.sim, task.setters = gym, None, setters
    task.config = EnvConfig(num_envs=3, device="cpu")
    task.obs_scales = SimpleNamespace(dof_pos=1.0)
    task.cfg = SimpleNamespace(
        noise=SimpleNamespace(add_noise=True, noise_level=1.0,
            noise_scales=SimpleNamespace(ang_vel=0.2, gravity=0.05,
                                        dof_pos=0.01, dof_vel=1.5)),
        normalization=SimpleNamespace(clip_observations=100.0),
        domain_rand=SimpleNamespace(push_robots=True, push_interval=5, max_push_vel_xy=1.0),
        rewards=SimpleNamespace(only_positive_rewards=True))
    task._acquire_buffers()
    task.root[:, 6] = 1
    task.dof[:, :, 0] = task.stand
    task.joint_order = torch.arange(12)
    task.commands_scale = torch.tensor([2.0, 2.0, 0.25])
    task.actor_indices = torch.arange(3, dtype=torch.int32)
    return task


def test_proprio_uses_native_scaling_noise_order_and_observation_clip(env):
    env.commands[:] = torch.tensor([1.0, 0.0, 1.0])
    env.config.observation_noise = False
    clean = env._proprio()
    torch.testing.assert_close(clean[:, 6:9], torch.tensor([[2.0, 0.0, 0.25]]).expand(3, -1))
    env.config.observation_noise = True
    with patch("torch.rand_like", side_effect=lambda tensor: torch.ones_like(tensor)):
        noisy = env._proprio()
    expected = torch.tensor([0.05] * 6 + [0.0] * 3 + [0.01] * 12
                            + [0.075] * 12 + [0.0] * 12)
    torch.testing.assert_close(noisy - clean, expected.expand(3, -1))
    env.dof[:, :, 1] = 4000
    assert env._proprio()[:, 21:33].max() == 100


def test_push_changes_only_due_actor_rows_and_keeps_other_velocity_components(env):
    env.episode_steps[:] = torch.tensor([1, 5, 10])
    env.root[:, 7:13] = torch.arange(18).reshape(3, 6)
    before = env.root.clone()
    env._apply_pushes()
    assert len(env.setters) == 1
    root, ids, count = env.setters[0]
    assert ids.dtype == torch.int32 and ids.tolist() == [1, 2] and count == 2
    torch.testing.assert_close(root[0], before[0])
    torch.testing.assert_close(root[:, 9:13], before[:, 9:13])
    assert (root[1:, 7:9].abs() <= 1.0).all()
    env.config.randomization.enabled = False
    env._apply_pushes()
    assert len(env.setters) == 1


def test_critic_clip_does_not_change_raw_supervision_targets(env):
    env.foot_indices = torch.zeros(4, dtype=torch.long)
    env.levels = env.columns = torch.zeros(3, dtype=torch.long)
    env.terrain = SimpleNamespace(sample=lambda points, levels, columns:
                                 torch.zeros(points.shape[:-1]))
    env.root[:, 7] = 1000
    observations = env._observations(torch.zeros(3, 45))
    assert observations["critic"][:, 45].max() == 100
    assert observations["targets"]["velocity"][:, 0].min() == 1000


def test_native_nonnegative_total_reward_preserves_negative_term_statistics(env):
    env.collision_indices = torch.tensor([0])
    env.contacts[:, 0, 2] = 2.0
    zeros = torch.zeros(3, 12)
    reward, terms = env._reward(zeros, zeros, zeros, zeros)
    assert (reward == 0).all()
    assert (terms["collision"] == -10).all()
    env.cfg.rewards.only_positive_rewards = False
    unclipped, same_terms = env._reward(zeros, zeros, zeros, zeros)
    torch.testing.assert_close(unclipped, torch.full((3,), -8 * env.config.policy_dt))
    torch.testing.assert_close(same_terms["collision"], terms["collision"])
