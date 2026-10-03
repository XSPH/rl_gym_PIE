"""Native task lifecycle and reward contracts with recorded CPU tensor setters."""
from types import SimpleNamespace as NS

import pytest
import torch

from native_cpu_helpers import class_to_dict, load_native_classes


@pytest.fixture(scope="module")
def classes():
    return load_native_classes()


def state(task_class, count=3):
    task = task_class.__new__(task_class)
    task.num_envs, task.num_actions, task.device = count, 12, "cpu"
    task.num_obs, task.num_privileged_obs = 45, 235
    task.num_dof = 12
    task.sim, task.viewer = None, None
    task.root_states = torch.zeros(count, 13)
    task.root_states[:, 6] = 1
    task.base_pos = torch.zeros(count, 3)
    task.base_quat = task.root_states[:, 3:7]
    task.rpy = torch.zeros(count, 3)
    task.base_lin_vel = torch.zeros(count, 3)
    task.base_ang_vel = torch.zeros(count, 3)
    task.projected_gravity = torch.zeros(count, 3)
    task.gravity_vec = torch.tensor([0., 0., -1.]).repeat(count, 1)
    task.dof_state = torch.zeros(count, 12, 2)
    task.dof_pos = task.dof_state[..., 0]
    task.dof_vel = task.dof_state[..., 1]
    task.actions = torch.zeros(count, 12)
    task.last_actions = torch.zeros(count, 12)
    task.last_last_actions = torch.zeros(count, 12)
    task.last_dof_vel = torch.zeros(count, 12)
    task.last_root_vel = torch.zeros(count, 6)
    task.torques = torch.zeros(count, 12)
    task.commands = torch.zeros(count, 4)
    task.contact_forces = torch.zeros(count, 2, 3)
    task.termination_contact_indices = torch.tensor([0])
    task.penalised_contact_indices = torch.tensor([0])
    task.obs_buf = torch.zeros(count, 45)
    task.privileged_obs_buf = torch.zeros(count, 235)
    task.rew_buf = torch.zeros(count)
    task.reset_buf = torch.zeros(count, dtype=torch.bool)
    task.time_out_buf = torch.zeros(count, dtype=torch.bool)
    task.feet_air_time = torch.zeros(count, 4)
    task.episode_length_buf = torch.zeros(count, dtype=torch.long)
    task.common_step_counter = 0
    task.max_episode_length, task.max_episode_length_s = 1000, 20.
    task.dt = .02
    task.extras = {}
    task.cfg = NS(normalization=NS(clip_actions=2., clip_observations=100.),
                  control=NS(decimation=4), env=NS(test=False, send_timeouts=True),
                  domain_rand=NS(push_robots=False), commands=NS(curriculum=False),
                  rewards=NS(only_positive_rewards=True, tracking_sigma=.25))
    return task


def test_native_step_advances_decimation_then_rewards_snapshot_reset_observations(classes):
    task = state(classes.task)
    events = []
    task.gym = NS(
        set_dof_actuation_force_tensor=lambda *args: events.append("torque_set"),
        simulate=lambda *args: events.append("simulate"),
        fetch_results=lambda *args: events.append("fetch"),
        refresh_dof_state_tensor=lambda *args: events.append("dof_refresh"),
        refresh_actor_root_state_tensor=lambda *args: events.append("root_refresh"),
        refresh_net_contact_force_tensor=lambda *args: events.append("contact_refresh"))
    task.render = lambda: events.append("render")
    task._compute_torques = lambda actions: events.append("PD") or actions.clone()
    task._post_physics_step_callback = lambda: events.append("callback")
    def check():
        events.append("termination")
        task.reset_buf[:] = torch.tensor([False, True, False])
    task.check_termination = check
    task.compute_reward = lambda: events.append("reward")
    task._before_reset = lambda: events.append("terminal_snapshot")
    def reset(ids):
        events.append(("reset", ids.tolist()))
        task.root_states[ids, 7] = -99
    task.reset_idx = reset
    task.compute_observations = lambda: events.append("observation")
    returned = task.step(torch.full((3, 12), 1000.))
    assert events[:1] == ["render"]
    assert events[1:21] == ["PD", "torque_set", "simulate", "fetch", "dof_refresh"] * 4
    assert events[21:] == ["root_refresh", "contact_refresh", "callback", "termination",
                           "reward", "terminal_snapshot", ("reset", [1]), "observation"]
    assert len(returned) == 5
    assert returned[0].shape == (3, 45) and returned[1].shape == (3, 235)
    assert returned[3].tolist() == [False, True, False]
    assert task.episode_length_buf.tolist() == [1, 1, 1]
    torch.testing.assert_close(task.actions, torch.full((3, 12), 2.))
    torch.testing.assert_close(task.last_actions, task.actions)


def test_reward_scales_are_registered_once_and_drive_native_episode_sums(classes):
    task = state(classes.task)
    task.commands[:, 0] = 1.
    task.torques.fill_(2.)
    task.dof_vel.fill_(3.)
    # The task configuration, including a zero entry, is the registration source.
    task.reward_scales = {"tracking_lin_vel": 1.5, "joint_power": -2e-5,
                          "smoothness": -.01, "collision": -10., "torques": 0.}
    configured = task.reward_scales.copy()
    task._prepare_reward_function()
    assert set(task.reward_names) == set(configured) - {"torques"}
    assert task.reward_scales == {key: value * .02 for key, value in configured.items()
                                 if value != 0}
    task.compute_reward()
    expected_tracking = 1.5 * .02 * torch.exp(torch.full((3,), -4.))
    expected_power = torch.full((3,), 12 * 6 * -2e-5 * .02)
    torch.testing.assert_close(task.episode_sums["tracking_lin_vel"], expected_tracking)
    torch.testing.assert_close(task.episode_sums["joint_power"], expected_power)
    torch.testing.assert_close(task.rew_buf, (expected_tracking + expected_power).clamp_min(0))
    task.compute_reward()
    torch.testing.assert_close(task.episode_sums["tracking_lin_vel"], expected_tracking * 2)


def test_native_total_clip_keeps_negative_reward_stats_and_partial_reset(classes):
    task = state(classes.base)
    task.reward_scales = {"collision": -10.}
    task._prepare_reward_function()
    task.contact_forces[:, 0, 2] = 2.
    task.compute_reward()
    assert torch.equal(task.rew_buf, torch.zeros(3))
    torch.testing.assert_close(task.episode_sums["collision"], torch.full((3,), -.2))
    task.episode_length_buf[:] = torch.tensor([8, 9, 10])
    task.actions.fill_(5)
    task._reset_dofs = lambda ids: None
    task._reset_root_states = lambda ids: None
    task._resample_commands = lambda ids: None
    task.reset_idx(torch.tensor([1]))
    assert task.episode_length_buf.tolist() == [8, 0, 10]
    assert task.episode_sums["collision"].tolist() == pytest.approx([-.2, 0., -.2])
    assert task.extras["episode"]["rew_collision"].item() == pytest.approx(-.01)
    assert torch.count_nonzero(task.actions[1]) == 0
    assert (task.actions[[0, 2]] == 5).all()


def test_only_ten_paper_reward_terms_are_enabled_and_base_physx_unchanged(classes):
    cfg = classes.config()
    nonzero = {key: value for key, value in class_to_dict(cfg.rewards.scales).items()
               if value != 0}
    assert nonzero == {"tracking_lin_vel": 1.5, "tracking_ang_vel": .5,
        "lin_vel_z": -1., "ang_vel_xy": -.05, "orientation": -1.,
        "dof_acc": -2.5e-7, "joint_power": -2e-5, "collision": -10.,
        "action_rate": -.01, "smoothness": -.01}
    base_cfg = classes.base.__init__.__annotations__["cfg"]()
    assert class_to_dict(cfg.sim.physx) == class_to_dict(base_cfg.sim.physx)
