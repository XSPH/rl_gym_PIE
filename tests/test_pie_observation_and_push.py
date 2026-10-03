"""Actual native PIE sensors with CPU images and recorded indexed setters."""
from types import SimpleNamespace as NS
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from native_cpu_helpers import class_to_dict, load_native_classes
from test_native_environment import state


@pytest.fixture
def sensor():
    classes = load_native_classes()
    task = state(classes.task)
    task.cfg = classes.config()
    task.cfg.domain_rand.randomize_pie = False
    task.cfg.camera.height = task.cfg.camera.width = 8
    task.cfg.terrain.curriculum = False
    task.sim_params = NS(dt=.005)
    task.num_bodies = 5
    task.body_names = ["TORSO"] + task.cfg.asset.foot_names
    task.foot_indices = torch.arange(1, 5)
    task.terrain_levels = task.terrain_types = torch.zeros(3, dtype=torch.long)
    task.joint_order = torch.arange(12)
    task.inverse_joint_order = task.joint_order.clone()
    task.default_dof_pos = torch.tensor([[task.cfg.init_state.default_joint_angles[name] for name in task.cfg.asset.joint_names]])
    task.dof_pos[:] = task.default_dof_pos
    task.root_states[:, 2] = .3
    task.base_quat = task.root_states[:, 3:7]
    task.base_pos = task.root_states[:, :3]
    task.last_contacts = torch.zeros(3, 4, dtype=torch.bool)
    task.obs_scales = NS(ang_vel=.25, lin_vel=2., dof_pos=1., dof_vel=.05)
    task.commands_scale = torch.tensor([2., 2., .25])
    task.add_noise = False
    task.noise_scale_vec = task._get_noise_scale_vec(task.cfg)
    task.add_noise = False
    rigid = torch.zeros(3, 5, 13)
    setters = []
    task.gym = NS(acquire_rigid_body_state_tensor=lambda sim: rigid,
                  refresh_rigid_body_state_tensor=lambda sim: None,
                  set_actor_root_state_tensor_indexed=lambda sim, root, ids, count:
                    setters.append((root.clone(), ids.clone(), count)))
    task.setters = setters
    actions_pointer = task.actions.data_ptr()
    task._init_pie_buffers()
    assert task.actions.data_ptr() == actions_pointer
    urdf = task.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(Path(__file__).resolve().parents[1]))
    task.fk = classes.kinematics.UrdfKinematics(urdf, task.cfg.asset.joint_names,
                            task.cfg.asset.base_name, "cpu")
    task.terrain_sampler = NS(sample=lambda points: torch.zeros(points.shape[:-1]))
    image = torch.zeros(3, 8, 8)
    def render(ids=None):
        selected = torch.arange(3) if ids is None else ids
        captured = (task.camera_capture_serial + 1) * 3 + selected
        image[selected] = captured[:, None, None].float()
        return image
    task.camera = NS(positions=torch.zeros(3, 3), orientations=torch.zeros(3, 4),
                     focal=torch.zeros(3), render=render, encode=lambda values: values)
    task._reset_pie_sensors(torch.arange(3))
    task.compute_observations()
    return task


def test_proprio_native_scaling_noise_and_privileged_clip_keep_raw_labels(sensor):
    sensor.commands[:] = torch.tensor([1., 0., 1., 0.])
    clean = sensor._proprio()
    torch.testing.assert_close(clean[:, 6:9], torch.tensor([[2., 0., .25]]).expand(3, -1))
    sensor.add_noise = True
    with patch("torch.rand_like", side_effect=lambda tensor: torch.ones_like(tensor)):
        noisy = sensor._proprio()
    torch.testing.assert_close(noisy-clean, sensor.noise_scale_vec.expand(3, -1))
    sensor.add_noise = False
    sensor.root_states[:, 7] = 1000
    sensor.compute_observations()
    obs = sensor.get_pie_observations()
    assert obs["critic"][:, 45].max() == 100
    assert obs["targets"]["velocity"][:, 0].min() == 1000
    assert obs["proprio"].shape == (3, 45)
    assert obs["proprio_history"].shape == (3, 10, 45)
    assert obs["depth"].shape == (3, 2, 8, 8)


def test_terminal_snapshot_precedes_reset_and_history_only_changes_selected_row(sensor):
    sensor.root_states[:, 7] = torch.tensor([1., 7., 3.])
    sensor._sync_base_quantities()
    sensor._termination_reasons = {"timeout": torch.tensor([False, True, False])}
    sensor.reset_buf[:] = torch.tensor([False, True, False])
    sensor.time_out_buf.copy_(sensor.reset_buf)
    sensor.common_step_counter = 1
    sensor._before_reset()
    terminal = sensor.extras["pie"]["terminal_critic"].clone()
    sensor.root_states[1, 7:13] = 0
    continuing_depth = sensor.depth_history[[0, 2]].clone()
    sensor._reset_pie_sensors(torch.tensor([1]))
    sensor.compute_observations()
    assert terminal[1, 45] == 7
    assert sensor.privileged_obs_buf[1, 45] == 0
    assert sensor.extras["pie"]["terminal_critic"][1, 45] == 7
    torch.testing.assert_close(sensor.depth_history[[0, 2]], continuing_depth, rtol=0, atol=0)
    torch.testing.assert_close(sensor.proprio_history[1], sensor.obs_buf[1].expand(10, -1))
    assert sensor._pending_reset.sum() == 0


def test_delayed_pd_hook_preserves_action_order_and_native_effort_bounds(sensor):
    sensor.cfg.control.control_type = "P"
    sensor.cfg.control.action_scale = .25
    sensor.p_gains, sensor.d_gains = torch.full((12,), 30.), torch.full((12,), .8)
    sensor.torque_limits = torch.full((12,), 3.)
    sensor.dof_pos_limits = torch.tensor([[-10., 10.]]).expand(12, -1)
    sensor.delay_steps[:] = torch.tensor([0, 1, 3])
    actions = torch.ones(3, 12)
    first = sensor._compute_torques(actions)
    assert (first[0] == 3).all() and (first[1:] == 0).all()
    second = sensor._compute_torques(actions)
    assert (second[:2] == 3).all() and (second[2] == 0).all()
    sensor._compute_torques(actions)
    fourth = sensor._compute_torques(actions)
    assert (fourth == 3).all()
    before = sensor.action_queue[[0, 2]].clone()
    sensor._reset_pie_sensors(torch.tensor([1]))
    assert torch.count_nonzero(sensor.action_queue[1]) == 0
    torch.testing.assert_close(sensor.action_queue[[0, 2]], before, rtol=0, atol=0)


def test_capture_ids_follow_actual_latency_queue_and_partial_reset(sensor):
    from rsl_rl.storage.rollout_storage_pie import DepthFramePool
    pool = DepthFramePool()
    captured = []
    def record():
        expected = sensor.depth_frame_ids[:, :, None, None].expand_as(sensor.depth_history).float()
        torch.testing.assert_close(sensor.depth_history, expected, rtol=0, atol=0)
        captured.append((sensor.depth_history.clone(), pool.add(sensor.depth_history, sensor.depth_frame_ids)))
    record()
    initial_ids = sensor.depth_frame_ids.clone()
    sensor._render_depth()
    record()
    torch.testing.assert_close(sensor.depth_frame_ids, initial_ids)
    sensor._render_depth()
    record()
    assert (sensor.depth_frame_ids[:, -1] != initial_ids[:, -1]).all()
    before = sensor.depth_frame_ids.clone()
    sensor._render_depth(torch.tensor([1]), reset=True)
    record()
    torch.testing.assert_close(sensor.depth_frame_ids[[0, 2]], before[[0, 2]])
    assert sensor.depth_frame_ids[1].unique().numel() == 1
    sensor.depth_history.fill_(-999)
    for original, indices in captured:
        torch.testing.assert_close(pool.materialize()[indices], original, rtol=0, atol=0)


def test_native_push_preserves_nonplanar_velocity_components(sensor):
    sensor.cfg.domain_rand.max_push_vel_xy = 1.
    sensor.cfg.domain_rand.push_interval = 5
    sensor.episode_length_buf[:] = torch.tensor([1, 5, 10])
    sensor.root_states[:, 7:13] = torch.arange(18).reshape(3, 6)
    before = sensor.root_states.clone()
    sensor._push_robots()
    assert len(sensor.setters) == 1
    pushed, indices, count = sensor.setters[0]
    assert indices.dtype == torch.int32 and indices.tolist() == [1, 2] and count == 2
    torch.testing.assert_close(pushed[0], before[0], rtol=0, atol=0)
    torch.testing.assert_close(pushed[:, :7], before[:, :7], rtol=0, atol=0)
    torch.testing.assert_close(pushed[:, 9:13], before[:, 9:13], rtol=0, atol=0)
    assert (pushed[1:, 7:9].abs() <= 1).all()


def test_failure_at_timeout_is_terminal_and_not_bootstrapped(sensor):
    sensor.episode_length_buf.fill_(sensor.max_episode_length + 1)
    sensor.contact_forces[1, 0, 2] = 2.
    sensor.check_termination()
    assert sensor.reset_buf.all()
    assert sensor.time_out_buf.tolist() == [True, False, True]
    assert sensor._termination_reasons["base_contact"].tolist() == [False, True, False]


def test_real_task_step_runs_native_rewards_resets_and_pie_terminal_snapshot(sensor):
    """Only Gym setters/physics advance are fake; all task methods are real."""
    sensor.p_gains, sensor.d_gains = torch.full((12,), 30.), torch.full((12,), .8)
    sensor.torque_limits = torch.full((12,), 30.5)
    sensor.dof_pos_limits = torch.tensor([[-10., 10.]]).expand(12, -1)
    sensor.command_ranges = class_to_dict(sensor.cfg.commands.ranges)
    sensor.reward_scales = class_to_dict(sensor.cfg.rewards.scales)
    sensor._prepare_reward_function()
    sensor.cfg.domain_rand.push_robots = False
    sensor.cfg.env.test = False
    sensor.custom_origins = True
    sensor.env_origins = torch.zeros(3, 3)
    sensor.base_init_state = torch.tensor([0., 0., .3, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
    sensor.gym.set_dof_state_tensor_indexed = lambda *args: None
    sensor.gym.set_dof_actuation_force_tensor = lambda *args: None
    advances = []
    sensor.gym.simulate = lambda *args: advances.append("simulate")
    sensor.gym.fetch_results = lambda *args: None
    sensor.gym.refresh_dof_state_tensor = lambda *args: None
    sensor.gym.refresh_actor_root_state_tensor = lambda *args: None
    sensor.gym.refresh_net_contact_force_tensor = lambda *args: None
    sensor.root_states[:, 7] = torch.tensor([1., 7., 3.])
    sensor.contact_forces[1, 0, 2] = 2.
    sensor.episode_length_buf.fill_(10)
    obs, critic, reward, done, extras = sensor.step(torch.full((3, 12), .1))
    assert len(advances) == sensor.cfg.control.decimation == 4
    assert done.tolist() == [False, True, False]
    assert extras["pie"]["terminal_critic"][1, 45] == 7
    assert critic[1, 45].abs() <= .5  # Native reset velocity range, not old episode.
    assert extras["episode"]["rew_collision"].item() == pytest.approx(-.01)
    assert sensor.episode_length_buf.tolist() == [11, 0, 11]
    assert sensor.episode_sums["collision"].tolist() == [0., 0., 0.]
    assert torch.count_nonzero(sensor.actions[1]) == 0
    assert (obs[1, -12:] == 0).all()
    torch.testing.assert_close(sensor.proprio_history[1], obs[1].expand(10, -1))
    assert sensor.depth_frame_ids[1].unique().numel() == 1


def test_nonfinite_state_stops_before_warp_render(sensor):
    rendered = []
    sensor.common_step_counter = sensor.cfg.camera.update_every
    sensor.camera.render = lambda *args: rendered.append(True)
    sensor.root_states[1, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        sensor._before_reset()
    assert not rendered


def test_camera_extrinsics_follow_robot_yaw_and_point_downward(sensor):
    quat_rotate = load_native_classes().torch_utils.quat_apply
    angle = torch.tensor(torch.pi / 4)
    sensor.root_states[1, 3:7] = torch.tensor([0., 0., torch.sin(angle), torch.cos(angle)])
    sensor._render_depth()
    offsets = sensor.camera.positions - sensor.root_states[:, :3]
    torch.testing.assert_close(offsets[0], torch.tensor([.25, 0., .06]), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(offsets[1], torch.tensor([0., .25, .06]), atol=1e-7, rtol=1e-6)
    forward = quat_rotate(sensor.camera.orientations, torch.tensor([[1., 0., 0.]]).expand(3, -1))
    pitch = torch.tensor(torch.pi / 6)
    torch.testing.assert_close(forward[0], torch.tensor([torch.cos(pitch), 0., -torch.sin(pitch)]),
                               atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(forward[1], torch.tensor([0., torch.cos(pitch), -torch.sin(pitch)]),
                               atol=5e-7, rtol=1e-6)  # Native quat_mul reorders FP32 arithmetic.


def test_policy_joint_order_maps_actions_and_sensors_to_native_dofs(sensor):
    sensor.joint_order = torch.arange(11, -1, -1)
    sensor.inverse_joint_order = torch.argsort(sensor.joint_order)
    sensor.p_gains, sensor.d_gains = torch.full((12,), 30.), torch.full((12,), .8)
    sensor.torque_limits = torch.full((12,), 30.5)
    sensor.dof_pos_limits = torch.tensor([[-10., 10.]]).expand(12, -1)
    actions = torch.zeros(3, 12)
    actions[:, 0] = 1.
    torque = sensor._compute_torques(actions)
    torch.testing.assert_close(torque[:, -1], torch.full((3,), 7.5))
    assert torch.count_nonzero(torque[:, :-1]) == 0
    sensor.dof_pos[:, -1] += .2
    sensor.actions.copy_(actions)
    proprio = sensor._proprio()
    torch.testing.assert_close(proprio[:, 9], torch.full((3,), .2))
    assert torch.count_nonzero(proprio[:, 10:21]) == 0
    torch.testing.assert_close(proprio[:, -12:], actions)
