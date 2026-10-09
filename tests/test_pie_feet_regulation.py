"""CTS foot-motion shaping, local terrain heights, and native reward integration."""
import math
from types import SimpleNamespace as NS

import pytest
import torch

from native_cpu_helpers import load_native_classes
from test_pie_observation_and_push import sensor


def feet_state(task, height=0.0, speed=1.0):
    """Place sphere centers for the requested sole-to-ground clearance."""
    task.rigid_body_states[:, task.foot_indices, :] = 0
    task.rigid_body_states[:, task.foot_indices, 2] = height + task.cfg.asset.foot_radius
    task.rigid_body_states[:, task.foot_indices, 7] = speed


def test_penalty_decreases_with_clearance_and_scales_quadratically_with_horizontal_speed(sensor):
    feet_state(sensor)
    low = sensor._reward_feet_regulation()
    torch.testing.assert_close(low, torch.full((3,), 4.))
    feet_state(sensor, height=.03)
    high = sensor._reward_feet_regulation()
    torch.testing.assert_close(high, torch.full((3,), 4 * math.exp(-4)))
    assert (high < low).all()
    feet_state(sensor, height=.03, speed=2.)
    torch.testing.assert_close(sensor._reward_feet_regulation(), 4 * high)
    sensor.rigid_body_states[:, sensor.foot_indices, 9] = 100
    torch.testing.assert_close(sensor._reward_feet_regulation(), 4 * high)
    feet_state(sensor, speed=0.)
    assert torch.count_nonzero(sensor._reward_feet_regulation()) == 0
    feet_state(sensor, height=-.01)
    torch.testing.assert_close(sensor._reward_feet_regulation(), low)
    assert torch.isfinite(sensor._reward_feet_regulation()).all()


def test_grounded_foot_center_gives_zero_clearance_consistent_with_auxiliary_target(sensor):
    feet_state(sensor, speed=1.)
    torch.testing.assert_close(
        sensor.rigid_body_states[:, sensor.foot_indices, 2],
        torch.full((3, 4), sensor.cfg.asset.foot_radius))
    assert torch.count_nonzero(sensor._terrain_targets()['foot_clearance']) == 0
    torch.testing.assert_close(sensor._reward_feet_regulation(), torch.full((3,), 4.))


def test_penalty_uses_each_foot_local_ground_and_is_independent_of_base_pose_and_camera(sensor):
    feet_state(sensor, height=.025)
    reference = sensor._reward_feet_regulation()
    platforms = torch.tensor([0., .1, .2, .3])
    sensor.rigid_body_states[:, sensor.foot_indices, 2] += platforms
    sensor.terrain_sampler = NS(sample=lambda points: platforms.expand(points.shape[:-1]))
    sensor.root_states[:, 2] = 10
    sensor.projected_gravity[:] = torch.tensor([1., 0., 0.])
    torch.testing.assert_close(sensor._reward_feet_regulation(), reference)
    # Reward accesses physical truth only, never camera data or predictions.
    sensor.cfg.camera.input_mode = 'zero'
    torch.testing.assert_close(sensor._reward_feet_regulation(), reference)
    sensor.cfg.camera.input_mode = 'depth'
    torch.testing.assert_close(sensor._reward_feet_regulation(), reference)


def test_penalty_refreshes_current_step_rigid_body_state_before_reading_feet(sensor):
    feet_state(sensor, height=1., speed=0.)
    calls = []
    def refresh(sim):
        calls.append(sim)
        feet_state(sensor, height=0., speed=2.)
    sensor.gym.refresh_rigid_body_state_tensor = refresh
    torch.testing.assert_close(sensor._reward_feet_regulation(), torch.full((3,), 16.))
    assert calls == [sensor.sim]


def test_native_registration_scales_once_and_logs_negative_episode_reward(sensor):
    assert sensor.cfg.rewards.scales.feet_regulation == -.05
    sensor.reward_scales = {'feet_regulation': sensor.cfg.rewards.scales.feet_regulation}
    feet_state(sensor, speed=2.)
    sensor._prepare_reward_function()
    assert sensor.reward_names == ['feet_regulation']
    assert sensor.reward_scales['feet_regulation'] == pytest.approx(-.05 * sensor.dt)
    sensor.compute_reward()
    torch.testing.assert_close(sensor.rew_buf, torch.full((3,), -16 * .05 * sensor.dt))
    torch.testing.assert_close(sensor.episode_sums['feet_regulation'], sensor.rew_buf)
    sensor.reward_scales = {'feet_regulation': 0.0}
    sensor._prepare_reward_function()
    assert 'feet_regulation' not in sensor.reward_names


@pytest.mark.parametrize('height', [0., -.3, float('nan'), float('inf')])
def test_enabled_penalty_rejects_invalid_exponential_height_scale(height):
    classes = load_native_classes()
    cfg = classes.config()
    cfg.rewards.base_height_target = height
    with pytest.raises(ValueError, match='feet_regulation.*base_height_target'):
        classes.task._validate_config(cfg, NS(dt=.005), 'cuda:0')
