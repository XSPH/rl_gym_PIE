"""Check playback restores a checkpoint's units without altering training defaults."""
from types import SimpleNamespace as NS

import pytest

from legged_gym.pie.config import EnvConfig
from legged_gym.pie.playback import restore_playback_config


def task_config():
    return NS(pie=EnvConfig(), seed=1, env=NS(num_envs=4, episode_length_s=20.0),
              normalization=NS(obs_scales=NS(lin_vel=2.0, ang_vel=0.25, dof_pos=1.0),
                               clip_observations=100.0),
              rewards=NS(only_positive_rewards=True),
              domain_rand=NS(randomize_friction=True, randomize_base_mass=True,
                             push_robots=True))


def test_legacy_checkpoint_restores_environment_and_raw_commands():
    training = task_config()
    saved = EnvConfig()
    saved.robot.action_clip = 4.0
    saved.command_seconds = 5.0
    saved.terrain.initial_max_level = 1
    cfg, source = restore_playback_config(training, {
        "environment_config": saved.to_dict(), "rsl_rl_base": "v1.0.2",
        "pie_checkpoint_version": 2, "ppo_config": {"epochs": 2}})
    assert cfg.pie.robot.action_clip == 4.0
    assert cfg.pie.command_seconds == 5.0
    assert cfg.pie.terrain.initial_max_level == 1
    assert (cfg.normalization.obs_scales.lin_vel, cfg.normalization.obs_scales.ang_vel) == (1.0, 1.0)
    assert cfg.pie.num_envs == 4
    assert not cfg.domain_rand.push_robots and not cfg.pie.randomization.enabled
    assert not cfg.pie.terrain.curriculum and not cfg.rewards.only_positive_rewards
    assert "legacy" in source
    assert training.pie.robot.action_clip == 100.0
    assert training.domain_rand.push_robots and training.pie.randomization.enabled


def test_formal_checkpoint_preserves_native_command_units():
    cfg, _ = restore_playback_config(task_config(), {
        "environment_config": EnvConfig().to_dict(), "rsl_rl_base": "v1.0.2",
        "pie_checkpoint_version": 2, "ppo_config": {"schedule": "adaptive"}})
    assert (cfg.normalization.obs_scales.lin_vel, cfg.normalization.obs_scales.ang_vel) == (2.0, 0.25)
    assert cfg.pie.robot.action_clip == 100.0
    assert cfg.rewards.only_positive_rewards


def test_new_checkpoint_observation_settings_take_precedence():
    cfg, _ = restore_playback_config(task_config(), {
        "environment_config": EnvConfig().to_dict(),
        "playback_config": {"command_scales": [3.0, 3.0, 0.5],
                            "joint_position_scale": 0.7, "clip_observations": 50.0,
                            "only_positive_rewards": False}})
    assert (cfg.normalization.obs_scales.lin_vel, cfg.normalization.obs_scales.ang_vel) == (3.0, 0.5)
    assert cfg.normalization.obs_scales.dof_pos == 0.7
    assert cfg.normalization.clip_observations == 50.0
    assert not cfg.rewards.only_positive_rewards


def test_unknown_checkpoint_does_not_silently_use_current_training_settings():
    with pytest.raises(ValueError, match="no known observation settings"):
        restore_playback_config(task_config(), {"environment_config": EnvConfig().to_dict()})
