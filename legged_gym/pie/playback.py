"""Restore the policy's environment before creating a playback simulator."""
from copy import deepcopy

from .config import config_from_dict


def restore_playback_config(env_cfg, checkpoint):
    cfg = deepcopy(env_cfg)
    environment = checkpoint.get("environment_config")
    if not environment:
        raise ValueError("PIE playback requires the checkpoint's environment_config")
    cfg.pie = config_from_dict(environment)
    # Runtime environment count/device/headless are selected by the play CLI.
    cfg.pie.num_envs = cfg.env.num_envs
    cfg.env.episode_length_s = cfg.pie.episode_seconds
    cfg.seed = cfg.pie.seed

    observation = checkpoint.get("playback_config")
    if observation:
        scales = observation["command_scales"]
        if len(scales) != 3 or scales[0] != scales[1]:
            raise ValueError("Expected equal x/y command scales and one yaw scale")
        cfg.normalization.obs_scales.lin_vel = scales[0]
        cfg.normalization.obs_scales.ang_vel = scales[2]
        cfg.normalization.obs_scales.dof_pos = observation["joint_position_scale"]
        cfg.normalization.clip_observations = observation["clip_observations"]
        cfg.rewards.only_positive_rewards = observation["only_positive_rewards"]
        source = "checkpoint environment and observation settings"
    elif (checkpoint.get("rsl_rl_base") == "v1.0.2"
          and checkpoint.get("pie_checkpoint_version") == 2
          and "ppo_config" in checkpoint):
        # The earlier v1.0.2 PIE runner predates native command scaling and
        # positive reward clipping. Its PPOConfig has no schedule field.
        legacy = "schedule" not in checkpoint["ppo_config"]
        cfg.normalization.obs_scales.lin_vel = 1.0 if legacy else 2.0
        cfg.normalization.obs_scales.ang_vel = 1.0 if legacy else 0.25
        cfg.normalization.obs_scales.dof_pos = 1.0
        cfg.normalization.clip_observations = 100.0
        cfg.rewards.only_positive_rewards = not legacy
        source = "legacy v1.0.2 PIE settings" if legacy else "v1.0.2 formal PIE settings"
    else:
        raise ValueError("Checkpoint has no known observation settings; refusing incompatible playback")

    cfg.pie.randomization.enabled = False
    cfg.pie.observation_noise = False
    cfg.pie.terrain.curriculum = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    return cfg, source
