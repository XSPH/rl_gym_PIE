import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.task == "lite3_pie":
        import json
        from rsl_rl.runners.on_policy_runner_pie import evaluate
        from legged_gym.pie.playback import restore_playback_config
        if args.checkpoint_file is None:
            raise ValueError("PIE play requires --checkpoint_file")
        checkpoint = torch.load(args.checkpoint_file, map_location="cpu", weights_only=True)
        env_cfg, source = restore_playback_config(env_cfg, checkpoint)
        scales = env_cfg.normalization.obs_scales
        print("[PIE playback] {}: action_clip={}, command_scales=({}, {}, {}), initial_max_level={}".format(
            source, env_cfg.pie.robot.action_clip, scales.lin_vel, scales.lin_vel,
            scales.ang_vel, env_cfg.pie.terrain.initial_max_level), flush=True)
        del checkpoint
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        try:
            print(json.dumps(evaluate(env, args.checkpoint_file, steps=args.steps), indent=2))
        finally:
            env.close()
        return
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
