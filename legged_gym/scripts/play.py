import os
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    show_depth = getattr(args, "show_depth", False)
    if show_depth and args.task != "lite3_pie":
        raise ValueError("--show_depth is available for lite3_pie playback")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.task == "lite3_pie":
        import json
        from legged_gym.utils.helpers import restore_playback_config
        from legged_gym.utils.helpers import update_class_from_dict
        if args.checkpoint_file is None:
            raise ValueError("PIE play requires --checkpoint_file")
        checkpoint = torch.load(args.checkpoint_file, map_location="cpu", weights_only=True)
        env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else 1
        env_cfg, source = restore_playback_config(env_cfg, checkpoint)
        update_class_from_dict(train_cfg, checkpoint["train_config"])
        train_cfg.runner.resume = False
        print("[PIE playback] {}: envs={}, steps={}".format(
            source, env_cfg.env.num_envs, args.steps), flush=True)
        del checkpoint
        if args.steps < 1:
            raise ValueError("--steps must be positive")
        depth_env = getattr(args, "depth_env", 0)
        if show_depth and not 0 <= depth_env < env_cfg.env.num_envs:
            raise ValueError("--depth_env must be between 0 and num_envs - 1")
        env, depth_viewer = None, None
        try:
            if show_depth:
                from legged_gym.utils.depth_viewer import DepthViewer
                depth_viewer = DepthViewer(env_cfg.camera, env_cfg.env.num_envs, depth_env,
                                           env_cfg.sim.dt * env_cfg.control.decimation)
            env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
            runner, _ = task_registry.make_alg_runner(
                env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None)
            runner.load(args.checkpoint_file, load_optimizer=False)
            policy = runner.get_inference_policy(device=env.device)
            obs = env.get_observations()
            episode_count = 0
            reset_messages = 0
            with torch.no_grad():
                for step in range(args.steps):
                    if depth_viewer is not None:
                        depth_viewer.update(env)
                    actions = policy(obs)
                    obs, _, rewards, dones, infos = env.step(actions)
                    runner.alg.actor_critic.reset(dones)
                    if not torch.isfinite(rewards).all():
                        raise FloatingPointError("Non-finite playback reward")
                    episode_count += int(dones.sum().item())
                    if dones.any() and reset_messages < 5:
                        reasons = infos.get("pie", {}).get("termination_reasons", {})
                        counts = {name: int(value[dones.bool()].sum().item())
                                  for name, value in reasons.items()}
                        print("[PIE reset] step={} count={} reasons={}".format(
                            step + 1, int(dones.sum().item()), counts), flush=True)
                        reset_messages += 1
            print(json.dumps({"steps": args.steps, "environments": env.num_envs,
                              "completed_episodes": episode_count,
                              "checkpoint": args.checkpoint_file}, indent=2))
        finally:
            try:
                if depth_viewer is not None:
                    depth_viewer.close()
            finally:
                if env is not None:
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
