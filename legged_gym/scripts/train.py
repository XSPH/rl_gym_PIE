import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    try:
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
        iterations = train_cfg.runner.max_iterations
        if args.task == "lite3_pie":
            # max_iterations is the total target, including a loaded checkpoint.
            iterations -= ppo_runner.current_learning_iteration
            if iterations <= 0:
                print("Checkpoint already reached the requested total iterations.")
                return
        ppo_runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    finally:
        if args.task == "lite3_pie":
            env.close()

if __name__ == '__main__':
    args = get_args()
    train(args)
