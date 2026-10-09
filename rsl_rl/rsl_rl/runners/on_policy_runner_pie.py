# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""PIE collection, joint updates, logging, and version-4 checkpoints."""
from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
import numpy as np
import torch
from .on_policy_runner import OnPolicyRunner
import os
import time
import statistics
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.modules import PIEActorCritic
from rsl_rl.algorithms import PIEPPO
from rsl_rl.utils.pie_config import (
    depth_input_mode, normalize_model_config, normalize_train_config, checkpoint_train_config,
)


def plain_config(value):
    """Serialize configuration fields, including nested BaseConfig classes."""
    if is_dataclass(value) and not isinstance(value, type):
        return plain_config(asdict(value))
    if isinstance(value, dict):
        return {key: plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_config(item) for item in value]
    # NumPy scalar subclasses (notably np.float64) can pass isinstance(float)
    # but their pickles require unsafe globals under torch weights_only loading.
    if isinstance(value, np.generic):
        return plain_config(value.item())
    if isinstance(value, np.ndarray):
        return plain_config(value.tolist())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, type) or hasattr(value, "__dict__"):
        result = {}
        for name in dir(value):
            if name.startswith('_'):
                continue
            item = getattr(value, name)
            if callable(item) and not isinstance(item, type):
                continue
            result[name] = plain_config(item)
        return result
    raise TypeError("Unsupported configuration value: {}".format(type(value).__name__))


class PIEOnPolicyRunner(OnPolicyRunner):
    """Native runner interface with PIE recurrent observations and supervision."""
    def __init__(self, env, train_cfg, log_dir=None, device=None):
        if not isinstance(train_cfg, dict) or not {"runner", "policy", "algorithm"}.issubset(train_cfg):
            raise TypeError("PIE runner requires the native runner/policy/algorithm train_cfg dictionary")
        device = torch.device(device or env.device)
        sim_device = torch.device(env.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if sim_device.type == "cuda" and sim_device.index is None:
            sim_device = torch.device("cuda", torch.cuda.current_device())
        if device != sim_device:
            raise ValueError("PIE requires matching simulation and RL devices")
        self.train_cfg = normalize_train_config(train_cfg)
        self.train_cfg["runner"]["completed_iteration_numbering"] = True
        runner = self.train_cfg["runner"]
        if runner["num_steps_per_env"] < 1 or runner["save_interval"] < 1:
            raise ValueError("PIE rollout length and save interval must be positive")
        self._rollout_reward_sum = None
        self._reset_counts = {}
        self.cfg = runner
        self.alg_cfg = self.train_cfg["algorithm"]
        self.policy_cfg = self.train_cfg["policy"]
        self.device = device
        self.env = env
        if runner["policy_class_name"] != "PIEActorCritic" or runner["algorithm_class_name"] != "PIEPPO":
            raise ValueError("PIEOnPolicyRunner requires PIEActorCritic and PIEPPO")
        num_critic_obs = env.num_privileged_obs if env.num_privileged_obs is not None else env.num_obs
        dimensions = dict(num_actor_obs=env.num_obs, num_critic_obs=num_critic_obs,
                          num_actions=env.num_actions)
        policy_kwargs = dict(self.policy_cfg)
        for name, dimension in dimensions.items():
            if policy_kwargs.pop(name, dimension) != dimension:
                raise ValueError("PIE model/environment mismatch for " + name)
        actor_critic = PIEActorCritic(**dimensions, **policy_kwargs).to(self.device)
        self.alg = PIEPPO(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = runner["num_steps_per_env"]
        self.save_interval = runner["save_interval"]
        self.alg.init_storage(env.num_envs, self.num_steps_per_env, [env.num_obs],
                              [num_critic_obs], [env.num_actions])
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self._reset_env()
        actor_critic.validate_observation(env.get_pie_observations(), env.num_actions)
        print("PIE training: envs={}, rollout={}, epochs={}, minibatches={}, lr={}, "
              "schedule={}, desired_kl={}, initial_std={}, target_iterations={}, save_interval={}".format(
                  env.num_envs, self.num_steps_per_env, self.alg.num_learning_epochs,
                  self.alg.num_mini_batches, self.alg.learning_rate, self.alg.schedule,
                  self.alg.desired_kl, actor_critic.init_noise_std,
                  runner.get("max_iterations", 15000), self.save_interval))

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if num_learning_iterations < 1:
            raise ValueError("num_learning_iterations must be positive")
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        start_iteration = self.current_learning_iteration
        tot_iter = start_iteration + num_learning_iterations
        self.last_metrics = {}
        for it in range(start_iteration, tot_iter):
            start = time.time()
            self._begin_rollout()
            # Camera tensors must remain usable by CNN backward; collection
            # therefore uses no_grad rather than inference_mode.
            with torch.no_grad():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(self.env.get_pie_observations(), critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device), critic_obs.to(self.device),
                        rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(rewards, dones, infos)
                    self._on_env_step(rewards, dones, infos)
                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            iteration_number = it + 1
            # Use an explicit logging view. Copying locals() here would include
            # the preceding iteration's locs dictionary and retain every rollout.
            locs = {"it": it, "iteration_number": iteration_number, "tot_iter": tot_iter,
                    "num_learning_iterations": num_learning_iterations,
                    "collection_time": collection_time, "learn_time": learn_time,
                    "mean_value_loss": mean_value_loss, "mean_surrogate_loss": mean_surrogate_loss,
                    "ep_infos": ep_infos, "rewbuffer": rewbuffer, "lenbuffer": lenbuffer}
            metrics = self._iteration_metrics(locs)
            locs['extra_log_string'] = self._extra_log_string(metrics)
            if self.log_dir is not None:
                self.log(locs)
            else:
                self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
                self.tot_time += collection_time + learn_time
            self.current_learning_iteration = it + 1
            metrics.update(
                iteration=it + 1, transitions=self.num_steps_per_env * self.env.num_envs,
                collection_time=collection_time, learning_time=learn_time,
                iteration_time=collection_time + learn_time, total_time=self.tot_time,
                total_timesteps=self.tot_timesteps,
                fps=self.num_steps_per_env * self.env.num_envs / max(collection_time + learn_time, 1e-9))
            self.last_metrics = metrics
            self._after_iteration(locs, metrics)
            if self.log_dir is not None and iteration_number % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(iteration_number)))
            ep_infos.clear()

        self.current_learning_iteration = tot_iter
        if self.log_dir is not None:
            self._save_final()
            self.writer.flush()
        return self.last_metrics

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        displayed_iteration = locs.get('iteration_number', locs['it'])
        target_iteration = locs.get('tot_iter', self.current_learning_iteration + locs['num_learning_iterations'])
        header = f" \033[1m Learning iteration {displayed_iteration}/{target_iteration} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{header.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{header.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        log_string += locs.get('extra_log_string', '')
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs.get('tot_iter', locs['num_learning_iterations']) - locs['it'] - 1):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        checkpoint = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration, "infos": infos,
            "learning_rate": float(self.alg.learning_rate),
            "model_config": self.alg.actor_critic.get_model_config(),
            "train_config": plain_config(checkpoint_train_config(
                self.train_cfg, self.alg.actor_critic.get_model_config())),
            "environment_cfg": plain_config(getattr(self.env, "cfg", {})),
            "seed": self.train_cfg.get("seed", 1), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "total_timesteps": self.tot_timesteps, "total_time": self.tot_time,
            "rsl_rl_base": "v1.0.2", "pie_checkpoint_version": 4,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            torch.save(checkpoint, temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load(self, path, load_optimizer=True):
        saved = torch.load(path, map_location=self.device, weights_only=True)
        if saved.get("pie_checkpoint_version") != 4 or saved.get("rsl_rl_base") != "v1.0.2":
            raise ValueError("This branch accepts only native PIE version-4 checkpoints")
        saved_mode = depth_input_mode(saved.get('environment_cfg', {}).get('camera', {}))
        current_mode = depth_input_mode(getattr(getattr(self.env, 'cfg', None), 'camera', {}))
        if saved_mode != current_mode:
            raise ValueError("Checkpoint camera.input_mode {!r} does not match this task's {!r}".format(
                saved_mode, current_mode))
        stored_config = normalize_model_config(saved["model_config"])
        current_config = self.alg.actor_critic.get_model_config()
        stored_config.pop("initial_std")
        current_config.pop("initial_std")
        if stored_config != current_config:
            raise ValueError("Checkpoint model configuration does not match this task")
        self.alg.actor_critic.load_state_dict(saved["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(saved["optimizer_state_dict"])
            self.alg.learning_rate = float(saved['learning_rate'])
            for group in self.alg.optimizer.param_groups:
                group['lr'] = self.alg.learning_rate
        self.current_learning_iteration = int(saved["iter"])
        self.tot_timesteps = int(saved.get("total_timesteps", 0))
        self.tot_time = float(saved.get("total_time", 0.0))
        torch.set_rng_state(saved["torch_rng"].cpu())
        if saved.get("cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        # Simulator episodes restart, as in native RSL resume.
        self._reset_env()
        self.alg.storage.clear()
        self.alg.actor_critic.reset()
        self.alg._reset_mask.fill_(True)
        return saved.get("infos")

    def get_inference_policy(self, device=None):
        model = self.alg.actor_critic
        if device is not None and torch.device(device) != torch.device(self.device):
            raise ValueError("PIE inference must use the environment device")
        model.eval()
        model.reset()
        @torch.no_grad()
        def policy(observations, reset_mask=None):
            if reset_mask is not None:
                model.reset(reset_mask)
            obs = observations if isinstance(observations, dict) else self.env.get_pie_observations()
            return model.act_inference(obs)
        return policy

    def add_git_repo_to_log(self, source_file):
        import subprocess
        if self.log_dir is None:
            return
        output = Path(self.log_dir)
        output.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(["git", "-C", str(Path(source_file).resolve().parent), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
        (output / "upstream_commit.txt").write_text(result.stdout)
        (output / "rsl_rl_base_commit.txt").write_text("2ad79cf0caa85b91721abfe358105f869a784121\n")

    def _reset_env(self):
        _, _ = self.env.reset()

    def _begin_rollout(self):
        self.alg.begin_rollout()
        self._rollout_reward_sum = torch.zeros((), device=self.device)
        self._reset_counts = {}

    def _on_env_step(self, rewards, dones, infos):
        self._rollout_reward_sum += rewards.sum()
        side = infos.get("pie", {})
        timeout = infos.get("time_outs", torch.zeros_like(dones)).bool()
        counts = {"timeouts": timeout.sum(), "failures": (dones.bool() & ~timeout).sum()}
        for name, flags in side.get("termination_reasons", {}).items():
            counts[name] = flags.bool().sum()
        for name, count in counts.items():
            if name not in self._reset_counts:
                self._reset_counts[name] = count.detach().clone()
            else:
                self._reset_counts[name] += count

    def _iteration_metrics(self, locs):
        metrics = dict(self.alg.metrics)
        metrics.update(learning_rate=self.alg.learning_rate,
                       mean_action_noise_std=float(self.alg.actor_critic.std.detach().mean()),
                       mean_reward=float(self._rollout_reward_sum) / (self.num_steps_per_env * self.env.num_envs))
        levels = getattr(self.env, "terrain_levels", None)
        if levels is not None:
            metrics.update(terrain_level=float(levels.float().mean()),
                           terrain_level_min=int(levels.min()), terrain_level_max=int(levels.max()))
        metrics["reset_counts"] = {name: int(value) for name, value in self._reset_counts.items()}
        if locs["rewbuffer"]:
            metrics["mean_episode_reward"] = sum(locs["rewbuffer"]) / len(locs["rewbuffer"])
            metrics["mean_episode_length"] = sum(locs["lenbuffer"]) / len(locs["lenbuffer"])
        ep_infos = locs["ep_infos"]
        if ep_infos:
            metrics["episode_rewards"] = {
                name: float(torch.cat([torch.as_tensor(info[name], device=self.device).flatten()
                                      for info in ep_infos if name in info]).float().mean())
                for name in ep_infos[0]}
        return metrics

    def _extra_log_string(self, metrics):
        fields = [
            ("Total loss:", "loss"), ("Velocity estimation loss:", "velocity"),
            ("Foot clearance loss:", "foot_clearance"), ("Height map reconstruction loss:", "heightmap"),
            ("Successor reconstruction loss:", "successor"), ("VAE KL loss:", "kl"),
            ("Policy KL divergence:", "policy_kl"), ("Gradient norm before clipping:", "grad_norm"),
            ("Learning rate:", "learning_rate"), ("Mean step reward:", "mean_reward"),
            ("Mean terrain level:", "terrain_level"), ("Depth frame pool (MiB):", "depth_pool_mib"),
            ("Equivalent dense depth (MiB):", "depth_dense_mib"), ("CNN feature reuse fraction:", "cnn_reuse_fraction"),
        ]
        result = ''.join("{:>35} {:.6f}\n".format(label, metrics[key])
                         for label, key in fields if key in metrics)
        for label, key in (("Min terrain level:", "terrain_level_min"), ("Max terrain level:", "terrain_level_max"),
                           ("Unique depth frames:", "depth_unique_frames"), ("CNN encoded stacks:", "cnn_encoded_stacks"),
                           ("Equivalent dense CNN stacks:", "cnn_dense_stacks")):
            if key in metrics:
                result += "{:>35} {}\n".format(label, int(metrics[key]))
        for name, count in metrics.get("reset_counts", {}).items():
            result += "{:>35} {}\n".format("Reset {}:".format(name), count)
        result += "{:>35} {}\n".format("Transitions this iteration:", self.num_steps_per_env * self.env.num_envs)
        return result

    def _after_iteration(self, locs, metrics):
        if self.log_dir is None:
            return
        output = Path(self.log_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        iteration = metrics["iteration"]
        tags = {
            "loss": "Loss/total", "velocity": "PIE/velocity_loss", "foot_clearance": "PIE/foot_clearance_loss",
            "heightmap": "PIE/heightmap_loss", "successor": "PIE/successor_loss", "kl": "PIE/vae_kl_loss",
            "policy_kl": "PIE/policy_kl", "grad_norm": "PIE/gradient_norm_before_clipping",
            "terrain_level": "Episode/terrain_level", "terrain_level_min": "Terrain/min_level",
            "terrain_level_max": "Terrain/max_level", "mean_reward": "Train/mean_step_reward",
        }
        for name, value in metrics.items():
            if name.startswith(("depth_", "cnn_")):
                tags[name] = "PIE/" + name
        for name, tag in tags.items():
            if name in metrics:
                self.writer.add_scalar(tag, metrics[name], iteration)
        for name, value in metrics["reset_counts"].items():
            self.writer.add_scalar("Reset/" + name, value, iteration)

    def _save_final(self):
        self.save(Path(self.log_dir) / "checkpoint.pt")
