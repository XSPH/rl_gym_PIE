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

"""PIE recurrent joint estimator/PPO extension of RSL-RL v1.0.2."""
import math
import torch
from torch import nn
from .ppo import PPO
from rsl_rl.storage.rollout_storage_pie import PIERolloutStorage
from rsl_rl.modules.actor_critic_pie import PIEDepthFeatureCache

class PIEPPO(PPO):
    """Native PPO interface with recurrent replay and joint PIE supervision."""
    def __init__(self, actor_critic, estimation_weight=1.0, kl_weight=1.0, **kwargs):
        if kwargs.get('schedule', 'fixed') not in ('fixed', 'adaptive'):
            raise ValueError("schedule must be fixed or adaptive")
        desired_kl = kwargs.get('desired_kl', .01)
        if desired_kl is not None and (not math.isfinite(desired_kl) or desired_kl <= 0):
            raise ValueError("desired_kl must be positive and finite, or None")
        super().__init__(actor_critic, **kwargs)
        self.estimation_weight = estimation_weight
        self.kl_weight = kl_weight
        self.transition = PIERolloutStorage.Transition()
        self.metrics = {}
        self._reset_mask = None

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, action_shape):
        # Playback may use one env with a saved four-minibatch training config.
        # Actual training validates minibatch size when generating trajectories.
        self.storage = PIERolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape,
            action_shape, device=self.device)
        self._reset_mask = torch.ones(num_envs, dtype=torch.bool, device=self.device)

    def begin_rollout(self):
        if self.storage.step:
            raise RuntimeError("Previous rollout has not been updated and cleared")
        self.actor_critic.begin_rollout(self.storage.num_envs)

    def act(self, obs, critic_obs):
        if self.actor_critic.get_hidden_states()[0] is None:
            self.actor_critic.begin_rollout(obs["proprio"].shape[0])
        actions = super().act(obs, critic_obs)
        self.storage.prepare_transition(obs, self.transition)
        self.transition.reset_mask = self._reset_mask.detach().clone()
        return actions

    def process_env_step(self, rewards, dones, infos):
        done = dones.bool().flatten()
        timeout = infos.get("time_outs", torch.zeros_like(done)).to(self.device).bool().flatten()
        if (timeout & ~done).any():
            raise ValueError("A timeout must also be an episode reset")
        side = infos.get("pie", {})
        successor = side.get("terminal_proprio")
        if successor is None:
            raise RuntimeError("PIE requires pre-reset terminal_proprio for every transition")
        self.transition.raw_rewards = rewards.detach().clone()
        self.transition.terminated = (done & ~timeout).detach().clone()
        self.transition.truncated = timeout.detach().clone()
        self.transition.successor = successor.detach().clone()
        self.transition.successor_valid = torch.isfinite(successor).all(-1)
        bootstrapped_rewards = rewards
        if timeout.any():
            terminal_critic = side.get("terminal_critic")
            if terminal_critic is None:
                raise RuntimeError("Timeouts require pre-reset terminal_critic")
            with torch.no_grad():
                terminal_value = self.actor_critic.evaluate(terminal_critic).squeeze(-1)
            bootstrapped_rewards = rewards + self.gamma * terminal_value * timeout.to(rewards.dtype)
        # Parent storage/GAE handles done trace boundaries. Remove time_outs so
        # the original current-value bootstrap is not added a second time.
        parent_infos = {key: value for key, value in infos.items() if key != "time_outs"}
        super().process_env_step(bootstrapped_rewards, done, parent_infos)
        self._reset_mask = done.detach().clone()

    def _evaluate_batch(self, batch):
        (observations, _, actions, _, _, _, _, _, _, hidden_states, reset_masks,
         targets, successors, valid, successor_valid, depth_frames) = batch
        hidden = hidden_states[0]
        num_steps, num_envs = reset_masks.shape
        actions = actions.view(num_steps, num_envs, -1)
        visual_cache = PIEDepthFeatureCache()
        logps, entropies, values, means, sigmas, auxiliary = [], [], [], [], [], []
        for step in range(num_steps):
            obs = {name: tensor[step] for name, tensor in observations.items()}
            visual = visual_cache.get(self.actor_critic, obs, depth_frames)
            logp, entropy, value, hidden, estimates = self.actor_critic.evaluate_actions(
                obs, hidden, actions[step], reset_masks[step], visual_features=visual)
            logps.append(logp)
            entropies.append(entropy)
            values.append(value)
            means.append(self.actor_critic.action_mean)
            sigmas.append(self.actor_critic.action_std)
            auxiliary.append(self.actor_critic.auxiliary_losses(
                estimates, {name: tensor[step] for name, tensor in targets.items()},
                successors[step], valid[step], successor_valid[step]))
        losses = {key: torch.stack([entry[key] for entry in auxiliary]).mean()
                  for key in auxiliary[0]}
        return {
            "logp": torch.stack(logps).flatten(0, 1),
            "entropy": torch.stack(entropies).flatten(0, 1),
            "value": torch.stack(values).flatten(0, 1).unsqueeze(-1),
            "mu": torch.stack(means).flatten(0, 1),
            "sigma": torch.stack(sigmas).flatten(0, 1),
            "auxiliary": losses,
            "cnn_encoded_stacks": visual_cache.encoded_stacks,
            "cnn_dense_stacks": num_steps * num_envs,
        }

    def _compute_auxiliary_loss(self, evaluation):
        losses = evaluation["auxiliary"]
        estimation = sum(losses[key] for key in ("velocity", "foot_clearance", "heightmap", "successor"))
        total = self.estimation_weight * (estimation + self.kl_weight * losses["kl"])
        metrics = dict(losses)
        metrics.update(cnn_encoded_stacks=evaluation["cnn_encoded_stacks"],
                       cnn_dense_stacks=evaluation["cnn_dense_stacks"])
        return total, metrics

    @torch.no_grad()
    def _refresh_hidden_states(self):
        frames = self.storage.depth_pool.materialize()
        if frames is not None:
            self.metrics.update(
                depth_unique_frames=frames.shape[0],
                depth_pool_mib=frames.numel() * frames.element_size() / 2**20,
                depth_dense_mib=(self.storage.num_transitions_per_env * self.storage.num_envs
                                 * self.actor_critic.depth_history * frames[0].numel()
                                 * frames.element_size() / 2**20))
        self.metrics["cnn_reuse_fraction"] = (1.0 - self.metrics["cnn_encoded_stacks"]
                                               / self.metrics["cnn_dense_stacks"])
        # Replay with updated weights, preserving episode memory between rollouts.
        hidden = self.storage.initial_hidden.detach().clone()
        visual_cache = PIEDepthFeatureCache()
        for step in range(self.storage.num_transitions_per_env):
            obs = self.storage.observation(step)
            visual = visual_cache.get(self.actor_critic, obs, frames)
            _, hidden = self.actor_critic.encode(
                obs, hidden, self.storage.reset_mask[step], visual_features=visual)
        self.actor_critic.set_hidden_states(hidden * (~self._reset_mask).unsqueeze(-1))

    def update(self):
        """Joint PPO and estimator update over complete environment trajectories."""
        records = []
        for batch in self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs):
            (_, _, _, target_values_batch, advantages_batch, returns_batch,
             old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, _, _) = batch[:11]
            evaluation = self._evaluate_batch(batch)
            actions_log_prob_batch = evaluation["logp"]
            value_batch = evaluation["value"]
            mu_batch, sigma_batch = evaluation["mu"], evaluation["sigma"]
            entropy_batch = evaluation["entropy"]

            # Original v1.0.2 diagonal Gaussian KL and adaptive learning rate.
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch)) - 0.5, dim=-1)
                kl_mean = torch.mean(kl)
                if self.desired_kl is not None and self.schedule == 'adaptive':
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
            surrogate = -advantages_batch.squeeze(-1) * ratio
            surrogate_clipped = -advantages_batch.squeeze(-1) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            auxiliary_loss, auxiliary_metrics = self._compute_auxiliary_loss(evaluation)
            loss = (surrogate_loss + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean() + auxiliary_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO/estimator loss")

            # One joint Adam step, with the original clipping boundary.
            self.optimizer.zero_grad()
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            with torch.no_grad():
                self.actor_critic.std.clamp_(min=torch.finfo(self.actor_critic.std.dtype).eps)
            record = {
                "value": value_loss.item(), "policy": surrogate_loss.item(),
                "loss": loss.item(), "policy_kl": kl_mean.item(),
                "grad_norm": float(gradient_norm),
            }
            record.update({name: float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                           for name, value in auxiliary_metrics.items()})
            records.append(record)

        if not records:
            raise RuntimeError("PPO produced no minibatches")
        self.metrics = {name: sum(record[name] for record in records) / len(records)
                        for name in records[0]}
        # These are work counts, rather than losses averaged over updates.
        for name in ("cnn_encoded_stacks", "cnn_dense_stacks"):
            if name in self.metrics:
                self.metrics[name] = sum(record[name] for record in records)
        self._refresh_hidden_states()
        self.storage.clear()
        return self.metrics["value"], self.metrics["policy"]
