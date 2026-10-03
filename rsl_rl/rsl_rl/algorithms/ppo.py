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

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.metrics = {}

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def _mini_batch_generator(self):
        if self.actor_critic.is_recurrent:
            return self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)
        return self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

    def _evaluate_batch(self, batch):
        obs, critic_obs, actions, _, _, _, _, _, _, hidden_states, masks = batch
        self.actor_critic.act(obs, masks=masks, hidden_states=hidden_states[0])
        return {
            "logp": self.actor_critic.get_actions_log_prob(actions),
            "value": self.actor_critic.evaluate(critic_obs, masks=masks, hidden_states=hidden_states[1]),
            "mu": self.actor_critic.action_mean,
            "sigma": self.actor_critic.action_std,
            "entropy": self.actor_critic.entropy,
        }

    def _auxiliary_loss(self, evaluation):
        """Optional estimator loss; ordinary PPO has no auxiliary objective."""
        return evaluation["value"].new_zeros(()), {}

    def _after_optimizer_step(self):
        pass

    def _after_update(self):
        pass

    def update(self):
        """Original PPO equations/optimizer shared by ordinary and PIE policies."""
        records = []
        for batch in self._mini_batch_generator():
            (_, _, _, target_values_batch, advantages_batch, returns_batch,
             old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, _, _) = batch
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

            auxiliary_loss, auxiliary_metrics = self._auxiliary_loss(evaluation)
            loss = (surrogate_loss + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean() + auxiliary_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO/estimator loss")

            # One joint Adam step, with the original clipping boundary.
            self.optimizer.zero_grad()
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self._after_optimizer_step()
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
        self._after_update()
        self.storage.clear()
        return self.metrics["value"], self.metrics["policy"]
