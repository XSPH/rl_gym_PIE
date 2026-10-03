"""PIE recurrent joint estimator/PPO extension of RSL-RL v1.0.2."""
from dataclasses import dataclass
from typing import Optional
import math
import torch
from .ppo import PPO
from rsl_rl.storage.rollout_storage_pie import PIERolloutStorage
from rsl_rl.modules.actor_critic_pie import PIEDepthFeatureCache

@dataclass
class PPOConfig:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 4
    entropy_weight: float = 0.01
    value_weight: float = 1.0
    estimation_weight: float = 1.0
    kl_weight: float = 1.0
    max_grad_norm: float = 1.0
    schedule: str = "adaptive"
    desired_kl: Optional[float] = 0.01

    def __post_init__(self):
        if self.schedule not in ("fixed", "adaptive"):
            raise ValueError("schedule must be fixed or adaptive")
        if self.desired_kl is not None and (
                not math.isfinite(self.desired_kl) or self.desired_kl <= 0):
            raise ValueError("desired_kl must be positive and finite, or None")

    def as_native_kwargs(self):
        return {
            "num_learning_epochs": self.epochs, "num_mini_batches": self.minibatches,
            "clip_param": self.clip, "gamma": self.gamma, "lam": self.gae_lambda,
            "value_loss_coef": self.value_weight, "entropy_coef": self.entropy_weight,
            "learning_rate": self.learning_rate, "max_grad_norm": self.max_grad_norm,
            "schedule": self.schedule, "desired_kl": self.desired_kl,
            "estimation_weight": self.estimation_weight, "kl_weight": self.kl_weight,
        }

def clone_observation(obs, depth_pool=None):
    stored = {key: value.detach().clone() for key, value in obs.items()
              if key in ("proprio", "proprio_history", "critic")}
    if depth_pool is not None and "depth_frame_ids" in obs:
        stored["depth_indices"] = depth_pool.add(obs["depth"], obs["depth_frame_ids"])
    else:
        stored["depth"] = obs["depth"].detach().clone()
    return stored


class PIEPPO(PPO):
    """PIE side inputs and recurrent replay within the native PPO lifecycle."""
    def __init__(self, actor_critic, estimation_weight=1.0, kl_weight=1.0, **kwargs):
        super().__init__(actor_critic, **kwargs)
        self.estimation_weight = estimation_weight
        self.kl_weight = kl_weight
        self.model = self.actor_critic
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
        self.model._visual_cache = PIEDepthFeatureCache()

    def act(self, obs, critic_obs):
        # Snapshot sensor labels before env.step can overwrite/reset them.
        frame = {
            "obs": clone_observation(obs, self.storage.depth_pool),
            "targets": {name: value.detach().clone() for name, value in obs["targets"].items()},
            "valid": torch.isfinite(obs["proprio"]).all(-1),
            "reset": self._reset_mask.detach().clone(),
        }
        if self.model._hidden is None:
            self.model._hidden = self.model.initial_state(obs["proprio"].shape[0])
        actions = super().act(obs, critic_obs)
        self.transition.observations = frame["obs"]["proprio"]
        self.transition.critic_observations = frame["obs"]["critic"]
        self.transition.pie_frame = frame
        return actions

    def process_env_step(self, rewards, dones, infos):
        frame = self.transition.pie_frame
        done = dones.bool().flatten()
        timeout = infos.get("time_outs", torch.zeros_like(done)).to(self.device).bool().flatten()
        if (timeout & ~done).any():
            raise ValueError("A timeout must also be an episode reset")
        side = infos.get("pie", {})
        successor = side.get("terminal_proprio")
        if successor is None:
            raise RuntimeError("PIE requires pre-reset terminal_proprio for every transition")
        frame["raw_rewards"] = rewards.detach().clone()
        frame["terminated"] = (done & ~timeout).detach().clone()
        frame["truncated"] = timeout.detach().clone()
        frame["successor"] = successor.detach().clone()
        frame["successor_valid"] = torch.isfinite(successor).all(-1)
        bootstrapped_rewards = rewards
        if timeout.any():
            terminal_critic = side.get("terminal_critic")
            if terminal_critic is None:
                raise RuntimeError("Timeouts require pre-reset terminal_critic")
            with torch.no_grad():
                terminal_value = self.model.evaluate(terminal_critic).squeeze(-1)
            bootstrapped_rewards = rewards + self.gamma * terminal_value * timeout.to(rewards.dtype)
        # Parent storage/GAE handles done trace boundaries. Remove time_outs so
        # the original current-value bootstrap is not added a second time.
        parent_infos = {key: value for key, value in infos.items() if key != "time_outs"}
        super().process_env_step(bootstrapped_rewards, done, parent_infos)
        self.transition.pie_frame = None
        self._reset_mask = done.detach().clone()

    def _mini_batch_generator(self):
        batch = self.storage.as_batch()
        frames = batch["frames"]
        for ids in self.storage.trajectory_mini_batch_generator(
                self.storage.num_envs, self.device, self.num_mini_batches, self.num_learning_epochs):
            # The parent update owns PPO/optimizer equations. PIE changes only
            # the forward batch: one complete time sequence per selected env.
            actor_batch = {"frames": frames, "ids": ids, "hidden": batch["hidden"][ids],
                           "depth_frames": batch["depth_frames"]}
            def flatten(tensor):
                return tensor[:, ids].flatten(0, 1)
            yield (actor_batch, flatten(self.storage.privileged_observations),
                   flatten(self.storage.actions), flatten(self.storage.values),
                   flatten(self.storage.advantages), flatten(self.storage.returns),
                   flatten(self.storage.actions_log_prob), flatten(self.storage.mu),
                   flatten(self.storage.sigma), (None, None), None)

    def _evaluate_batch(self, batch):
        sequence = batch[0]
        ids, hidden = sequence["ids"], sequence["hidden"]
        visual_cache = PIEDepthFeatureCache()
        logps, entropies, values, means, sigmas, auxiliary = [], [], [], [], [], []
        for frame in sequence["frames"]:
            obs = self.storage.observation(frame, ids)
            visual = visual_cache.get(self.model, obs, sequence["depth_frames"])
            logp, entropy, value, hidden, estimates = self.model.evaluate_actions(
                obs, hidden, frame["actions"][ids], frame["reset"][ids], visual_features=visual)
            logps.append(logp); entropies.append(entropy); values.append(value)
            means.append(self.model.action_mean); sigmas.append(self.model.action_std)
            targets = {key: tensor[ids] for key, tensor in frame["targets"].items()}
            auxiliary.append(self.model.auxiliary_losses(
                estimates, targets, frame["successor"][ids], frame["valid"][ids],
                frame["successor_valid"][ids]))
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
            "cnn_dense_stacks": len(sequence["frames"]) * ids.numel(),
        }

    def _auxiliary_loss(self, evaluation):
        losses = evaluation["auxiliary"]
        estimation = sum(losses[key] for key in ("velocity", "foot_clearance", "heightmap", "successor"))
        total = self.estimation_weight * (estimation + self.kl_weight * losses["kl"])
        metrics = dict(losses)
        metrics.update(cnn_encoded_stacks=evaluation["cnn_encoded_stacks"],
                       cnn_dense_stacks=evaluation["cnn_dense_stacks"])
        return total, metrics

    def _after_optimizer_step(self):
        with torch.no_grad():
            self.model.std.clamp_(min=torch.finfo(self.model.std.dtype).eps)

    @torch.no_grad()
    def _after_update(self):
        batch = self.storage.as_batch()
        frames = batch["depth_frames"]
        if frames is not None:
            self.metrics.update(
                depth_unique_frames=frames.shape[0],
                depth_pool_mib=frames.numel() * frames.element_size() / 2**20,
                depth_dense_mib=(self.storage.num_transitions_per_env * self.storage.num_envs
                                 * self.model.cfg.depth_history * frames[0].numel()
                                 * frames.element_size() / 2**20))
        self.metrics["cnn_reuse_fraction"] = (1.0 - self.metrics["cnn_encoded_stacks"]
                                               / self.metrics["cnn_dense_stacks"])
        # Keep episode memory across optimizer updates, with current weights.
        hidden = batch["hidden"].detach().clone()
        visual_cache = PIEDepthFeatureCache()
        for frame in batch["frames"]:
            obs = self.storage.observation(frame)
            visual = visual_cache.get(self.model, obs, batch["depth_frames"])
            _, hidden = self.model.encode(obs, hidden, frame["reset"], visual_features=visual)
        self.model._hidden = (hidden * (~self._reset_mask).unsqueeze(-1)).detach()
        self.model._visual_cache = PIEDepthFeatureCache()
