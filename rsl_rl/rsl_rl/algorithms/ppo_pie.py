"""PIE recurrent joint estimator/PPO extension of RSL-RL v1.0.2."""
from dataclasses import dataclass
import torch
from .ppo import PPO
from rsl_rl.storage.rollout_storage_pie import PIERolloutStorage, gae

@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 2
    minibatches: int = 2
    entropy_weight: float = 0.01
    value_weight: float = 1.0
    estimation_weight: float = 1.0
    kl_weight: float = 1.0
    max_grad_norm: float = 1.0

def clone_observation(obs):
    return {key: value.detach().clone() for key, value in obs.items()
            if key in ("proprio", "proprio_history", "depth", "critic")}

class PIEPPO(PPO):
    """One optimizer jointly trains actor, critic, encoder, and estimators.

    Actor gradients pass through the estimator as well as its supervised losses.
    This is an explicit reproduction choice; the paper specifies concurrent
    optimization without defining the optimizer/gradient boundary.
    """
    def __init__(self, actor_critic, cfg=None, device=None, pie_config=None):
        if cfg is not None and pie_config is not None:
            raise ValueError("Provide cfg or pie_config, not both")
        self.cfg = cfg or PPOConfig(**(pie_config or {}))
        c = self.cfg
        if c.epochs < 1 or c.minibatches < 1:
            raise ValueError("epochs/minibatches must be positive")
        super().__init__(
            actor_critic, num_learning_epochs=c.epochs, num_mini_batches=c.minibatches,
            clip_param=c.clip, gamma=c.gamma, lam=c.gae_lambda,
            value_loss_coef=c.value_weight, entropy_coef=c.entropy_weight,
            learning_rate=c.learning_rate, max_grad_norm=c.max_grad_norm,
            use_clipped_value_loss=True, schedule="fixed", desired_kl=None,
            device=device or str(next(actor_critic.parameters()).device))
        self.model = self.actor_critic

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, action_shape):
        self.storage = PIERolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape,
            action_shape, device=self.device)

    def act(self, obs, critic_obs):
        raise RuntimeError("Use PIEPPO.collect for dict observations and pre-reset labels")

    def process_env_step(self, rewards, dones, infos):
        raise RuntimeError("Use PIEPPO.collect for terminal/timeout-aware transitions")

    def compute_returns(self, last_critic_obs=None):
        self.storage.compute_returns(None, self.gamma, self.lam)

    def test_mode(self):
        self.actor_critic.eval()

    @torch.no_grad()
    def collect(self, env, obs, hidden, reset_mask, steps):
        if steps < 1:
            raise ValueError("rollout_steps must be positive")
        if (self.storage is None or self.storage.num_envs != hidden.shape[0]
                or self.storage.num_transitions_per_env != steps):
            self.init_storage(hidden.shape[0], steps, [obs["proprio"].shape[-1]],
                              [obs["critic"].shape[-1]], [self.model.cfg.action_dim])
        self.storage.start(hidden)
        for _ in range(steps):
            stored = clone_observation(obs)
            targets = {k: v.detach().clone() for k, v in obs["targets"].items()}
            actions, logp, values, next_hidden = self.model.act(obs, hidden, reset_mask)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            terminated, truncated = terminated.bool().clone(), truncated.bool().clone()
            next_values = self.model.value(next_obs)
            if truncated.any():
                final_obs = info.get("terminal_observation")
                if final_obs is None:
                    raise RuntimeError("time limits require terminal_observation BEFORE reset")
                final_value = self.model.value(final_obs)
                next_values = torch.where(truncated, final_value, next_values)
            successor = info.get("terminal_proprio")
            if successor is None:
                if (terminated | truncated).any():
                    raise RuntimeError("autoreset transitions require pre-reset terminal_proprio")
                successor = next_obs["proprio"]
            valid = torch.isfinite(stored["proprio"]).all(-1)
            successor_valid = torch.isfinite(successor).all(-1)
            self.storage.add_frame({
                "obs": stored, "targets": targets, "actions": actions.detach().clone(),
                "old_logp": logp.detach().clone(), "values": values.detach().clone(),
                "rewards": rewards.detach().clone(), "next_values": next_values.detach().clone(),
                "terminated": terminated, "truncated": truncated,
                "successor": successor.detach().clone(), "valid": valid,
                "successor_valid": successor_valid,
                "reset": reset_mask.detach().clone()},
                self.model.action_mean.detach(), self.model.action_std.detach())
            obs, hidden = next_obs, next_hidden.detach()
            reset_mask = terminated | truncated
        self.compute_returns()
        batch = self.storage.as_batch()
        return batch, obs, hidden, reset_mask

    def update(self, batch=None):
        """Minibatches contain whole environment trajectories, not shuffled steps."""
        batch = self.storage.as_batch() if batch is None else batch
        frames = batch["frames"]
        batch_size = batch["hidden"].shape[0]
        records = []
        if self.cfg.epochs < 1 or self.cfg.minibatches < 1:
            raise ValueError("epochs/minibatches must be positive")
        self.model.train()
        for ids in self.storage.trajectory_mini_batch_generator(
                batch_size, batch["hidden"].device, self.num_mini_batches,
                self.num_learning_epochs):
            hidden = batch["hidden"][ids]
            logps, entropies, values, aux = [], [], [], []
            for frame in frames:
                obs = {k: v[ids] for k, v in frame["obs"].items()}
                logp, entropy, value, hidden, estimates = self.model.evaluate(
                    obs, hidden, frame["actions"][ids], frame["reset"][ids])
                logps.append(logp); entropies.append(entropy); values.append(value)
                targets = {k: v[ids] for k, v in frame["targets"].items()}
                aux.append(self.model.auxiliary_losses(
                    estimates, targets, frame["successor"][ids], frame["valid"][ids],
                    frame["successor_valid"][ids]))
            logps, entropies, values = torch.stack(logps), torch.stack(entropies), torch.stack(values)
            old_logps = torch.stack([f["old_logp"][ids] for f in frames])
            ratio = (logps - old_logps).exp()
            advantage = batch["advantages"][:, ids]
            policy = -torch.minimum(ratio * advantage,
                ratio.clamp(1-self.clip_param, 1+self.clip_param) * advantage).mean()
            old_values = torch.stack([f["values"][ids] for f in frames])
            clipped_values = old_values + (values-old_values).clamp(-self.clip_param, self.clip_param)
            returns = batch["returns"][:, ids]
            value_loss = torch.maximum((values-returns).square(),
                                       (clipped_values-returns).square()).mean()
            losses = {k: torch.stack([a[k] for a in aux]).mean() for k in aux[0]}
            estimation = sum(losses[k] for k in ("velocity", "foot_clearance", "heightmap", "successor"))
            total = (policy + self.value_loss_coef * value_loss
                     - self.entropy_coef * entropies.mean()
                     + self.cfg.estimation_weight * (estimation + self.cfg.kl_weight * losses["kl"]))
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite PPO/estimator loss")
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            with torch.no_grad():
                self.model.std.clamp_(min=0.006737946999085467, max=7.38905609893065)
            records.append({**{k: float(v.detach()) for k, v in losses.items()},
                "loss": float(total.detach()), "policy": float(policy.detach()),
                "value": float(value_loss.detach()), "grad_norm": float(grad_norm.detach())})
        return {key: sum(r[key] for r in records)/len(records) for key in records[0]}

    @torch.no_grad()
    def refresh_hidden(self, batch):
        """Replay this rollout under updated weights, retaining episode memory.

        The detached rollout-start state is the truncated recurrent boundary.
        Done masks reset only the corresponding episode. The successor
        observation has not been encoded yet and belongs to the next rollout.
        """
        hidden = batch["hidden"].detach().clone()
        for frame in batch["frames"]:
            _, hidden = self.model.encode(frame["obs"], hidden, frame["reset"])
        final = batch["frames"][-1]
        reset_mask = final["terminated"] | final["truncated"]
        hidden = hidden * (~reset_mask).unsqueeze(-1)
        return hidden.detach(), reset_mask.detach().clone()


