"""v1.0.2 rollout storage extended with visual sequences and PIE labels."""
import torch
from .rollout_storage import RolloutStorage


def gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    """Bootstrap pre-reset timeout states, never propagate traces across resets."""
    advantages = torch.zeros_like(rewards)
    following = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        bootstrap = torch.where(terminated[t], torch.zeros_like(next_values[t]), next_values[t])
        trace = torch.where(terminated[t] | truncated[t], torch.zeros_like(following), following)
        following = rewards[t] + gamma * bootstrap - values[t] + gamma * lam * trace
        advantages[t] = following
    return advantages, advantages + values


class PIERolloutStorage(RolloutStorage):
    """Native PPO tensors plus raw camera/history/label sequences for GRU replay."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = []
        self.initial_hidden = None

    def start(self, hidden):
        self.clear()
        self.initial_hidden = hidden.detach().clone()

    def add_frame(self, frame, action_mean, action_sigma):
        transition = self.Transition()
        transition.observations = frame["obs"]["proprio"]
        transition.critic_observations = frame["obs"]["critic"]
        transition.actions = frame["actions"]
        transition.rewards = frame["rewards"]
        transition.dones = frame["terminated"] | frame["truncated"]
        transition.values = frame["values"].unsqueeze(-1)
        transition.actions_log_prob = frame["old_logp"]
        transition.action_mean = action_mean
        transition.action_sigma = action_sigma
        super().add_transitions(transition)
        # Standard transition fields now refer to the native tensor storage.
        index = self.step - 1
        frame["obs"]["proprio"] = self.observations[index]
        frame["obs"]["critic"] = self.privileged_observations[index]
        for key, tensor in (("actions", self.actions), ("old_logp", self.actions_log_prob),
                            ("values", self.values), ("rewards", self.rewards)):
            frame[key] = tensor[index] if key == "actions" else tensor[index, :, 0]
        self.frames.append(frame)

    def compute_returns(self, last_values, gamma, lam):
        if self.step != self.num_transitions_per_env:
            raise RuntimeError("PIE requires a complete rollout")
        # Every frame retains its true next-state value, including timeouts.
        def stack(key):
            return torch.stack([frame[key] for frame in self.frames])
        advantages, returns = gae(
            self.rewards[:, :, 0], self.values[:, :, 0], stack("next_values"),
            stack("terminated"), stack("truncated"), gamma, lam)
        self.returns.copy_(returns.unsqueeze(-1))
        normalized = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        self.advantages.copy_(normalized.unsqueeze(-1))

    def as_batch(self):
        return {"frames": self.frames, "hidden": self.initial_hidden,
                "advantages": self.advantages[:, :, 0], "returns": self.returns[:, :, 0]}

    @staticmethod
    def trajectory_mini_batch_generator(batch_size, device, num_mini_batches, num_epochs):
        if num_mini_batches < 1 or num_epochs < 1 or batch_size < 1:
            raise ValueError("batch size, epochs and minibatches must be positive")
        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=device)
            yield from torch.tensor_split(indices, min(batch_size, num_mini_batches))

    def clear(self):
        super().clear()
        self.frames = []
        self.initial_hidden = None

