"""v1.0.2 rollout storage extended with visual sequences and PIE labels."""
import torch
from .rollout_storage import RolloutStorage



class DepthFramePool:
    """Lossless per-rollout FP32 images keyed by captured-image IDs.

    History stacks and consecutive control observations share individual image
    rows. Lookup stays on the tensor device; there is no Python mapping or
    image quantization. The pool is sealed before PPO replay.
    """
    def __init__(self):
        self._source_ids = None
        self._pool_ids = None
        self._chunks = []
        self._tensor = None
        self.num_frames = 0

    def add(self, depth, frame_ids):
        if self._tensor is not None:
            raise RuntimeError("Cannot append to a sealed depth frame pool")
        if depth.ndim != 4 or frame_ids.shape != depth.shape[:2]:
            raise ValueError("Depth frame IDs must match the (env, history) image axes")
        if frame_ids.dtype != torch.long or frame_ids.device != depth.device:
            raise ValueError("Depth frame IDs must be int64 on the image device")
        if (frame_ids < 0).any():
            raise ValueError("Depth images must be captured before adding them to a rollout")
        flat_ids = frame_ids.flatten()
        unique_ids, inverse = torch.unique(flat_ids, sorted=True, return_inverse=True)
        first = torch.full_like(unique_ids, flat_ids.numel())
        first.scatter_reduce_(0, inverse, torch.arange(flat_ids.numel(), device=depth.device),
                              reduce="amin")
        if self._source_ids is None:
            missing = torch.ones_like(unique_ids, dtype=torch.bool)
        else:
            positions = torch.searchsorted(self._source_ids, unique_ids)
            missing = (positions == self._source_ids.numel()) | (
                self._source_ids[positions.clamp_max(self._source_ids.numel() - 1)] != unique_ids)
        new_ids = unique_ids[missing]
        if new_ids.numel():
            images = depth.flatten(0, 1)[first[missing]].detach()
            self._chunks.append(images)
            new_pool_ids = torch.arange(self.num_frames, self.num_frames + new_ids.numel(),
                                        device=depth.device)
            self.num_frames += new_ids.numel()
            if self._source_ids is None:
                self._source_ids, self._pool_ids = new_ids, new_pool_ids
            else:
                keys = torch.cat((self._source_ids, new_ids))
                order = keys.argsort()
                self._source_ids = keys[order]
                self._pool_ids = torch.cat((self._pool_ids, new_pool_ids))[order]
        return self._pool_ids[torch.searchsorted(self._source_ids, flat_ids)].view_as(frame_ids)

    def materialize(self):
        if self._tensor is None and self._chunks:
            self._tensor = self._chunks[0] if len(self._chunks) == 1 else torch.cat(self._chunks)
            self._chunks = []
        return self._tensor


class PIERolloutStorage(RolloutStorage):
    """Native transition tensors extended with full PIE actor trajectories.

    Time and environment are the leading axes of every side buffer. The critic
    is feedforward; only the actor state at the start of a rollout is retained.
    """
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.proprio_history = None
            self.depth_indices = None
            self.depth = None
            self.targets = None
            self.valid = None
            self.reset_mask = None
            self.successor = None
            self.successor_valid = None
            self.raw_rewards = None
            self.terminated = None
            self.truncated = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proprio_history = None
        self.depth_indices = None
        self.depth = None
        self.targets = {}
        self.valid = None
        self.reset_mask = None
        self.successor = None
        self.successor_valid = None
        self.raw_rewards = None
        self.terminated = None
        self.truncated = None
        self.initial_hidden = None
        self.depth_pool = DepthFramePool()

    def prepare_transition(self, observations, transition):
        # Snapshot before env.step() can overwrite observations or reset labels.
        transition.observations = observations["proprio"].detach().clone()
        transition.critic_observations = observations["critic"].detach().clone()
        transition.proprio_history = observations["proprio_history"].detach().clone()
        transition.targets = {name: value.detach().clone()
                              for name, value in observations["targets"].items()}
        transition.valid = torch.isfinite(observations["proprio"]).all(-1)
        if "depth_frame_ids" in observations:
            transition.depth_indices = self.depth_pool.add(
                observations["depth"], observations["depth_frame_ids"])
        else:
            transition.depth = observations["depth"].detach().clone()

    def _save_hidden_states(self, hidden_states):
        if self.step == 0:
            if hidden_states is None or hidden_states[0] is None:
                raise RuntimeError("PIE transitions require an actor GRU state")
            self.initial_hidden = hidden_states[0].detach().clone()

    def _store_tensor(self, buffer, value):
        if buffer is None:
            buffer = value.new_empty(self.num_transitions_per_env, *value.shape)
        buffer[self.step].copy_(value)
        return buffer

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        for name in ("proprio_history", "depth_indices", "depth", "valid", "reset_mask",
                     "successor", "successor_valid", "raw_rewards", "terminated", "truncated"):
            value = getattr(transition, name)
            if value is not None:
                setattr(self, name, self._store_tensor(getattr(self, name), value))
        for name, value in transition.targets.items():
            self.targets[name] = self._store_tensor(self.targets.get(name), value)
        super().add_transitions(transition)

    def observation(self, step, ids=None, restore_depth=False):
        observations = {"proprio": self.observations[step],
                        "critic": self.privileged_observations[step],
                        "proprio_history": self.proprio_history[step]}
        if self.depth_indices is not None:
            observations["depth_indices"] = self.depth_indices[step]
        else:
            observations["depth"] = self.depth[step]
        if ids is not None:
            observations = {name: value[ids] for name, value in observations.items()}
        if restore_depth and "depth_indices" in observations:
            observations["depth"] = self.depth_pool.materialize()[observations["depth_indices"]]
        return observations

    @staticmethod
    def trajectory_mini_batch_generator(batch_size, device, num_mini_batches, num_epochs):
        if num_mini_batches < 1 or num_epochs < 1 or batch_size < num_mini_batches:
            raise ValueError("PIE requires num_envs >= minibatches, both positive")
        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=device)
            yield from torch.tensor_split(indices, num_mini_batches)

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.step != self.num_transitions_per_env:
            raise RuntimeError("PIE training requires a complete rollout")
        depth_frames = self.depth_pool.materialize()
        for ids in self.trajectory_mini_batch_generator(
                self.num_envs, self.device, num_mini_batches, num_epochs):
            observations = {"proprio": self.observations[:, ids],
                            "critic": self.privileged_observations[:, ids],
                            "proprio_history": self.proprio_history[:, ids]}
            if self.depth_indices is not None:
                observations["depth_indices"] = self.depth_indices[:, ids]
            else:
                observations["depth"] = self.depth[:, ids]
            yield (observations, observations["critic"].flatten(0, 1),
                   self.actions[:, ids].flatten(0, 1), self.values[:, ids].flatten(0, 1),
                   self.advantages[:, ids].flatten(0, 1), self.returns[:, ids].flatten(0, 1),
                   self.actions_log_prob[:, ids].flatten(0, 1), self.mu[:, ids].flatten(0, 1),
                   self.sigma[:, ids].flatten(0, 1), (self.initial_hidden[ids], None),
                   self.reset_mask[:, ids],
                   {name: value[:, ids] for name, value in self.targets.items()},
                   self.successor[:, ids], self.valid[:, ids], self.successor_valid[:, ids],
                   depth_frames)

    def clear(self):
        super().clear()
        self.initial_hidden = None
        self.depth_pool = DepthFramePool()
