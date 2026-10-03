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
    """Original transition tensors and GAE plus PIE sequence side buffers.

    The actor is recurrent and the critic is not. Whole environment trajectories
    preserve all control-step GRU links, including resets inside a rollout.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = []
        self.initial_hidden = None
        self.depth_pool = DepthFramePool()

    def _save_hidden_states(self, hidden_states):
        # Parent recurrent storage assumes both actor and critic have memory.
        # PIE needs only the detached rollout-start actor state for full replay.
        if self.step == 0:
            if hidden_states is None or hidden_states[0] is None:
                raise RuntimeError("PIE transitions require an actor GRU state")
            actor = hidden_states[0]
            self.initial_hidden = actor.detach().clone()

    def add_transitions(self, transition):
        frame = transition.pie_frame
        super().add_transitions(transition)
        index = self.step - 1
        frame["obs"]["proprio"] = self.observations[index]
        frame["obs"]["critic"] = self.privileged_observations[index]
        for key, tensor in (("actions", self.actions), ("old_logp", self.actions_log_prob),
                            ("values", self.values), ("rewards", self.rewards),
                            ("old_mu", self.mu), ("old_sigma", self.sigma)):
            frame[key] = tensor[index] if key in ("actions", "old_mu", "old_sigma") else tensor[index, :, 0]
        self.frames.append(frame)

    def as_batch(self):
        if self.step != self.num_transitions_per_env:
            raise RuntimeError("PIE training requires a complete rollout")
        return {"frames": self.frames, "hidden": self.initial_hidden,
                "depth_frames": self.depth_pool.materialize(),
                "advantages": self.advantages[:, :, 0], "returns": self.returns[:, :, 0]}

    @staticmethod
    def observation(frame, ids=None, depth_frames=None, restore_depth=False):
        obs = {key: value if ids is None else value[ids]
               for key, value in frame["obs"].items()}
        if restore_depth and "depth_indices" in obs:
            if depth_frames is None:
                raise ValueError("Indexed observations require their rollout depth frame pool")
            obs["depth"] = depth_frames[obs["depth_indices"]]
        return obs

    @staticmethod
    def trajectory_mini_batch_generator(batch_size, device, num_mini_batches, num_epochs):
        if num_mini_batches < 1 or num_epochs < 1 or batch_size < num_mini_batches:
            raise ValueError("PIE requires num_envs >= minibatches, both positive")
        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=device)
            yield from torch.tensor_split(indices, num_mini_batches)

    def clear(self):
        super().clear()
        self.frames = []
        self.initial_hidden = None
        self.depth_pool = DepthFramePool()
