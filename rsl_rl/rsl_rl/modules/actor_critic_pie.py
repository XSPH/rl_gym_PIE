"""PIE estimator and asymmetric actor-critic. No simulator imports."""
import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.checkpoint import checkpoint
from .actor_critic import ActorCritic


def mlp(input_dim, widths, output_dim):
    layers = []
    for width in widths:
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ELU())
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class PIEDepthFeatureCache:
    """Reuse deterministic CNN outputs while retaining every use's gradient.

    A training cache belongs to one logical optimizer update only. Updating
    rows is out-of-place: prior recurrent steps keep their original features,
    and repeated uses share the CNN graph rather than detached rollout latents.
    Observations without image IDs are encoded directly.
    """
    def __init__(self):
        self.frame_ids = None
        self.features = None
        self.encoded_stacks = 0

    def get(self, model, obs, depth_frames=None):
        frame_ids = obs.get("depth_indices", obs.get("depth_frame_ids"))
        if frame_ids is None:
            self.encoded_stacks += obs["depth"].shape[0]
            return model.encode_depth(obs["depth"])
        if self.frame_ids is None:
            changed = torch.arange(frame_ids.shape[0], device=frame_ids.device)
        else:
            if frame_ids.shape != self.frame_ids.shape:
                raise ValueError("A depth cache cannot change its environment/history axes")
            changed = torch.nonzero((frame_ids != self.frame_ids).any(-1)).flatten()
        if changed.numel():
            if "depth_indices" in obs:
                if depth_frames is None:
                    raise ValueError("Indexed images require a depth frame pool")
                depth = depth_frames[frame_ids[changed]]
            else:
                depth = obs["depth"][changed]
            encoded = model.encode_depth(depth)
            if self.features is None:
                self.features = encoded
            else:
                self.features = self.features.index_copy(0, changed, encoded)
            self.frame_ids = frame_ids.detach().clone()
            self.encoded_stacks += changed.numel()
        return self.features


class PIEActorCritic(ActorCritic):
    """The actor uses estimates, never privileged targets.

    PPO uses posterior means for repeatable action probabilities. Only the
    successor decoder samples the VAE. This is an explicit reproduction choice,
    because the paper does not specify how latent samples are replayed in PPO.
    Transformer dropout is zero for the same reason. The native recurrent API
    owns rollout state; explicit states are used only for minibatch replay.
    """
    is_recurrent = True

    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 proprio_history=10,
                 depth_history=2,
                 heightmap_dim=187,
                 token_dim=128,
                 gru_dim=128,
                 latent_dim=16,
                 map_latent_dim=32,
                 transformer_heads=4,
                 transformer_layers=1,
                 init_noise_std=1.0,
                 activation='elu',
                 proprio_hidden_dims=(512, 256),
                 cnn_hidden_channels=(32, 64),
                 cnn_kernel_sizes=(5, 3, 3),
                 cnn_strides=(2, 2, 2),
                 cnn_paddings=(2, 1, 1),
                 visual_grid=(4, 4),
                 transformer_ffn_multiplier=2,
                 transformer_dropout=0.0,
                 actor_hidden_dims=(512, 256, 128),
                 critic_hidden_dims=(512, 256, 128),
                 successor_hidden_dims=(128, 128),
                 height_decoder_hidden_dims=(128, 128)):
        self.num_actor_obs = num_actor_obs
        self.proprio_history = proprio_history
        self.depth_history = depth_history
        self.num_actions = num_actions
        self.heightmap_dim = heightmap_dim
        self.num_critic_obs = num_critic_obs
        self.token_dim = token_dim
        self.gru_dim = gru_dim
        self.latent_dim = latent_dim
        self.map_latent_dim = map_latent_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.init_noise_std = init_noise_std
        self.activation = activation
        self.proprio_hidden_dims = tuple(proprio_hidden_dims)
        self.cnn_hidden_channels = tuple(cnn_hidden_channels)
        self.cnn_kernel_sizes = tuple(cnn_kernel_sizes)
        self.cnn_strides = tuple(cnn_strides)
        self.cnn_paddings = tuple(cnn_paddings)
        self.visual_grid = tuple(visual_grid)
        self.transformer_ffn_multiplier = transformer_ffn_multiplier
        self.transformer_dropout = transformer_dropout
        self.actor_hidden_dims = tuple(actor_hidden_dims)
        self.critic_hidden_dims = tuple(critic_hidden_dims)
        self.successor_hidden_dims = tuple(successor_hidden_dims)
        self.height_decoder_hidden_dims = tuple(height_decoder_hidden_dims)
        self._validate_dimensions()

        # Policy and value networks
        estimate_dim = 3 + 4 + self.map_latent_dim + self.latent_dim
        super().__init__(
            self.num_actor_obs + estimate_dim, self.num_critic_obs, self.num_actions,
            actor_hidden_dims=list(self.actor_hidden_dims), critic_hidden_dims=list(self.critic_hidden_dims),
            activation=self.activation, init_noise_std=self.init_noise_std)
        # Proprioceptive encoder
        self.proprio_encoder = mlp(self.num_actor_obs * self.proprio_history,
                                  self.proprio_hidden_dims, self.token_dim)

        # Depth encoder
        channels = (self.depth_history,) + self.cnn_hidden_channels + (self.token_dim,)
        convolution = []
        for index in range(3):
            convolution.extend((
                nn.Conv2d(channels[index], channels[index + 1], self.cnn_kernel_sizes[index],
                          stride=self.cnn_strides[index], padding=self.cnn_paddings[index]),
                nn.ELU()))
        self.depth_encoder = nn.Sequential(*convolution, nn.AdaptiveAvgPool2d(self.visual_grid))

        # Spatial and temporal fusion
        self.position = nn.Parameter(torch.zeros(1, self.token_count, self.token_dim))
        layer = nn.TransformerEncoderLayer(
            self.token_dim, self.transformer_heads, self.token_dim * self.transformer_ffn_multiplier,
            dropout=self.transformer_dropout, batch_first=True, activation="gelu",
            layer_norm_eps=1e-5, norm_first=False)
        self.transformer = nn.TransformerEncoder(layer, self.transformer_layers)
        self.gru = nn.GRUCell(self.token_count * self.token_dim, self.gru_dim)

        # Auxiliary prediction heads and decoders
        self.velocity_head = nn.Linear(self.gru_dim, 3)
        self.clearance_head = nn.Linear(self.gru_dim, 4)
        self.map_head = nn.Linear(self.gru_dim, self.map_latent_dim)
        self.mu_head = nn.Linear(self.gru_dim, self.latent_dim)
        self.logvar_head = nn.Linear(self.gru_dim, self.latent_dim)
        self.successor_decoder = mlp(estimate_dim, self.successor_hidden_dims, self.num_actor_obs)
        self.height_decoder = mlp(self.map_latent_dim, self.height_decoder_hidden_dims, self.heightmap_dim)
        for label, module in (
            ("Proprio Encoder MLP", self.proprio_encoder),
            ("Depth Encoder CNN", self.depth_encoder),
            ("Fusion Transformer Encoder", self.transformer),
            ("Temporal Fusion GRU", self.gru),
            ("Velocity Head", self.velocity_head),
            ("Foot Clearance Head", self.clearance_head),
            ("Height Map Latent Head", self.map_head),
            ("VAE Mean Head", self.mu_head),
            ("VAE Log Variance Head", self.logvar_head),
            ("Proprio Decoder MLP", self.successor_decoder),
            ("Height Map Decoder MLP", self.height_decoder),
        ):
            print("{}: {}".format(label, module))
        print("Positional Embedding: {}".format(tuple(self.position.shape)))
        self._hidden = None
        self._visual_cache = PIEDepthFeatureCache()

    def _validate_dimensions(self):
        sequence_fields = (
            "proprio_hidden_dims", "cnn_hidden_channels", "cnn_kernel_sizes", "cnn_strides",
            "cnn_paddings", "visual_grid", "actor_hidden_dims", "critic_hidden_dims",
            "successor_hidden_dims", "height_decoder_hidden_dims")
        positive = (self.num_actor_obs, self.proprio_history, self.depth_history, self.num_actions,
                    self.heightmap_dim, self.num_critic_obs, self.token_dim, self.gru_dim,
                    self.latent_dim, self.map_latent_dim, self.transformer_heads,
                    self.transformer_layers, self.transformer_ffn_multiplier)
        if any(value <= 0 for value in positive):
            raise ValueError("PIE dimensions and layer counts must be positive")
        if self.token_dim % self.transformer_heads:
            raise ValueError("token_dim must be divisible by transformer_heads")
        for name in sequence_fields:
            values = getattr(self, name)
            if not values or any(v < (0 if name == "cnn_paddings" else 1) for v in values):
                raise ValueError("Invalid network dimensions in " + name)
        expected_lengths = {"cnn_hidden_channels": 2, "cnn_kernel_sizes": 3,
                            "cnn_strides": 3, "cnn_paddings": 3, "visual_grid": 2}
        if any(len(getattr(self, name)) != length for name, length in expected_lengths.items()):
            raise ValueError("PIE uses three CNN layers and a two-dimensional token grid")
        if self.transformer_dropout != 0.0:
            raise ValueError("PIE PPO replay requires transformer_dropout=0")
        if self.init_noise_std <= 0:
            raise ValueError("init_noise_std must be positive")

    def validate_observation(self, obs, num_actions):
        """Check the environment schema once before training or playback."""
        observed = {
            "num_actor_obs": obs["proprio"].shape[-1],
            "proprio_history": obs["proprio_history"].shape[1],
            "depth_history": obs["depth"].shape[1],
            "heightmap_dim": obs["targets"]["heightmap"].shape[-1],
            "num_critic_obs": obs["critic"].shape[-1],
            "num_actions": num_actions,
        }
        for name, actual in observed.items():
            if getattr(self, name) != actual:
                raise ValueError("PIE model/environment mismatch for {}: configured {}, observed {}".format(
                    name, getattr(self, name), actual))
        return self

    @property
    def token_count(self):
        return 1 + self.visual_grid[0] * self.visual_grid[1]

    def get_model_config(self):
        """Return the effective architecture in the version-4 checkpoint schema."""
        return {
            "proprio_dim": self.num_actor_obs,
            "proprio_history": self.proprio_history,
            "depth_history": self.depth_history,
            "action_dim": self.num_actions,
            "heightmap_dim": self.heightmap_dim,
            "critic_dim": self.num_critic_obs,
            "token_dim": self.token_dim,
            "gru_dim": self.gru_dim,
            "latent_dim": self.latent_dim,
            "map_latent_dim": self.map_latent_dim,
            "transformer_heads": self.transformer_heads,
            "transformer_layers": self.transformer_layers,
            "initial_std": self.init_noise_std,
            "activation": self.activation,
            "proprio_hidden_dims": self.proprio_hidden_dims,
            "cnn_hidden_channels": self.cnn_hidden_channels,
            "cnn_kernel_sizes": self.cnn_kernel_sizes,
            "cnn_strides": self.cnn_strides,
            "cnn_paddings": self.cnn_paddings,
            "visual_grid": self.visual_grid,
            "transformer_ffn_multiplier": self.transformer_ffn_multiplier,
            "transformer_dropout": self.transformer_dropout,
            "actor_hidden_dims": self.actor_hidden_dims,
            "critic_hidden_dims": self.critic_hidden_dims,
            "successor_hidden_dims": self.successor_hidden_dims,
            "height_decoder_hidden_dims": self.height_decoder_hidden_dims,
        }

    def begin_rollout(self, num_envs):
        if self._hidden is None:
            self._hidden = self.initial_state(num_envs)
        self._visual_cache = PIEDepthFeatureCache()

    def set_hidden_states(self, hidden_states):
        self._hidden = hidden_states.detach()
        self._visual_cache = PIEDepthFeatureCache()

    def initial_state(self, batch_size, device=None):
        return torch.zeros(batch_size, self.gru_dim,
                           device=device or self.std.device, dtype=self.std.dtype)

    def encode_depth(self, depth):
        # A trajectory minibatch retains the complete recurrent graph. Recompute
        # newly encoded image stacks during backward; unchanged stacks are
        # reused by PIEDepthFeatureCache without another CNN call. Non-reentrant
        # checkpointing trains CNN weights even when camera inputs have no grad.
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.depth_encoder, depth, use_reentrant=False)
        else:
            return self.depth_encoder(depth)

    def encode(self, obs, hidden, reset_mask=None, visual_features=None):
        if reset_mask is not None:
            hidden = hidden * (~reset_mask.bool()).unsqueeze(-1)
        proprio_features = self.proprio_encoder(obs["proprio_history"].flatten(1)).unsqueeze(1)
        visual = self.encode_depth(obs["depth"]) if visual_features is None else visual_features
        depth_tokens = visual.flatten(2).transpose(1, 2)
        tokens = self.transformer(torch.cat((proprio_features, depth_tokens), dim=1) + self.position)
        hidden = self.gru(tokens.flatten(1), hidden)
        estimates = {
            "velocity": self.velocity_head(hidden),
            "foot_clearance": self.clearance_head(hidden),
            "map_latent": self.map_head(hidden),
            "mu": self.mu_head(hidden),
            "logvar": self.logvar_head(hidden).clamp(-10.0, 5.0)}
        return estimates, hidden

    @staticmethod
    def features(estimates, z=None):
        return torch.cat((estimates["velocity"], estimates["foot_clearance"],
                          estimates["map_latent"], estimates["mu"] if z is None else z), dim=-1)

    def policy_distribution(self, obs, hidden, reset_mask=None, visual_features=None):
        estimates, hidden = self.encode(obs, hidden, reset_mask, visual_features)
        mean = self.actor(torch.cat((obs["proprio"], self.features(estimates)), dim=-1))
        # distribution is an attribute in v1.0.2; keep its probability API.
        scale = self.std.clamp_min(torch.finfo(self.std.dtype).eps)
        self.distribution = Normal(mean, scale, validate_args=False)
        return self.distribution, hidden, estimates

    def evaluate(self, critic_observations, **kwargs):
        """Native RSL value interface; the critic has no recurrent memory."""
        if isinstance(critic_observations, dict):
            critic_observations = critic_observations["critic"]
        return self.critic(critic_observations)

    def _rollout_distribution(self, observations, masks=None, hidden_states=None):
        if hidden_states is None:
            if self._hidden is None or self._hidden.shape[0] != observations["proprio"].shape[0]:
                self._hidden = self.initial_state(observations["proprio"].shape[0])
                self._visual_cache = PIEDepthFeatureCache()
            hidden = self._hidden
        else:
            hidden = hidden_states.squeeze(0) if hidden_states.ndim == 3 else hidden_states
        reset_mask = None if masks is None else ~masks.bool().flatten()
        visual = self._visual_cache.get(self, observations)
        distribution, next_hidden, _ = self.policy_distribution(
            observations, hidden, reset_mask, visual_features=visual)
        if hidden_states is None:
            self._hidden = next_hidden
        return distribution

    def act(self, observations, masks=None, hidden_states=None):
        """Sample actions through the original ActorCritic interface."""
        return self._rollout_distribution(observations, masks, hidden_states).sample()

    def evaluate_actions(self, obs, hidden, actions, reset_mask=None, visual_features=None):
        """Explicit recurrent replay without replacing the live rollout state."""
        _, hidden, estimates = self.policy_distribution(obs, hidden, reset_mask, visual_features)
        return (self.get_actions_log_prob(actions), self.entropy,
                self.evaluate(obs).squeeze(-1), hidden, estimates)

    def get_hidden_states(self):
        return self._hidden, None

    def reset(self, dones=None):
        if dones is None:
            self._hidden = None
            self._visual_cache = PIEDepthFeatureCache()
        elif self._hidden is not None:
            self._hidden = self._hidden * (~dones.bool().flatten()).unsqueeze(-1)

    def auxiliary_losses(self, estimates, targets, successor, valid=None,
                         successor_valid=None):
        z = estimates["mu"] + torch.randn_like(estimates["mu"]) * (
            0.5 * estimates["logvar"]).exp()
        next_prediction = self.successor_decoder(self.features(estimates, z))
        map_prediction = self.height_decoder(estimates["map_latent"])
        if valid is None:
            valid = torch.ones(successor.shape[0], dtype=torch.bool, device=successor.device)
        if successor_valid is None:
            successor_valid = valid
        def masked_mean(loss, mask):
            weights = mask.to(successor.dtype)
            return (loss * weights).sum() / weights.sum().clamp_min(1.0)
        def mse(prediction, truth, mask=valid):
            # Each label has its own validity. An invalid successor must not
            # discard valid current-state velocity/map/clearance supervision.
            mask = mask & torch.isfinite(truth).all(-1)
            # torch.where prevents NaN labels in invalid entries poisoning a batch.
            safe_truth = torch.where(mask[:, None], truth, prediction.detach())
            return masked_mean((prediction - safe_truth).square().mean(-1), mask)
        kl = 0.5 * (estimates["mu"].square() + estimates["logvar"].exp()
                    - estimates["logvar"] - 1).sum(-1)
        return {
            "velocity": mse(estimates["velocity"], targets["velocity"]),
            "foot_clearance": mse(estimates["foot_clearance"], targets["foot_clearance"]),
            "heightmap": mse(map_prediction, targets["heightmap"]),
            "successor": mse(next_prediction, successor, valid & successor_valid),
            "kl": masked_mean(kl, valid)}

    def update_distribution(self, observations):
        self._rollout_distribution(observations)

    def act_inference(self, observations):
        return self._rollout_distribution(observations).mean
