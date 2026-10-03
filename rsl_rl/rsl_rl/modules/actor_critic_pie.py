"""PIE estimator and asymmetric actor-critic. No simulator imports."""
from dataclasses import asdict, dataclass
from typing import Tuple
import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.checkpoint import checkpoint
from .actor_critic import ActorCritic

@dataclass
class ModelConfig:
    proprio_dim: int = 45
    proprio_history: int = 10
    depth_history: int = 2
    action_dim: int = 12
    heightmap_dim: int = 187
    critic_dim: int = 235
    token_dim: int = 128
    gru_dim: int = 128
    latent_dim: int = 16
    map_latent_dim: int = 32
    transformer_heads: int = 4
    transformer_layers: int = 1
    initial_std: float = 1.0
    activation: str = "elu"
    # Reproduction choices, with evidence/rationale in PIE_NETWORK.md.
    proprio_hidden_dims: Tuple[int, ...] = (512, 256)
    cnn_hidden_channels: Tuple[int, int] = (32, 64)
    cnn_kernel_sizes: Tuple[int, int, int] = (5, 3, 3)
    cnn_strides: Tuple[int, int, int] = (2, 2, 2)
    cnn_paddings: Tuple[int, int, int] = (2, 1, 1)
    visual_grid: Tuple[int, int] = (4, 4)
    transformer_ffn_multiplier: int = 2
    transformer_dropout: float = 0.0
    actor_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    successor_hidden_dims: Tuple[int, ...] = (128, 128)
    height_decoder_hidden_dims: Tuple[int, ...] = (128, 128)

    def __post_init__(self):
        # YAML/JSON may deserialize tuples as lists; keep checkpoint comparison stable.
        sequence_fields = ("proprio_hidden_dims", "cnn_hidden_channels", "cnn_kernel_sizes",
                           "cnn_strides", "cnn_paddings", "visual_grid", "actor_hidden_dims",
                           "critic_hidden_dims", "successor_hidden_dims", "height_decoder_hidden_dims")
        for name in sequence_fields:
            setattr(self, name, tuple(getattr(self, name)))
        positive = (self.proprio_dim, self.proprio_history, self.depth_history, self.action_dim,
                    self.heightmap_dim, self.critic_dim, self.token_dim, self.gru_dim,
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
        if self.initial_std <= 0:
            raise ValueError("initial_std must be positive")

    def validate_observation(self, obs, num_actions):
        """Check the environment schema once before training or playback."""
        observed = {
            "proprio_dim": obs["proprio"].shape[-1],
            "proprio_history": obs["proprio_history"].shape[1],
            "depth_history": obs["depth"].shape[1],
            "heightmap_dim": obs["targets"]["heightmap"].shape[-1],
            "critic_dim": obs["critic"].shape[-1],
            "action_dim": num_actions,
        }
        for name, actual in observed.items():
            if getattr(self, name) != actual:
                raise ValueError("PIE model/environment mismatch for {}: configured {}, observed {}".format(
                    name, getattr(self, name), actual))
        return self

    @property
    def token_count(self):
        return 1 + self.visual_grid[0] * self.visual_grid[1]

def mlp(input_dim, widths, output_dim):
    layers = []
    for width in widths:
        layers += [nn.Linear(input_dim, width), nn.ELU()]
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))

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
    def __init__(self, num_actor_obs=None, num_critic_obs=None, num_actions=None,
                 model_config=None, init_noise_std=None, actor_hidden_dims=None,
                 critic_hidden_dims=None, activation=None):
        # Accept the native v1.0.2 runner signature and the configuration helper.
        if isinstance(num_actor_obs, ModelConfig):
            cfg = num_actor_obs
        else:
            cfg = ModelConfig(**(model_config or {}))
            for name, actual in (("proprio_dim", num_actor_obs),
                                 ("critic_dim", num_critic_obs), ("action_dim", num_actions)):
                if actual is not None and getattr(cfg, name) != actual:
                    raise ValueError("PIE model/environment mismatch for " + name)
        # Native policy fields are authoritative for actor/critic/exploration.
        resolved = asdict(cfg)
        for name, value in (("initial_std", init_noise_std), ("actor_hidden_dims", actor_hidden_dims),
                            ("critic_hidden_dims", critic_hidden_dims), ("activation", activation)):
            if value is not None:
                resolved[name] = value
        cfg = ModelConfig(**resolved)
        estimate_dim = 3 + 4 + cfg.map_latent_dim + cfg.latent_dim
        super().__init__(
            cfg.proprio_dim + estimate_dim, cfg.critic_dim, cfg.action_dim,
            actor_hidden_dims=list(cfg.actor_hidden_dims), critic_hidden_dims=list(cfg.critic_hidden_dims),
            activation=cfg.activation, init_noise_std=cfg.initial_std)
        self.cfg = cfg
        self.proprio_encoder = mlp(cfg.proprio_dim * cfg.proprio_history,
                                  cfg.proprio_hidden_dims, cfg.token_dim)
        channels = (cfg.depth_history,) + cfg.cnn_hidden_channels + (cfg.token_dim,)
        convolution = []
        for index in range(3):
            convolution.extend((
                nn.Conv2d(channels[index], channels[index + 1], cfg.cnn_kernel_sizes[index],
                          stride=cfg.cnn_strides[index], padding=cfg.cnn_paddings[index]),
                nn.ELU()))
        self.depth_encoder = nn.Sequential(*convolution, nn.AdaptiveAvgPool2d(cfg.visual_grid))
        self.position = nn.Parameter(torch.zeros(1, cfg.token_count, cfg.token_dim))
        layer = nn.TransformerEncoderLayer(
            cfg.token_dim, cfg.transformer_heads, cfg.token_dim * cfg.transformer_ffn_multiplier,
            dropout=cfg.transformer_dropout, batch_first=True, activation="gelu",
            layer_norm_eps=1e-5, norm_first=False)
        self.transformer = nn.TransformerEncoder(layer, cfg.transformer_layers)
        self.gru = nn.GRUCell(cfg.token_count * cfg.token_dim, cfg.gru_dim)
        self.velocity_head = nn.Linear(cfg.gru_dim, 3)
        self.clearance_head = nn.Linear(cfg.gru_dim, 4)
        self.map_head = nn.Linear(cfg.gru_dim, cfg.map_latent_dim)
        self.mu_head = nn.Linear(cfg.gru_dim, cfg.latent_dim)
        self.logvar_head = nn.Linear(cfg.gru_dim, cfg.latent_dim)
        self.successor_decoder = mlp(estimate_dim, cfg.successor_hidden_dims, cfg.proprio_dim)
        self.height_decoder = mlp(cfg.map_latent_dim, cfg.height_decoder_hidden_dims, cfg.heightmap_dim)
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

    def initial_state(self, batch_size, device=None):
        return torch.zeros(batch_size, self.cfg.gru_dim,
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
        prop = self.proprio_encoder(obs["proprio_history"].flatten(1)).unsqueeze(1)
        visual = self.encode_depth(obs["depth"]) if visual_features is None else visual_features
        depth = visual.flatten(2).transpose(1, 2)
        tokens = self.transformer(torch.cat((prop, depth), dim=1) + self.position)
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

    def value(self, obs):
        return self.evaluate(obs).squeeze(-1)

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
                self.value(obs), hidden, estimates)

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
