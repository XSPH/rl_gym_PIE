"""PIE estimator and asymmetric actor-critic. No simulator imports."""
from dataclasses import dataclass
from typing import Tuple
import torch
from torch import nn
from torch.distributions import Normal
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
    initial_std: float = 0.5
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

class PIEActorCritic(ActorCritic):
    """The actor uses estimates, never privileged targets.
    
    PPO uses posterior means for repeatable action probabilities. Only the
    successor decoder samples the VAE. This is an explicit reproduction choice,
    because the paper does not specify how latent samples are replayed in PPO.
    Transformer dropout is zero for the same reason. GRU state is passed in and
    out, so it cannot leak between environments or survive episode boundaries.
    """
    is_recurrent = True
    def __init__(self, num_actor_obs=None, num_critic_obs=None, num_actions=None,
                 model_config=None):
        # Accept the native v1.0.2 runner signature and the configuration helper.
        if isinstance(num_actor_obs, ModelConfig):
            cfg = num_actor_obs
        else:
            cfg = ModelConfig(**(model_config or {}))
            for name, actual in (("proprio_dim", num_actor_obs),
                                 ("critic_dim", num_critic_obs), ("action_dim", num_actions)):
                if actual is not None and getattr(cfg, name) != actual:
                    raise ValueError("PIE model/environment mismatch for " + name)
        if cfg.initial_std <= 0:
            raise ValueError("initial_std must be positive")
        estimate_dim = 3 + 4 + cfg.map_latent_dim + cfg.latent_dim
        super().__init__(
            cfg.proprio_dim + estimate_dim, cfg.critic_dim, cfg.action_dim,
            actor_hidden_dims=list(cfg.actor_hidden_dims), critic_hidden_dims=list(cfg.critic_hidden_dims),
            activation="elu", init_noise_std=cfg.initial_std)
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

    @property
    def log_std(self):
        """Compatibility view; the learnable parameter is v1.0.2's native std."""
        return self.std.clamp(min=0.006737946999085467, max=7.38905609893065).log()

    def load_state_dict(self, state_dict, strict=True):
        # Previous minimal checkpoints wrapped the 2.2.4 ActorCritic in policy.
        # Only weights are migrated; old optimizer moments cannot be reused.
        state_dict = dict(state_dict)
        for key in list(state_dict):
            if key.startswith(("policy.actor.", "policy.critic.")):
                state_dict[key[len("policy."):]] = state_dict.pop(key)
        if "policy.log_std" in state_dict:
            state_dict["std"] = state_dict.pop("policy.log_std").exp()
        return super().load_state_dict(state_dict, strict=strict)

    def initial_state(self, batch_size, device=None):
        return torch.zeros(batch_size, self.cfg.gru_dim,
                           device=device or self.std.device)

    def encode(self, obs, hidden, reset_mask=None):
        if reset_mask is not None:
            hidden = hidden * (~reset_mask.bool()).unsqueeze(-1)
        prop = self.proprio_encoder(obs["proprio_history"].flatten(1)).unsqueeze(1)
        depth = self.depth_encoder(obs["depth"]).flatten(2).transpose(1, 2)
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

    def policy_distribution(self, obs, hidden, reset_mask=None):
        estimates, hidden = self.encode(obs, hidden, reset_mask)
        mean = self.actor(torch.cat((obs["proprio"], self.features(estimates)), dim=-1))
        # distribution is an attribute in v1.0.2; keep its probability API.
        scale = self.std.clamp(min=0.006737946999085467, max=7.38905609893065)
        self.distribution = Normal(mean, scale, validate_args=False)
        return self.distribution, hidden, estimates

    def value(self, obs):
        return self.critic(obs["critic"]).squeeze(-1)

    def act(self, obs, hidden, reset_mask=None, deterministic=False):
        dist, hidden, estimates = self.policy_distribution(obs, hidden, reset_mask)
        actions = dist.mean if deterministic else dist.sample()
        return actions, self.get_actions_log_prob(actions), self.value(obs), hidden

    def evaluate(self, obs, hidden, actions, reset_mask=None):
        dist, hidden, estimates = self.policy_distribution(obs, hidden, reset_mask)
        return (self.get_actions_log_prob(actions), self.entropy,
                self.value(obs), hidden, estimates)

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
        raise RuntimeError("PIE requires policy_distribution(obs_dict, hidden, reset_mask)")

    def act_inference(self, observations, hidden, reset_mask=None):
        distribution, hidden, _ = self.policy_distribution(observations, hidden, reset_mask)
        return distribution.mean, hidden
