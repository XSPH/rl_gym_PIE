"""PIE camera semantics and version-4 checkpoint configuration boundaries."""
from copy import deepcopy


def depth_input_mode(camera_cfg):
    """Missing fields in existing version-4 checkpoints mean real depth."""
    if isinstance(camera_cfg, dict):
        mode = camera_cfg.get('input_mode', 'depth')
    else:
        mode = getattr(camera_cfg, 'input_mode', 'depth')
    if mode not in ('depth', 'zero'):
        raise ValueError("camera.input_mode must be 'depth' or 'zero'; got {!r}".format(mode))
    return mode


# Defaults of the published version-4 checkpoint schema.
V4_MODEL_DEFAULTS = {
    "proprio_dim": 45,
    "proprio_history": 10,
    "depth_history": 2,
    "action_dim": 12,
    "heightmap_dim": 187,
    "critic_dim": 235,
    "token_dim": 128,
    "gru_dim": 128,
    "latent_dim": 16,
    "map_latent_dim": 32,
    "transformer_heads": 4,
    "transformer_layers": 1,
    "initial_std": 1.0,
    "activation": 'elu',
    "proprio_hidden_dims": (512, 256),
    "cnn_hidden_channels": (32, 64),
    "cnn_kernel_sizes": (5, 3, 3),
    "cnn_strides": (2, 2, 2),
    "cnn_paddings": (2, 1, 1),
    "visual_grid": (4, 4),
    "transformer_ffn_multiplier": 2,
    "transformer_dropout": 0.0,
    "actor_hidden_dims": (512, 256, 128),
    "critic_hidden_dims": (512, 256, 128),
    "successor_hidden_dims": (128, 128),
    "height_decoder_hidden_dims": (128, 128),
}


_MODEL_NAMES = {
    "proprio_dim": "num_actor_obs",
    "critic_dim": "num_critic_obs",
    "action_dim": "num_actions",
    "initial_std": "init_noise_std",
}


def normalize_model_config(config):
    """Normalize a v4 snapshot without constructing a model or consuming RNG."""
    unknown = set(config) - set(V4_MODEL_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown checkpoint model fields: {sorted(unknown)}")
    result = dict(V4_MODEL_DEFAULTS)
    result.update(config)
    for name, default in V4_MODEL_DEFAULTS.items():
        if isinstance(default, tuple):
            result[name] = tuple(result[name])
    return result


def normalize_train_config(config):
    """Accept legacy v4 policy.model_config only at the runner/play boundary."""
    result = deepcopy(config)
    policy = result["policy"]
    legacy = policy.pop("model_config", None)
    if legacy is not None:
        expanded = normalize_model_config(legacy)
        expanded = {_MODEL_NAMES.get(name, name): value for name, value in expanded.items()}
        expanded.update(policy)
        result["policy"] = expanded
    return result


def checkpoint_train_config(config, model_config):
    """Keep the v4 on-disk format readable by existing training/playback code."""
    result = deepcopy(config)
    model_config = dict(model_config)
    policy = {}
    for name in ("initial_std", "actor_hidden_dims", "critic_hidden_dims", "activation"):
        policy[_MODEL_NAMES.get(name, name)] = model_config.pop(name)
    policy["model_config"] = model_config
    result["policy"] = policy
    return result
