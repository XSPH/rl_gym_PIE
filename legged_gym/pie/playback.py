"""Restore only the native-training branch's checkpoint configuration."""
from copy import deepcopy
from pathlib import Path


def _restore_fields(destination, values, path=""):
    for name, value in values.items():
        field = path + name
        # LeggedRobot._parse_cfg derives this after restoring push_interval_s.
        if field == "domain_rand.push_interval":
            continue
        if name.startswith("_") or not hasattr(destination, name):
            raise ValueError("Unknown checkpoint configuration field: " + name)
        current = getattr(destination, name)
        if isinstance(value, dict) and hasattr(current, "__dict__"):
            _restore_fields(current, value, field + ".")
        else:
            setattr(destination, name, deepcopy(value))


def restore_playback_config(env_cfg, checkpoint):
    if checkpoint.get("pie_checkpoint_version") != 3:
        raise ValueError("This branch requires a native PIE version-3 checkpoint; "
                         "old PIE models are not supported")
    environment = checkpoint.get("environment_cfg")
    if not isinstance(environment, dict):
        raise ValueError("Native PIE checkpoint is missing environment_cfg")
    cfg = deepcopy(env_cfg)
    runtime_count = cfg.env.num_envs
    # TaskRegistry normally supplies this; accept the saved seed when this
    # helper is called with an otherwise fresh task configuration as well.
    if "seed" in environment and not hasattr(cfg, "seed"):
        cfg.seed = environment["seed"]
    _restore_fields(cfg, environment)
    cfg.env.num_envs = runtime_count
    # A moved checkout uses its own bundled asset. Explicit robot URDFs are
    # still validated by the environment constructor.
    if not Path(cfg.asset.file).is_file():
        cfg.asset.file = env_cfg.asset.file
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    cfg.terrain.curriculum = False
    cfg.pie.randomization.enabled = False
    cfg.pie.observation_noise = False
    return cfg, "native PIE checkpoint configuration (version 3)"
