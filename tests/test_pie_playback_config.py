"""Only the native branch's schema3 configuration is accepted for playback."""
import pytest

from native_cpu_helpers import class_to_dict, load_native_classes
def restore_playback_config(*args):
    return load_native_classes().helpers.restore_playback_config(*args)


def test_native_checkpoint_restores_training_units_and_keeps_runtime_env_count():
    classes = load_native_classes()
    training = classes.config()
    training.env.num_envs = 4
    saved = classes.config()
    saved.control.action_scale = .3
    saved.commands.resampling_time = 5.
    saved.normalization.obs_scales.lin_vel = 3.
    saved.normalization.obs_scales.ang_vel = .5
    saved.normalization.clip_observations = 50.
    saved.rewards.only_positive_rewards = False
    saved.camera.pitch_degrees = 28.
    cfg, source = restore_playback_config(training, {
        "pie_checkpoint_version": 3, "environment_cfg": class_to_dict(saved)})
    assert cfg.env.num_envs == 4
    assert cfg.control.action_scale == .3
    assert cfg.commands.resampling_time == 5.
    assert cfg.normalization.obs_scales.lin_vel == 3.
    assert cfg.normalization.obs_scales.ang_vel == .5
    assert cfg.normalization.clip_observations == 50.
    assert not cfg.rewards.only_positive_rewards
    assert cfg.camera.pitch_degrees == 28.
    assert "version 3" in source
    assert training.control.action_scale == .25
    assert training.rewards.only_positive_rewards
    assert training.noise.add_noise and training.domain_rand.push_robots


def test_playback_disables_randomization_without_changing_saved_reward_scales():
    classes = load_native_classes()
    training = classes.config()
    snapshot = class_to_dict(training)
    cfg, _ = restore_playback_config(training, {
        "pie_checkpoint_version": 3, "environment_cfg": snapshot})
    assert not cfg.noise.add_noise
    assert not cfg.domain_rand.randomize_friction
    assert not cfg.domain_rand.randomize_base_mass
    assert not cfg.domain_rand.push_robots
    assert not cfg.terrain.curriculum
    assert not cfg.domain_rand.randomize_pie
    assert class_to_dict(cfg.rewards) == snapshot["rewards"]
    assert class_to_dict(training) == snapshot


@pytest.mark.parametrize("version", [None, 1, 2, 4])
def test_old_or_unknown_checkpoint_schema_is_rejected(version):
    classes = load_native_classes()
    with pytest.raises(ValueError, match="version-3"):
        restore_playback_config(classes.config(), {"pie_checkpoint_version": version})


def test_missing_environment_config_does_not_silently_use_current_defaults():
    classes = load_native_classes()
    with pytest.raises(ValueError, match="environment_cfg"):
        restore_playback_config(classes.config(), {"pie_checkpoint_version": 3})


def test_real_native_helper_serializes_config_fields_and_roundtrips_without_methods():
    classes = load_native_classes()
    cfg = classes.config()
    serialize = classes.helpers.class_to_dict
    snapshot = serialize(cfg)
    assert snapshot["camera"]["history"] == 2
    assert "pie" not in snapshot
    assert "validate" not in snapshot
    assert snapshot["control"]["decimation"] == 4
    assert snapshot["sim"]["physx"]["max_gpu_contact_pairs"] > 0
    changed = classes.config()
    changed.control.decimation = 99
    changed.camera.history = 99
    classes.helpers.update_class_from_dict(changed, snapshot)
    assert serialize(changed) == snapshot
    restored, _ = restore_playback_config(cfg, {
        "pie_checkpoint_version": 3, "environment_cfg": snapshot})
    assert restored.control.decimation == 4 and restored.camera.history == 2


def test_playback_of_parsed_runtime_cfg_accepts_recomputed_push_interval_and_seed():
    import numpy as np
    from types import SimpleNamespace
    classes = load_native_classes()
    parsed = classes.task.__new__(classes.task)
    parsed.cfg = classes.config()
    parsed.cfg.seed = np.int64(8)
    parsed.cfg.domain_rand.friction_range = np.asarray([.2, 1.2])
    parsed.sim_params = SimpleNamespace(dt=.005)
    parsed._parse_cfg(parsed.cfg)
    snapshot = classes.helpers.class_to_dict(parsed.cfg)
    assert type(snapshot["domain_rand"]["push_interval"]) is float
    assert type(snapshot["seed"]) is int
    original = classes.config()
    original.env.num_envs = 1
    restored, _ = restore_playback_config(original, {
        "pie_checkpoint_version": 3, "environment_cfg": snapshot})
    assert restored.env.num_envs == 1
    assert not restored.domain_rand.push_robots
    assert restored.domain_rand.friction_range == [.2, 1.2]
    # The interval is a duration-derived field, recalculated by the real parser.
    reparsed = classes.task.__new__(classes.task)
    reparsed.cfg, reparsed.sim_params = restored, SimpleNamespace(dt=.01)
    reparsed._parse_cfg(restored)
    assert restored.domain_rand.push_interval == 375.
