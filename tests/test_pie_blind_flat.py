"""CPU checks of the blind experiment through actual task and RSL methods."""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from native_cpu_helpers import load_native_classes
from native_rsl_helpers import TensorEnvironment, model_config, train_config
from rsl_rl.modules.actor_critic_pie import PIEActorCritic
from rsl_rl.runners.on_policy_runner_pie import PIEOnPolicyRunner
from test_pie_observation_and_push import sensor


@pytest.fixture(scope='module', autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def blind_sensor(sensor):
    sensor.cfg.camera.input_mode = 'zero'
    sensor._init_pie_buffers()
    sensor._reset_pie_sensors(torch.arange(sensor.num_envs))
    sensor.compute_observations()
    return sensor


def assert_blind_inputs(task):
    obs = task.get_pie_observations()
    assert obs['depth'].shape == (task.num_envs, 2, task.cfg.camera.height, task.cfg.camera.width)
    for value in (task.depth_queue, task.depth_history, obs['depth']):
        assert torch.isfinite(value).all()
        assert torch.count_nonzero(value) == 0
    assert torch.count_nonzero(task.camera.depth) > 0
    return obs


def test_blind_flat_defaults_keep_training_scale_and_all_flat_geometry():
    classes = load_native_classes()
    cfg, training = classes.config(), classes.train_config()
    assert cfg.camera.input_mode == 'zero' and not cfg.domain_rand.randomize_camera
    assert cfg.camera.noise_std == cfg.camera.salt_pepper_probability == 0
    assert cfg.domain_rand.randomize_pie and cfg.noise.add_noise
    assert cfg.domain_rand.randomize_friction and cfg.domain_rand.randomize_base_mass
    assert cfg.domain_rand.push_robots
    assert training.runner.experiment_name == 'lite3_pie_blind_flat'
    assert not training.runner.resume
    assert training.algorithm.estimation_weight == training.algorithm.kl_weight == 1.0
    assert (cfg.env.num_envs, training.runner.num_steps_per_env,
            training.algorithm.num_learning_epochs, training.algorithm.num_mini_batches,
            training.runner.max_iterations, training.runner.save_interval) == (4096, 24, 5, 4, 15000, 500)
    terrain = classes.terrain(cfg.terrain, cfg.env.num_envs, seed=4)
    assert terrain.atlas.kinds == ['flat'] * 20
    assert terrain.env_origins.shape == (10, 20, 3)
    assert np.count_nonzero(terrain.height_field_raw) == 0
    assert np.count_nonzero(terrain.env_origins[..., 2]) == 0
    task = classes.task.__new__(classes.task)
    task.cfg, task.device, task.num_envs = cfg, 'cpu', cfg.env.num_envs
    task.terrain_sampler = classes.sampler(terrain.atlas, 'cpu')
    task._get_env_origins()
    assert torch.count_nonzero(task.terrain_levels) == 0
    assert not cfg.terrain.curriculum


@pytest.mark.parametrize('mode', ['', 'blind', 'DEPTH', None, 0])
def test_invalid_camera_input_mode_fails_before_simulation(mode):
    classes = load_native_classes()
    cfg = classes.config()
    cfg.camera.input_mode = mode
    with pytest.raises(ValueError, match='camera.input_mode'):
        classes.task._validate_config(cfg, NS(dt=.005), 'cuda:0')


@pytest.mark.parametrize('normalized', [False, True])
def test_initialization_and_first_capture_are_zero_with_both_encodings(sensor, normalized):
    sensor.cfg.camera.input_mode, sensor.cfg.camera.normalize = 'zero', normalized
    sensor._init_pie_buffers()
    assert torch.count_nonzero(sensor.depth_history) == 0
    assert torch.count_nonzero(sensor.depth_queue) == 0
    assert (sensor.depth_frame_ids == -1).all()
    sensor._reset_pie_sensors(torch.arange(sensor.num_envs))
    sensor.compute_observations()
    assert_blind_inputs(sensor)
    assert (sensor.depth_frame_ids >= 0).all()


def test_camera_period_partial_reset_and_native_full_reset_keep_zero_inputs(blind_sensor, monkeypatch):
    task = blind_sensor
    initial_ids = task.depth_frame_ids.clone()
    initial_capture = task.camera_capture_serial
    task._termination_reasons = {}
    task.common_step_counter = 1
    task._before_reset()
    assert task.camera_capture_serial == initial_capture
    task.common_step_counter = task.cfg.camera.update_every
    task._before_reset()
    assert task.camera_capture_serial == initial_capture + 1
    torch.testing.assert_close(task.depth_frame_ids, initial_ids, rtol=0, atol=0)
    assert_blind_inputs(task)
    task._render_depth()
    assert (task.depth_frame_ids[:, -1] != initial_ids[:, -1]).all()
    previous_ids = task.depth_frame_ids.clone()
    reset_ids = []
    def native_reset_stub(env, ids):
        reset_ids.append(ids.clone())
        env.extras['episode'] = {}
    monkeypatch.setattr(type(task).__bases__[0], 'reset_idx', native_reset_stub)
    def unexpected_curriculum(ids):
        raise AssertionError('Flat experiment must not update terrain curriculum')
    monkeypatch.setattr(task, '_update_terrain_curriculum', unexpected_curriculum)
    task.reset_idx(torch.tensor([1]))
    task.compute_observations()
    assert_blind_inputs(task)
    torch.testing.assert_close(task.depth_frame_ids[[0, 2]], previous_ids[[0, 2]], rtol=0, atol=0)
    assert (task.depth_frame_ids[1] != previous_ids[1]).all()
    assert task.depth_frame_ids[1].unique().numel() == 1
    def tensor_step(actions):
        task.actions.copy_(actions)
        task.compute_observations()
        return task.obs_buf, task.privileged_obs_buf, task.rew_buf, task.reset_buf, task.extras
    monkeypatch.setattr(task, 'step', tensor_step)
    task.reset()  # Actual BaseTask.reset -> Lite3PIE.reset_idx -> sensor reset.
    assert reset_ids[-1].tolist() == [0, 1, 2]
    assert (task.depth_frame_ids[:, 0] == task.depth_frame_ids[:, 1]).all()
    assert_blind_inputs(task)


@pytest.mark.parametrize('actuators', [False, True])
@pytest.mark.parametrize('camera', [False, True])
def test_camera_randomization_is_independent_and_partial_reset_is_local(blind_sensor, actuators, camera):
    task = blind_sensor
    task.cfg.domain_rand.randomize_pie = actuators
    task.cfg.domain_rand.randomize_camera = camera
    before = {name: getattr(task, name).clone() for name in
              ('kp_factors', 'kd_factors', 'motor_factors', 'delay_steps',
               'camera_offsets', 'camera_pitch', 'camera_fov')}
    torch.manual_seed(71)
    task._reset_pie_sensors(torch.tensor([1]))
    for name, previous in before.items():
        torch.testing.assert_close(getattr(task, name)[[0, 2]], previous[[0, 2]], rtol=0, atol=0)
    for name in ('kp_factors', 'kd_factors', 'motor_factors'):
        assert torch.equal(getattr(task, name)[1], torch.ones(12)) == (not actuators)
    nominal_position = torch.tensor(task.cfg.camera.position)
    assert torch.equal(task.camera_offsets[1], nominal_position) == (not camera)
    if not camera:
        torch.testing.assert_close(task.camera_pitch[1], torch.tensor(np.deg2rad(task.cfg.camera.pitch_degrees), dtype=torch.float32))
        assert task.camera_fov[1] == task.cfg.camera.hfov_degrees
    if not actuators:
        assert task.delay_steps[1] == 0
    assert_blind_inputs(task)


def test_different_raw_captures_produce_identical_actions_for_fixed_proprio_and_gru(blind_sensor):
    task = blind_sensor
    torch.manual_seed(91)
    model = PIEActorCritic(model_config(proprio_dim=45, proprio_history=10,
                                       action_dim=12, heightmap_dim=187, critic_dim=235)).eval()
    first = assert_blind_inputs(task)
    original_raw = task.camera.depth.clone()
    hidden = torch.randn(task.num_envs, model.cfg.gru_dim)
    task.camera.render = lambda ids=None: task.camera.depth.fill_(2.75)
    task._render_depth()
    task._render_depth()  # Advance the fixed one-frame delay as well.
    second = assert_blind_inputs(task)
    assert not torch.equal(original_raw, task.camera.depth)
    assert not torch.equal(first['depth_frame_ids'], second['depth_frame_ids'])
    with torch.no_grad():
        expected, expected_hidden, _ = model.policy_distribution(first, hidden)
        actual, actual_hidden, _ = model.policy_distribution(second, hidden)
    torch.testing.assert_close(actual.mean, expected.mean, rtol=0, atol=0)
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)


class ModeEnvironment(TensorEnvironment):
    def __init__(self, mode):
        super().__init__()
        self.cfg = load_native_classes().config()
        self.cfg.camera.input_mode = mode

    def get_pie_observations(self):
        obs = super().get_pie_observations()
        if self.cfg.camera.input_mode == 'zero':
            obs['depth'].zero_()
        return obs


def mode_runner(mode, output=None):
    return PIEOnPolicyRunner(ModeEnvironment(mode), train_config(),
                             log_dir=None if output is None else str(output), device='cpu')


@pytest.mark.parametrize('saved_mode,current_mode', [('depth', 'depth'), ('zero', 'zero'),
                                                   ('depth', 'zero'), ('zero', 'depth')])
@pytest.mark.parametrize('load_optimizer', [False, True])
def test_checkpoint_mode_matching_precedes_weight_or_optimizer_loading(tmp_path, saved_mode, current_mode, load_optimizer):
    original = mode_runner(saved_mode)
    original.current_learning_iteration = 500
    path = tmp_path / 'mode.pt'
    original.save(path)
    saved = torch.load(path, weights_only=True)
    assert saved['pie_checkpoint_version'] == 4
    assert saved['environment_cfg']['camera']['input_mode'] == saved_mode
    target = mode_runner(current_mode)
    if current_mode == saved_mode:
        target.load(path, load_optimizer=load_optimizer)
        assert target.current_learning_iteration == 500
    else:
        before = deepcopy(target.alg.actor_critic.state_dict())
        with pytest.raises(ValueError, match='camera.input_mode'):
            target.load(path, load_optimizer=load_optimizer)
        assert target.current_learning_iteration == 0
        for name, value in before.items():
            torch.testing.assert_close(target.alg.actor_critic.state_dict()[name], value, rtol=0, atol=0)


def test_blind_resume_restores_training_state_and_keeps_all_auxiliary_losses(tmp_path):
    original = mode_runner('zero', tmp_path)
    assert original.current_learning_iteration == original.tot_timesteps == 0
    original.learn(1)
    original.alg.learning_rate = 7e-4
    original.alg.optimizer.param_groups[0]['lr'] = 7e-4
    path = tmp_path / 'resume.pt'
    original.save(path)
    target = mode_runner('zero')
    target.load(path)
    assert target.current_learning_iteration == 1 and target.tot_timesteps == 9
    assert target.alg.learning_rate == pytest.approx(7e-4)
    assert target.alg.optimizer.param_groups[0]['lr'] == pytest.approx(7e-4)
    for parameter, states in original.alg.optimizer.state_dict()['state'].items():
        for name, value in states.items():
            torch.testing.assert_close(target.alg.optimizer.state_dict()['state'][parameter][name], value, rtol=0, atol=0)
    report = target.learn(1)
    assert report['iteration'] == 2 and target.tot_timesteps == 18
    for name in ('velocity', 'foot_clearance', 'heightmap', 'successor', 'kl'):
        assert np.isfinite(report[name]) and report[name] > 0
    assert target.alg.actor_critic.depth_encoder[0].weight.grad is not None
    assert torch.count_nonzero(target.alg.actor_critic.depth_encoder[0].weight.grad) == 0
    assert target.alg.actor_critic.depth_encoder[0].bias.grad.abs().sum() > 0
    assert target.alg.actor_critic.gru.weight_hh.grad.abs().sum() > 0
    original.writer.close()


@pytest.mark.parametrize('mode', ['depth', 'zero'])
def test_play_restores_mode_and_full_saved_config_before_runner_load(mode, tmp_path):
    classes = load_native_classes()
    original = mode_runner(mode)
    original.env.cfg.domain_rand.randomize_camera = True
    original.env.cfg.control.stiffness = {'joint': 17.0}
    path = tmp_path / 'play.pt'
    original.save(path)
    saved = torch.load(path, weights_only=True)
    restored, _ = classes.helpers.restore_playback_config(classes.config(), saved)
    assert restored.camera.input_mode == mode
    assert restored.control.stiffness == {'joint': 17.0}
    assert not restored.domain_rand.randomize_camera
    target = mode_runner('zero')
    target.env.cfg = restored
    target.load(path, load_optimizer=False)


def test_existing_v4_without_new_fields_plays_depth_and_cannot_resume_blind(tmp_path):
    classes = load_native_classes()
    path = tmp_path / 'existing_v4.pt'
    mode_runner('depth').save(path)
    saved = torch.load(path, weights_only=True)
    del saved['environment_cfg']['camera']['input_mode']
    del saved['environment_cfg']['domain_rand']['randomize_camera']
    torch.save(saved, path)
    restored, _ = classes.helpers.restore_playback_config(classes.config(), saved)
    assert restored.camera.input_mode == 'depth'
    assert not restored.domain_rand.randomize_camera
    visual = mode_runner('depth')
    visual.env.cfg = restored
    visual.load(path, load_optimizer=False)
    with pytest.raises(ValueError, match='camera.input_mode'):
        mode_runner('zero').load(path)


@pytest.mark.parametrize('mode', ['invalid', None])
def test_invalid_saved_camera_mode_is_rejected_by_play_and_runner(tmp_path, mode):
    classes = load_native_classes()
    path = tmp_path / 'invalid_mode.pt'
    mode_runner('zero').save(path)
    saved = torch.load(path, weights_only=True)
    saved['environment_cfg']['camera']['input_mode'] = mode
    torch.save(saved, path)
    with pytest.raises(ValueError, match='camera.input_mode'):
        classes.helpers.restore_playback_config(classes.config(), saved)
    with pytest.raises(ValueError, match='camera.input_mode'):
        mode_runner('zero').load(path)
