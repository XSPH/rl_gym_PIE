"""Compare real task/model behavior with the pre-cleanup commit 09c2122."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from native_cpu_helpers import load_native_classes
from rsl_rl.modules.actor_critic_pie import PIEActorCritic
from test_pie_observation_and_push import sensor


BASELINE = json.loads((Path(__file__).parent / 'fixtures/pie_before_cleanup.json').read_text())


def test_effective_nested_configuration_matches_previous_values():
    classes = load_native_classes()
    cfg = classes.config()
    assert classes.helpers.class_to_dict(cfg) == BASELINE['configuration']
    assert not hasattr(cfg, 'pie')
    train = classes.train_config()
    assert (cfg.env.num_envs, train.runner.num_steps_per_env,
            train.algorithm.num_learning_epochs, train.algorithm.num_mini_batches,
            train.runner.max_iterations, train.runner.save_interval) == (4096, 24, 5, 4, 15000, 500)
    classes.task._validate_config(cfg, SimpleNamespace(dt=.005), 'cuda:0')


def test_mesh_heights_origins_and_triangles_match_previous_seed():
    classes = load_native_classes()
    terrain = classes.terrain(classes.config().terrain, 4096, seed=4)
    for name, expected in BASELINE['geometry_seed_4'].items():
        array = np.ascontiguousarray(getattr(terrain, name))
        assert hashlib.sha256(array.tobytes()).hexdigest() == expected, name


def test_camera_fk_and_auxiliary_labels_match_previous_values(sensor):
    sensor.base_quat[:, 2] = np.sin(.3)
    sensor.base_quat[:, 3] = np.cos(.3)
    sensor._render_depth()
    expected = BASELINE['sensors']
    for key, actual in (('camera_position', sensor.camera.positions),
                        ('camera_orientation', sensor.camera.orientations),
                        ('camera_focal', sensor.camera.focal)):
        torch.testing.assert_close(actual, torch.tensor(expected[key]), atol=2e-7, rtol=1e-6)
    states = sensor.fk.forward(sensor.root_states[:, :3], sensor.base_quat,
                               sensor.dof_pos[:, sensor.joint_order])
    for name, (position, orientation) in states.items():
        torch.testing.assert_close(position, torch.tensor(expected['fk'][name][0]), atol=2e-7, rtol=1e-6)
        torch.testing.assert_close(orientation, torch.tensor(expected['fk'][name][1]), atol=2e-7, rtol=1e-6)
    for name, value in sensor._terrain_targets().items():
        torch.testing.assert_close(value, torch.tensor(expected['targets'][name]), atol=2e-7, rtol=1e-6)


def test_default_network_parameters_and_recurrent_output_match_previous_seed():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        classes = load_native_classes()
        torch.manual_seed(101)
        model = PIEActorCritic(45, 235, 12, **classes.helpers.class_to_dict(classes.train_config().policy)).eval()
        digest = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().numpy().tobytes())
        expected = BASELINE['model_seed_101']
        assert digest.hexdigest() == expected['state_sha256']
        prop = torch.linspace(-.1, .1, 45).reshape(1, 45)
        obs = {'proprio': prop, 'proprio_history': prop[:, None].repeat(1, 10, 1),
               'depth': torch.linspace(-.2, .2, 60*80).reshape(1, 1, 60, 80).repeat(1, 2, 1, 1),
               'critic': torch.cat((prop, torch.zeros(1, 190)), -1)}
        with torch.no_grad():
            for output in expected['actions']:
                torch.testing.assert_close(model.act_inference(obs), torch.tensor(output), rtol=0, atol=0)
            torch.testing.assert_close(model.evaluate(obs['critic']), torch.tensor(expected['critic']), rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize('section,field,value,message', [
    ('env', 'num_envs', 0, 'num_envs'),
    ('env', 'proprio_history', 0, 'history'),
    ('asset', 'joint_names', ['duplicate'] * 12, 'distinct policy'),
    ('camera', 'history', 3, 'depth history'),
    ('camera', 'near', 4, 'clipping'),
    ('domain_rand', 'max_delay_seconds', -.01, 'delay'),
    ('domain_rand', 'gain_factor', [1.1, .9], 'range'),
    ('terrain', 'num_rows', 0, 'positive'),
    ('terrain', 'terrain_length', 6, 'Course length'),
    ('terrain', 'kinds', ['invalid'], 'Unsupported'),
    ('terrain', 'measured_points_x', [], 'scan'),
])
def test_task_initialization_rejects_invalid_configuration(section, field, value, message):
    classes = load_native_classes()
    cfg = classes.config()
    setattr(getattr(cfg, section), field, value)
    with pytest.raises(ValueError, match=message):
        classes.task._validate_config(cfg, SimpleNamespace(dt=.005), 'cuda:0')


def test_pie_randomization_switch_preserves_independent_native_switches(sensor):
    sensor.cfg.domain_rand.randomize_friction = True
    sensor.cfg.domain_rand.randomize_base_mass = True
    sensor.cfg.domain_rand.push_robots = True
    sensor.cfg.domain_rand.randomize_pie = False
    sensor._reset_pie_sensors(torch.tensor([1]))
    assert sensor.cfg.domain_rand.randomize_friction
    assert sensor.cfg.domain_rand.randomize_base_mass
    assert sensor.cfg.domain_rand.push_robots
    assert torch.equal(sensor.kp_factors[1], torch.ones(12))
    assert sensor.delay_steps[1] == 0
    sensor.cfg.domain_rand.randomize_pie = True
    sensor.cfg.domain_rand.randomize_friction = False
    sensor.cfg.domain_rand.randomize_base_mass = False
    sensor.cfg.domain_rand.push_robots = False
    torch.manual_seed(71)
    unchanged = sensor.kp_factors[[0, 2]].clone()
    sensor._reset_pie_sensors(torch.tensor([1]))
    assert not torch.equal(sensor.kp_factors[1], torch.ones(12))
    torch.testing.assert_close(sensor.kp_factors[[0, 2]], unchanged, rtol=0, atol=0)
    assert not sensor.cfg.domain_rand.randomize_friction
    assert not sensor.cfg.domain_rand.randomize_base_mass
    assert not sensor.cfg.domain_rand.push_robots
