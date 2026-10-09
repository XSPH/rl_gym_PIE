"""Independent pre-refactor reference and native extension boundary checks."""
from copy import deepcopy
import gzip
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from native_cpu_helpers import load_native_classes
from native_rsl_helpers import TensorEnvironment, algorithm_config, model_config, train_config, rollout
from rsl_rl.algorithms import PPO, PIEPPO
from rsl_rl.modules import PIEActorCritic
from rsl_rl.runners import OnPolicyRunner, PIEOnPolicyRunner
from rsl_rl.storage.rollout_storage_pie import PIERolloutStorage
from rsl_rl.utils.pie_config import (
    V4_MODEL_DEFAULTS, normalize_model_config, normalize_train_config, checkpoint_train_config,
)


@pytest.fixture(scope='module', autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_common_ppo_and_runner_are_byte_identical_to_upstream_v102():
    # Raw files at leggedrobotics/rsl_rl commit 2ad79cf0caa85b91721abfe358105f869a784121.
    for cls, digest in (
        (PPO, '33d531858767e49bbffce8030ac2879a30a00a64caeef0f87502285fa15301cd'),
        (OnPolicyRunner, '3a704d0cdf74c4637dc34e25660de122308a0443a0a91eff03b1bbcafe3fb8d7'),
    ):
        assert hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest() == digest
    assert issubclass(PIEPPO, PPO) and PIEPPO.update is not PPO.update
    assert issubclass(PIEOnPolicyRunner, OnPolicyRunner) and PIEOnPolicyRunner.learn is not OnPolicyRunner.learn


def test_flat_task_policy_matches_explicit_native_model_signature():
    classes = load_native_classes()
    policy = classes.helpers.class_to_dict(classes.train_config().policy)
    assert 'model_config' not in policy
    signature = inspect.signature(PIEActorCritic)
    signature.bind(45, 235, 12, **policy)
    assert all(parameter.kind != inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    model = PIEActorCritic(45, 235, 12, **policy)
    assert model.get_model_config() == V4_MODEL_DEFAULTS
    with pytest.raises(TypeError, match='model_config'):
        PIEActorCritic(45, 235, 12, model_config={})


def test_v4_boundary_normalizes_missing_fields_without_rng_or_source_mutation():
    legacy = train_config()
    legacy['policy'] = {'model_config': {'proprio_dim': 3, 'critic_dim': 4, 'action_dim': 2,
                                        'token_dim': 16, 'actor_hidden_dims': [19]},
                        'actor_hidden_dims': [11, 7], 'init_noise_std': .73}
    original, rng = deepcopy(legacy), torch.get_rng_state()
    normalized = normalize_train_config(legacy)
    assert legacy == original
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    policy = normalized['policy']
    assert 'model_config' not in policy
    assert policy['actor_hidden_dims'] == [11, 7]
    assert policy['init_noise_std'] == .73
    assert policy['token_dim'] == 16 and policy['transformer_dropout'] == 0.
    model = PIEActorCritic(**policy)
    saved = checkpoint_train_config(normalized, model.get_model_config())
    restored = PIEActorCritic(**normalize_train_config(saved)['policy'])
    assert restored.get_model_config() == model.get_model_config()
    assert set(saved['policy']) == {'model_config', 'actor_hidden_dims', 'critic_hidden_dims',
                                    'init_noise_std', 'activation'}
    assert 'num_actor_obs' not in saved['policy']['model_config']
    assert normalize_model_config({}) == V4_MODEL_DEFAULTS
    with pytest.raises(ValueError, match='Unknown checkpoint model fields'):
        normalize_model_config({'misspelled_dimension': 7})


def test_playback_legacy_policy_overrides_current_flat_architecture():
    classes = load_native_classes()
    current = classes.train_config()
    current.policy.token_dim = 64
    current.policy.proprio_hidden_dims = (9,)
    legacy = train_config()
    legacy['policy'] = {'model_config': dict(V4_MODEL_DEFAULTS, gru_dim=32)}
    # This is the same boundary used by play.py before applying saved config.
    classes.helpers.update_class_from_dict(current, normalize_train_config(legacy))
    learner = classes.helpers.class_to_dict(current)
    model = PIEActorCritic(**learner['policy'])
    assert model.token_dim == 128 and model.gru_dim == 32
    assert model.proprio_hidden_dims == (512, 256)
    assert model.get_model_config() == dict(V4_MODEL_DEFAULTS, gru_dim=32)


@pytest.mark.parametrize('indexed', [True, False])
def test_native_transition_snapshots_env_buffers_before_reset(indexed):
    model = PIEActorCritic(**model_config())
    algorithm = PIEPPO(model, **algorithm_config(num_mini_batches=1))
    env = TensorEnvironment(indexed=indexed)
    algorithm.init_storage(3, 1, [3], [4], [2])
    algorithm.begin_rollout()
    obs = env.get_pie_observations()
    expected = deepcopy(obs)
    with torch.no_grad():
        actions = algorithm.act(obs, obs['critic'])
        _, _, reward, done, infos = env.step(actions)
        for name, value in obs.items():
            if isinstance(value, dict):
                for tensor in value.values():
                    tensor.fill_(-999)
            else:
                value.fill_(-999)
        algorithm.process_env_step(reward, done, infos)
    storage = algorithm.storage
    assert isinstance(algorithm.transition, PIERolloutStorage.Transition)
    assert algorithm.transition.targets is None
    assert not hasattr(storage, 'frames') and not hasattr(storage, 'as_batch')
    restored = storage.observation(0, restore_depth=True)
    for name in ('proprio', 'critic', 'proprio_history', 'depth'):
        torch.testing.assert_close(restored[name], expected[name], rtol=0, atol=0)
    for name, target in expected['targets'].items():
        torch.testing.assert_close(storage.targets[name][0], target, rtol=0, atol=0)
    assert storage.proprio_history.shape == (1, 3, 2, 3)


def test_native_minibatches_keep_whole_trajectories_and_reuse_allocations():
    algorithm, _ = rollout(steps=6)
    storage = algorithm.storage
    batches = list(storage.recurrent_mini_batch_generator(2, 2))
    assert len(batches) == 4
    for batch in batches:
        obs, critic, actions, _, _, _, _, _, _, hidden, masks, targets, successor, valid, next_valid, frames = batch
        steps, envs = masks.shape
        assert steps == 6 and envs in (1, 2)
        assert actions.shape == (steps * envs, 2)
        assert hidden[0].shape == (envs, 8) and hidden[1] is None
        assert critic.data_ptr() == obs['critic'].data_ptr()
        assert targets['velocity'].shape == successor.shape == (steps, envs, 3)
        assert valid.shape == next_valid.shape == (steps, envs)
        assert frames.data_ptr() == storage.depth_pool.materialize().data_ptr()
    buffer = storage.proprio_history
    storage.clear()
    assert storage.proprio_history is buffer
    assert storage.step == 0 and storage.initial_hidden is None
    assert storage.depth_pool.materialize() is None
    with pytest.raises(RuntimeError, match='complete rollout'):
        next(storage.recurrent_mini_batch_generator(1, 1))


def _assert_reference(actual, expected, path='reference'):
    if isinstance(expected, dict) and 'tensor' in expected:
        reference = torch.tensor(expected['tensor'], dtype=getattr(torch, expected['dtype']))
        if path == 'reference.initial' or path.startswith('reference.initial.'):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0, msg=path)
        elif '.model.transformer.layers.0.self_attn.in_proj_bias' in path:
            # Attention key bias cancels in softmax. Adam amplifies its tiny
            # FP32 residuals; only that slice needs the looser parameter bound.
            torch.testing.assert_close(actual[:8], reference[:8], rtol=1e-5, atol=2e-7, msg=path)
            torch.testing.assert_close(actual[16:], reference[16:], rtol=1e-5, atol=2e-7, msg=path)
            torch.testing.assert_close(actual[8:16], reference[8:16], rtol=1e-5, atol=5e-6, msg=path)
        else:
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=2e-7, msg=path)
    elif isinstance(expected, dict):
        actual = {str(key): value for key, value in actual.items()}
        assert list(actual) == list(expected), path
        for name, value in expected.items():
            _assert_reference(actual[name], value, path + '.' + name)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (value, reference) in enumerate(zip(actual, expected)):
            _assert_reference(value, reference, path + '.' + str(index))
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-5, abs=2e-7), path
    else:
        assert actual == expected, path


def test_two_updates_match_independent_pre_refactor_checkpoint_gradient_and_adam_reference():
    fixture = Path(__file__).parent / 'fixtures/pie_before_rsl_refactor.json.gz'
    with gzip.open(fixture, 'rt') as stream:
        baseline = json.load(stream)
    assert baseline['source_commit'] == '3f64242'
    torch.manual_seed(baseline['seed'])
    runner = PIEOnPolicyRunner(TensorEnvironment(), train_config(
        cfg=algorithm_config(num_learning_epochs=2, num_mini_batches=2), rollout=6), device='cpu')
    _assert_reference(runner.alg.actor_critic.state_dict(), baseline['reference']['initial'], 'reference.initial')
    for expected in baseline['reference']['updates']:
        runner.learn(1)
        model = runner.alg.actor_critic
        actual = {'model': model.state_dict(),
                  'gradients': {name: param.grad for name, param in model.named_parameters() if param.grad is not None},
                  'optimizer': runner.alg.optimizer.state_dict(), 'hidden': model.get_hidden_states()[0],
                  'metrics': runner.alg.metrics, 'learning_rate': runner.alg.learning_rate,
                  'rng': torch.get_rng_state(), 'iteration': runner.current_learning_iteration}
        _assert_reference(actual, expected)
