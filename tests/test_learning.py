"""CPU checks for native actor privacy, recurrent reset, and timeout bootstraps."""
from native_rsl_helpers import restore_observation
from copy import deepcopy

import pytest
import torch

from native_rsl_helpers import algorithm_config, TensorEnvironment, collect_native, model_config, rollout
from rsl_rl.algorithms.ppo_pie import PIEPPO
from rsl_rl.modules.actor_critic_pie import PIEActorCritic


def test_privileged_inputs_cannot_change_actor_but_do_change_critic():
    torch.manual_seed(5)
    model = PIEActorCritic(**model_config()).eval()
    obs = TensorEnvironment().get_pie_observations()
    hidden = model.initial_state(3)
    actor1 = model.policy_distribution(obs, hidden)[0].mean
    critic1 = model.evaluate(obs["critic"])
    changed = deepcopy(obs)
    changed["critic"].add_(1000)
    for target in changed["targets"].values():
        target.add_(1000)
    actor2 = model.policy_distribution(changed, hidden)[0].mean
    critic2 = model.evaluate(changed["critic"])
    torch.testing.assert_close(actor1, actor2, rtol=0, atol=0)
    assert not torch.equal(critic1, critic2)


def test_native_actor_probability_replay_and_partial_recurrent_reset():
    torch.manual_seed(9)
    model = PIEActorCritic(**model_config()).eval()
    obs = TensorEnvironment().get_pie_observations()
    actions = model.act(obs)
    old_logp = model.get_actions_log_prob(actions).detach().clone()
    replay = model.evaluate_actions(obs, model.initial_state(3), actions)
    torch.testing.assert_close(replay[0], old_logp, rtol=0, atol=0)
    actor_hidden, critic_hidden = model.get_hidden_states()
    assert actor_hidden.shape == (3, model.gru_dim) and critic_hidden is None
    continuing = actor_hidden[[0, 2]].clone()
    model.reset(torch.tensor([False, True, False]))
    actor_hidden, _ = model.get_hidden_states()
    assert torch.count_nonzero(actor_hidden[1]) == 0
    torch.testing.assert_close(actor_hidden[[0, 2]], continuing, rtol=0, atol=0)
    dirty = torch.randn_like(actor_hidden)
    reset = torch.tensor([True, False, True])
    masked = model.policy_distribution(obs, dirty, reset)[0].mean
    fresh = model.policy_distribution(obs, model.initial_state(3))[0].mean
    torch.testing.assert_close(masked[[0, 2]], fresh[[0, 2]], rtol=0, atol=0)
    assert not torch.equal(masked[1], fresh[1])


def test_native_storage_uses_pre_reset_successor_and_truefinal_value_once():
    cfg = algorithm_config(num_learning_epochs=1, num_mini_batches=1, gamma=.9, lam=.95)
    algorithm, batch = rollout(cfg)
    first, timeout, last = batch["frames"]
    assert first["terminated"].tolist() == [True, False, False]
    assert timeout["truncated"].tolist() == [False, True, False]
    assert first["successor"][0, 0].item() == pytest.approx(7.2)
    assert timeout["successor"][1, 0].item() == pytest.approx(7.7)
    assert [frame["reset"].tolist() for frame in batch["frames"]] == [
        [True, True, True], [True, False, False], [False, True, False]]
    env = TensorEnvironment()
    env.step_index = 2
    terminal = env.get_privileged_observations().clone()
    terminal[1] += 11
    with torch.no_grad():
        final_value = algorithm.actor_critic.evaluate(terminal)[:, 0]
        reset_value = algorithm.actor_critic.evaluate(env.get_privileged_observations())[:, 0]
    assert not torch.equal(final_value[1], reset_value[1])
    torch.testing.assert_close(timeout["rewards"][1],
                               timeout["raw_rewards"][1] + .9 * final_value[1])
    torch.testing.assert_close(batch["returns"][1, 1], timeout["rewards"][1])
    torch.testing.assert_close(batch["returns"][0, 0], first["raw_rewards"][0])
    # The episode after either termination cannot contribute to its predecessor.
    assert batch["returns"][0, 0] == 1
    assert last["reset"].tolist() == [False, True, False]


def test_joint_update_trains_actor_cnn_gru_and_all_auxiliary_heads():
    algorithm, batch = rollout(algorithm_config(num_learning_epochs=2, num_mini_batches=2, schedule="fixed"))
    before = {name: value.detach().clone() for name, value in algorithm.actor_critic.named_parameters()}
    losses = algorithm.update()
    assert isinstance(losses, tuple) and len(losses) == 2
    assert all(torch.isfinite(torch.tensor(value)) for value in algorithm.metrics.values())
    assert algorithm.storage.step == 0 and algorithm.storage.initial_hidden is None
    for name in ("actor.0.weight", "critic.0.weight", "depth_encoder.0.weight",
                 "gru.weight_ih", "velocity_head.weight", "clearance_head.weight",
                 "height_decoder.0.weight", "successor_decoder.0.weight",
                 "mu_head.weight", "logvar_head.weight"):
        actual = dict(algorithm.actor_critic.named_parameters())[name]
        assert not torch.equal(actual, before[name]), name


def test_replay_under_updated_weights_preserves_continuing_episode_memory():
    model = PIEActorCritic(**model_config())
    algorithm = PIEPPO(model, device="cpu", **algorithm_config(num_learning_epochs=1, num_mini_batches=1))
    environment = TensorEnvironment(resets={1: (0, False), 3: (1, True)})
    batch, _ = collect_native(algorithm, environment)
    algorithm.update()
    with torch.no_grad():
        expected = batch["hidden"].clone()
        for frame in batch["frames"]:
            obs = restore_observation(
                frame, depth_frames=batch["depth_frames"], restore_depth=True)
            _, expected = algorithm.actor_critic.encode(obs, expected, frame["reset"])
        expected[1] = 0
    hidden, _ = algorithm.actor_critic.get_hidden_states()
    torch.testing.assert_close(hidden, expected, rtol=1e-5, atol=1e-6)
    assert hidden[0].norm() > 0 and hidden[2].norm() > 0
    assert not hidden.requires_grad


def test_invalid_successor_masks_only_next_state_loss():
    model = PIEActorCritic(**model_config())
    obs = TensorEnvironment().get_pie_observations()
    _, _, estimates = model.policy_distribution(obs, model.initial_state(3))
    successor = torch.full_like(obs["proprio"], float("nan"))
    losses = model.auxiliary_losses(estimates, obs["targets"], successor,
                                   successor_valid=torch.zeros(3, dtype=torch.bool))
    assert losses["successor"].item() == 0
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert model.velocity_head.weight.grad.abs().sum() > 0
    assert model.clearance_head.weight.grad.abs().sum() > 0
    assert model.height_decoder[0].weight.grad.abs().sum() > 0


def test_actor_policy_loss_trains_estimator_without_critic_or_variance():
    model = PIEActorCritic(**model_config())
    obs = TensorEnvironment().get_pie_observations()
    distribution, _, _ = model.policy_distribution(obs, model.initial_state(3))
    actions = distribution.mean.detach() + .1
    distribution.log_prob(actions).sum().backward()
    for name in ("depth_encoder.0.weight", "gru.weight_ih", "velocity_head.weight",
                 "map_head.weight", "mu_head.weight"):
        assert dict(model.named_parameters())[name].grad.abs().sum() > 0
    assert model.logvar_head.weight.grad is None
    assert model.critic[0].weight.grad is None
