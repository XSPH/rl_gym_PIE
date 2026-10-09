"""Native PPO scheduling, behavior distributions, and activation gradients."""
from native_rsl_helpers import restore_observation

import pytest
import torch
from torch.distributions import Normal, kl_divergence

from native_rsl_helpers import algorithm_config, rollout
from rsl_rl.modules.actor_critic_pie import PIEActorCritic


@pytest.fixture(scope="module", autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_formal_defaults_are_passed_to_native_ppo_constructor():
    cfg = algorithm_config()
    algorithm, _ = rollout(cfg)
    assert cfg["learning_rate"] == 1e-3 and cfg["num_learning_epochs"] == 5 and cfg["num_mini_batches"] == 4
    assert cfg["schedule"] == "adaptive" and cfg["desired_kl"] == .01
    assert algorithm.num_learning_epochs == 5 and algorithm.num_mini_batches == 4
    assert algorithm.learning_rate == algorithm.optimizer.param_groups[0]["lr"] == 1e-3
    assert algorithm.gamma == .99 and algorithm.lam == .95
    assert algorithm.use_clipped_value_loss


@pytest.mark.parametrize('override,message', [
    ({'schedule': 'unknown'}, 'schedule'),
    ({'desired_kl': 0}, 'desired_kl'),
    ({'desired_kl': float('inf')}, 'desired_kl'),
])
def test_invalid_native_algorithm_parameters_fail_at_initialization(override, message):
    with pytest.raises(ValueError, match=message):
        rollout(algorithm_config(**override))


def test_native_policy_configuration_controls_std_width_and_activation():
    from native_rsl_helpers import model_config
    cfg = model_config()
    cfg.update(init_noise_std=.73, actor_hidden_dims=[11, 7], critic_hidden_dims=[13], activation="tanh")
    model = PIEActorCritic(**cfg)
    torch.testing.assert_close(model.std, torch.full((2,), .73), rtol=0, atol=0)
    assert model.actor[0].out_features == 11 and model.actor[2].out_features == 7
    assert model.critic[0].out_features == 13
    assert isinstance(model.actor[1], torch.nn.Tanh)
    assert model.init_noise_std == .73
    assert model.actor_hidden_dims == (11, 7) and model.critic_hidden_dims == (13,)


def test_behavior_distributions_share_native_storage_and_are_immutable():
    algorithm, batch = rollout()
    old_mu, old_sigma = algorithm.storage.mu.clone(), algorithm.storage.sigma.clone()
    for index, frame in enumerate(batch["frames"]):
        assert frame["old_mu"].data_ptr() == algorithm.storage.mu[index].data_ptr()
        assert frame["old_sigma"].data_ptr() == algorithm.storage.sigma[index].data_ptr()
        assert not frame["old_mu"].requires_grad
    with torch.no_grad():
        algorithm.actor_critic.actor[-1].bias.add_(1)
        algorithm.actor_critic.std.mul_(2)
    torch.testing.assert_close(algorithm.storage.mu, old_mu, rtol=0, atol=0)
    torch.testing.assert_close(algorithm.storage.sigma, old_sigma, rtol=0, atol=0)


@pytest.mark.parametrize("shift,scale,lr,schedule,desired_kl,expected_lr", [
    (.1, 1., 1e-3, "adaptive", .01, 1e-3 / 1.5),
    (.02, 1., 1e-3, "adaptive", .01, 1e-3 * 1.5),
    (.05, 1., 1e-3, "adaptive", .01, 1e-3),
    (0., 2., 1e-3, "adaptive", .01, 1e-3 / 1.5),
    (.1, 1., 1e-5, "adaptive", .01, 1e-5),
    (.02, 1., 1e-2, "adaptive", .01, 1e-2),
    (.1, 1., 1e-3, "fixed", .01, 1e-3),
    (.02, 1., 1e-3, "fixed", .01, 1e-3),
    (.1, 1., 1e-3, "adaptive", None, 1e-3),
    (.1, 1., 1e-3, "adaptive", .1, 1e-3 * 1.5),
])
def test_lr_schedule_precedes_each_native_optimizer_step(
        monkeypatch, shift, scale, lr, schedule, desired_kl, expected_lr):
    algorithm, _ = rollout(algorithm_config(num_learning_epochs=1, num_mini_batches=1, learning_rate=lr,
                                    schedule=schedule, desired_kl=desired_kl))
    with torch.no_grad():
        algorithm.actor_critic.actor[-1].bias.add_(shift)
        algorithm.actor_critic.std.mul_(scale)
    step_lrs = []
    original = algorithm.optimizer.step
    def step():
        step_lrs.append(algorithm.optimizer.param_groups[0]["lr"])
        return original()
    monkeypatch.setattr(algorithm.optimizer, "step", step)
    algorithm.update()
    assert step_lrs == [pytest.approx(expected_lr)]
    assert algorithm.learning_rate == pytest.approx(expected_lr)
    assert "policy_kl" in algorithm.metrics and "kl" in algorithm.metrics


def test_policy_kl_unrolls_recurrent_gaussians_and_keeps_vae_kl_separate():
    algorithm, batch = rollout(algorithm_config(num_learning_epochs=1, num_mini_batches=1, schedule="fixed"))
    with torch.no_grad():
        algorithm.actor_critic.actor[0].weight.add_(.05)
        algorithm.actor_critic.gru.weight_hh.add_(.03)
        algorithm.actor_critic.std.copy_(torch.tensor([.75, .35]))
        hidden = batch["hidden"].clone()
        references = []
        for frame in batch["frames"]:
            obs = restore_observation(frame, depth_frames=batch["depth_frames"], restore_depth=True)
            _, _, _, hidden, _ = algorithm.actor_critic.evaluate_actions(
                obs, hidden, frame["actions"], frame["reset"])
            current = algorithm.actor_critic.distribution
            behavior = Normal(frame["old_mu"], frame["old_sigma"])
            correction = torch.log1p(1e-5 * behavior.scale / current.scale)
            references.append((kl_divergence(behavior, current) + correction).sum(-1))
        expected = torch.stack(references).mean().item()
    algorithm.update()
    assert algorithm.metrics["policy_kl"] == pytest.approx(expected, abs=3e-7)
    assert algorithm.metrics["kl"] != algorithm.metrics["policy_kl"]


def test_adaptive_lr_runs_for_every_whole_trajectory_minibatch(monkeypatch):
    algorithm, _ = rollout(algorithm_config(num_learning_epochs=2, num_mini_batches=2))
    with torch.no_grad():
        algorithm.actor_critic.actor[-1].bias.add_(1)
    original = algorithm.optimizer.step
    step_lrs = []
    def step():
        step_lrs.append(algorithm.optimizer.param_groups[0]["lr"])
        return original()
    monkeypatch.setattr(algorithm.optimizer, "step", step)
    algorithm.update()
    assert step_lrs == pytest.approx([1e-3 / 1.5 ** index for index in range(1, 5)])


def test_activation_checkpoint_preserves_joint_update_and_gradients(monkeypatch):
    import rsl_rl.modules.actor_critic_pie as network
    settings = algorithm_config(num_learning_epochs=2, num_mini_batches=2, schedule="fixed")
    reference, _ = rollout(settings)
    checkpointed, _ = rollout(settings)
    torch.manual_seed(17)
    with monkeypatch.context() as plain:
        plain.setattr(network, "checkpoint", lambda function, *args, **kwargs: function(*args))
        reference.update()
    torch.manual_seed(17)
    checkpointed.update()
    for name, expected in reference.actor_critic.named_parameters():
        actual = dict(checkpointed.actor_critic.named_parameters())[name]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
    assert checkpointed.actor_critic.depth_encoder[0].weight.grad.abs().sum() > 0
    assert checkpointed.actor_critic.gru.weight_hh.grad.abs().sum() > 0


def test_visual_checkpoint_reduces_retained_activation_storage(monkeypatch):
    import rsl_rl.modules.actor_critic_pie as network
    model = PIEActorCritic(45, 235, 12).train()
    obs = {"proprio_history": torch.zeros(4, 10, 45), "depth": torch.randn(4, 2, 60, 80)}
    parameter_storages = {value.untyped_storage().data_ptr() for value in model.parameters()}
    def retained_bytes():
        saved = {}
        def pack(tensor):
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in parameter_storages:
                saved[storage.data_ptr()] = storage.nbytes()
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            model.encode(obs, model.initial_state(4))
        return sum(saved.values())
    with monkeypatch.context() as plain:
        plain.setattr(network, "checkpoint", lambda function, *args, **kwargs: function(*args))
        direct = retained_bytes()
    assert retained_bytes() < direct * .6
