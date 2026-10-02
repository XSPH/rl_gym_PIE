"""CPU tensor checks for native PPO scheduling and recurrent policy KL."""
from dataclasses import asdict

import pytest
import torch
from torch.distributions import Normal, kl_divergence

from rsl_rl.algorithms.ppo_pie import PIEPPO, PPOConfig
from rsl_rl.modules.actor_critic_pie import ModelConfig, PIEActorCritic


@pytest.fixture(scope="module", autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class _TensorObservations:
    """Recorded-shaped inputs and reset masks; no simulator dependencies."""
    def __init__(self, count=3):
        self.count = count
        self.step_index = 0

    def observation(self):
        prop = torch.arange(self.count * 3, dtype=torch.float32).reshape(self.count, 3)
        prop = prop / 10 + self.step_index / 5
        return {
            "proprio": prop,
            "proprio_history": torch.stack((prop - 0.1, prop), dim=1),
            "depth": prop[:, :1, None, None].expand(-1, 1, 8, 8).clone(),
            "critic": torch.cat((prop, prop[:, :1]), dim=-1),
            "targets": {"velocity": prop,
                        "foot_clearance": torch.zeros(self.count, 4),
                        "heightmap": prop / 2},
        }

    def step(self, actions):
        self.step_index += 1
        obs = self.observation()
        terminated = torch.zeros(self.count, dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        terminated[0] = self.step_index == 1
        truncated[1] = self.step_index == 2
        return obs, torch.arange(self.count, dtype=torch.float32) + 1, terminated, truncated, {
            "terminal_proprio": obs["proprio"], "terminal_observation": obs}


def _rollout(cfg=None):
    torch.manual_seed(41)
    model = PIEActorCritic(ModelConfig(
        # Fixed variance for analytic scheduler thresholds; independent of
        # the task's fresh-training exploration default.
        initial_std=0.5, proprio_dim=3, proprio_history=2, depth_history=1, action_dim=2,
        heightmap_dim=3, critic_dim=4, token_dim=8, gru_dim=8,
        latent_dim=2, map_latent_dim=2, transformer_heads=2,
        proprio_hidden_dims=(8,), cnn_hidden_channels=(2, 2),
        cnn_kernel_sizes=(3, 3, 3), cnn_strides=(1, 1, 1),
        cnn_paddings=(1, 1, 1), visual_grid=(1, 1),
        actor_hidden_dims=(8,), critic_hidden_dims=(8,),
        successor_hidden_dims=(8,), height_decoder_hidden_dims=(8,)))
    alg = PIEPPO(model, cfg or PPOConfig(epochs=1, minibatches=1), device="cpu")
    inputs = _TensorObservations()
    initial_hidden = torch.full((inputs.count, model.cfg.gru_dim), 0.3)
    batch, *_ = alg.collect(inputs, inputs.observation(), initial_hidden,
                            torch.tensor([True, False, False]), steps=3)
    return alg, batch


def test_formal_defaults_and_native_constructor_settings():
    cfg = PPOConfig()
    assert asdict(cfg) == {
        "learning_rate": 1e-3, "gamma": 0.99, "gae_lambda": 0.95,
        "clip": 0.2, "epochs": 5, "minibatches": 4,
        "entropy_weight": 0.01, "value_weight": 1.0,
        "estimation_weight": 1.0, "kl_weight": 1.0, "max_grad_norm": 1.0,
        "schedule": "adaptive", "desired_kl": 0.01,
    }
    alg, _ = _rollout(cfg)
    assert alg.num_learning_epochs == 5 and alg.num_mini_batches == 4
    assert alg.learning_rate == alg.optimizer.param_groups[0]["lr"] == 1e-3
    assert alg.schedule == "adaptive" and alg.desired_kl == 0.01
    assert alg.gamma == 0.99 and alg.lam == 0.95
    assert alg.use_clipped_value_loss
    # Older checkpoint config dictionaries remain accepted without new keys.
    legacy_config = asdict(cfg)
    del legacy_config["schedule"], legacy_config["desired_kl"]
    assert PPOConfig(**legacy_config) == cfg


def test_behavior_distribution_reuses_native_storage_and_is_immutable():
    alg, batch = _rollout()
    old_mu, old_sigma = alg.storage.mu.clone(), alg.storage.sigma.clone()
    for index, frame in enumerate(batch["frames"]):
        assert frame["old_mu"].data_ptr() == alg.storage.mu[index].data_ptr()
        assert frame["old_sigma"].data_ptr() == alg.storage.sigma[index].data_ptr()
        assert not frame["old_mu"].requires_grad and not frame["old_sigma"].requires_grad
    with torch.no_grad():
        alg.model.actor[-1].bias.add_(1)
        alg.model.std.mul_(2)
    torch.testing.assert_close(alg.storage.mu, old_mu, rtol=0, atol=0)
    torch.testing.assert_close(alg.storage.sigma, old_sigma, rtol=0, atol=0)
    assert [frame["reset"].tolist() for frame in batch["frames"]] == [
        [True, False, False], [True, False, False], [False, True, False]]


@pytest.mark.parametrize("invalid", [
    {"schedule": "linear"}, {"desired_kl": 0}, {"desired_kl": -0.01},
    {"desired_kl": float("nan")}, {"desired_kl": float("inf")},
])
def test_invalid_schedule_configuration_is_rejected(invalid):
    with pytest.raises(ValueError):
        PPOConfig(**invalid)


@pytest.mark.parametrize("shift,scale,lr,schedule,desired_kl,expected_lr", [
    (0.1, 1.0, 1e-3, "adaptive", 0.01, 1e-3 / 1.5),
    (0.02, 1.0, 1e-3, "adaptive", 0.01, 1e-3 * 1.5),
    (0.05, 1.0, 1e-3, "adaptive", 0.01, 1e-3),
    (0.0, 2.0, 1e-3, "adaptive", 0.01, 1e-3 / 1.5),
    (0.1, 1.0, 1e-5, "adaptive", 0.01, 1e-5),
    (0.02, 1.0, 1e-2, "adaptive", 0.01, 1e-2),
    (0.1, 1.0, 1e-3, "fixed", 0.01, 1e-3),
    (0.02, 1.0, 1e-3, "fixed", 0.01, 1e-3),
    (0.1, 1.0, 1e-3, "adaptive", None, 1e-3),
    (0.1, 1.0, 1e-3, "adaptive", 0.1, 1e-3 * 1.5),
])
def test_native_schedule_changes_lr_before_optimizer_step(
        monkeypatch, shift, scale, lr, schedule, desired_kl, expected_lr):
    alg, batch = _rollout(PPOConfig(
        epochs=1, minibatches=1, learning_rate=lr,
        schedule=schedule, desired_kl=desired_kl))
    with torch.no_grad():
        alg.model.actor[-1].bias.add_(shift)
        alg.model.std.mul_(scale)
    step_lrs = []
    optimizer_step = alg.optimizer.step

    def record_step():
        step_lrs.append([group["lr"] for group in alg.optimizer.param_groups])
        optimizer_step()

    monkeypatch.setattr(alg.optimizer, "step", record_step)
    report = alg.update(batch)
    assert alg.learning_rate == pytest.approx(expected_lr)
    assert step_lrs == [[pytest.approx(expected_lr)]]
    assert "policy_kl" in report and "kl" in report
    assert all(torch.isfinite(torch.tensor(value)) for value in report.values())


def test_policy_kl_averages_recurrent_gaussians_and_preserves_vae_metric():
    alg, batch = _rollout(PPOConfig(epochs=1, minibatches=1, schedule="fixed"))
    original_hidden = batch["hidden"].clone()
    with torch.no_grad():
        alg.model.actor[0].weight.add_(0.05)
        alg.model.gru.weight_hh.add_(0.03)
        alg.model.std.copy_(torch.tensor([0.75, 0.35]))
        hidden = batch["hidden"].clone()
        references, vae_kls = [], []
        for frame in batch["frames"]:
            _, _, _, hidden, estimates = alg.model.evaluate(
                frame["obs"], hidden, frame["actions"], frame["reset"])
            current = alg.model.distribution
            behavior = Normal(frame["old_mu"], frame["old_sigma"])
            # Torch's exact KL plus the epsilon used in native v1.0.2 PPO.
            correction = torch.log1p(1e-5 * behavior.scale / current.scale)
            references.append((kl_divergence(behavior, current) + correction).sum(-1))
            vae_kls.append(alg.model.auxiliary_losses(
                estimates, frame["targets"], frame["successor"],
                frame["valid"], frame["successor_valid"])["kl"])
        expected_policy_kl = torch.stack(references).mean().item()
        expected_vae_kl = torch.stack(vae_kls).mean().item()
    assert torch.stack(references).std() > 0
    report = alg.update(batch)
    assert report["policy_kl"] == pytest.approx(expected_policy_kl, abs=1e-7)
    assert report["kl"] == pytest.approx(expected_vae_kl, abs=1e-7)
    torch.testing.assert_close(batch["hidden"], original_hidden, rtol=0, atol=0)


def test_schedule_updates_on_every_trajectory_minibatch(monkeypatch):
    alg, batch = _rollout(PPOConfig(epochs=2, minibatches=2))
    with torch.no_grad():
        alg.model.actor[-1].bias.add_(1)
    optimizer_step = alg.optimizer.step
    step_lrs = []

    def record_step():
        step_lrs.append(alg.optimizer.param_groups[0]["lr"])
        optimizer_step()

    monkeypatch.setattr(alg.optimizer, "step", record_step)
    alg.update(batch)
    assert step_lrs == pytest.approx([1e-3 / 1.5 ** step for step in range(1, 5)])


def test_std_guard_only_enforces_numeric_positivity(monkeypatch):
    alg, batch = _rollout(PPOConfig(epochs=1, minibatches=1, schedule="fixed"))
    optimizer_step = alg.optimizer.step

    def inject_scales_after_step():
        optimizer_step()
        with torch.no_grad():
            alg.model.std.copy_(torch.tensor([-1.0, 20.0]))

    monkeypatch.setattr(alg.optimizer, "step", inject_scales_after_step)
    alg.update(batch)
    assert alg.model.std[0].item() == torch.finfo(alg.model.std.dtype).eps
    assert alg.model.std[1].item() == 20.0


def test_visual_checkpoint_preserves_recurrent_joint_update(monkeypatch):
    """Real PPO updates agree, including masked auxiliaries and GRU gradients."""
    import rsl_rl.modules.actor_critic_pie as network
    settings = PPOConfig(epochs=2, minibatches=2, schedule="fixed")
    reference, reference_batch = _rollout(settings)
    checkpointed, checkpointed_batch = _rollout(settings)

    def direct_call(function, *args, **kwargs):
        return function(*args)

    torch.manual_seed(17)
    with monkeypatch.context() as plain:
        plain.setattr(network, "checkpoint", direct_call)
        expected = reference.update(reference_batch)
    torch.manual_seed(17)
    actual = checkpointed.update(checkpointed_batch)

    for key in expected:
        assert actual[key] == pytest.approx(expected[key], rel=1e-5, abs=1e-7)
    reference_parameters = dict(reference.model.named_parameters())
    for name, parameter in checkpointed.model.named_parameters():
        torch.testing.assert_close(parameter, reference_parameters[name], rtol=1e-5, atol=1e-7)
        if reference_parameters[name].grad is not None:
            torch.testing.assert_close(parameter.grad, reference_parameters[name].grad,
                                       rtol=1e-5, atol=1e-7)
    # Depth inputs never require gradients; the CNN must still receive them.
    assert not checkpointed_batch["frames"][0]["obs"]["depth"].requires_grad
    assert checkpointed.model.depth_encoder[0].weight.grad.abs().sum() > 0
    assert checkpointed.model.gru.weight_hh.grad.abs().sum() > 0


def test_visual_checkpoint_reduces_saved_activation_bytes(monkeypatch):
    import rsl_rl.modules.actor_critic_pie as network
    model = PIEActorCritic(ModelConfig()).train()
    obs = {"proprio_history": torch.zeros(4, 10, 45),
           "depth": torch.randn(4, 2, 60, 80)}
    parameter_storages = {parameter.untyped_storage().data_ptr()
                          for parameter in model.parameters()}

    def retained_bytes():
        saved = {}

        def pack(tensor):
            storage = tensor.untyped_storage()
            # Parameters already exist, and views can share one allocation.
            # Count activation storage rather than repeated references to weights.
            if storage.data_ptr() not in parameter_storages:
                saved[storage.data_ptr()] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            model.encode(obs, model.initial_state(4))
        return sum(saved.values())

    def direct_call(function, *args, **kwargs):
        return function(*args)

    with monkeypatch.context() as plain:
        plain.setattr(network, "checkpoint", direct_call)
        before = retained_bytes()
    after = retained_bytes()
    assert after < before * 0.6
