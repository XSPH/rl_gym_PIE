"""CPU equivalence checks for lossless depth storage and differentiable reuse."""
from copy import deepcopy
import json

import pytest
import torch

from rsl_rl.algorithms.ppo_pie import PIEPPO, PPOConfig, clone_observation
from rsl_rl.modules.actor_critic_pie import ModelConfig, PIEActorCritic, PIEDepthFeatureCache
from rsl_rl.storage.rollout_storage_pie import DepthFramePool, PIERolloutStorage
from native_rsl_helpers import collect_native


@pytest.fixture(scope="module", autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def precision(request):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    try:
        yield request.param
    finally:
        torch.set_default_dtype(previous)


def _model():
    cfg = ModelConfig(
        proprio_dim=3, proprio_history=2, depth_history=2, action_dim=2,
        heightmap_dim=3, critic_dim=4, token_dim=8, gru_dim=8,
        latent_dim=2, map_latent_dim=2, transformer_heads=2,
        proprio_hidden_dims=(8,), cnn_hidden_channels=(2, 2),
        cnn_kernel_sizes=(3, 3, 3), cnn_strides=(1, 1, 1),
        cnn_paddings=(1, 1, 1), visual_grid=(1, 1),
        actor_hidden_dims=(8,), critic_hidden_dims=(8,),
        successor_hidden_dims=(8,), height_decoder_hidden_dims=(8,))
    return PIEActorCritic(cfg).cpu()


def _sequence(steps=24, count=3, shape=(8, 8)):
    """10 Hz images, distinct channels, and asynchronous episode resets."""
    frame_ids = torch.arange(count, dtype=torch.long).repeat(2, 1).t().clone()
    serial = 1
    grid = torch.linspace(-0.1, 0.1, shape[0] * shape[1]).reshape(shape)
    observations, resets = [], []
    for step in range(steps + 1):
        reset = torch.zeros(count, dtype=torch.bool)
        if step == 0:
            reset[:] = True
        if step and step % 5 == 0:
            frame_ids[:, 0] = frame_ids[:, 1]
            frame_ids[:, 1] = serial * count + torch.arange(count)
            serial += 1
        partial = {7: 0, 13: 1, 24: 2}.get(step)
        if partial is not None and partial < count:
            frame_ids[partial] = serial * count + partial
            serial += 1
            reset[partial] = True
        prop = torch.sin(torch.arange(count * 3).reshape(count, 3) / 7 + step / 6)
        observations.append({
            "proprio": prop,
            "proprio_history": torch.stack((prop - 0.1, prop), 1),
            "depth": frame_ids[:, :, None, None].float() / 30 + grid,
            "depth_frame_ids": frame_ids.clone(),
            "critic": torch.cat((prop, prop[:, :1] * 0.5), -1),
            "targets": {"velocity": prop * 0.1,
                        "foot_clearance": torch.zeros(count, 4),
                        "heightmap": prop * 0.2},
        })
        resets.append(reset)
    return observations, resets


def _legacy(obs):
    return {key: value for key, value in obs.items() if key != "depth_frame_ids"}


class _RecordedEnvironment:
    def __init__(self, indexed, steps=24):
        self.observations, self.resets = _sequence(steps)
        self.indexed = indexed
        self.step_index = 0
        self.num_envs, self.num_obs, self.num_privileged_obs, self.num_actions = 3, 3, 4, 2

    def get_pie_observations(self):
        return self.observation()

    def observation(self):
        obs = self.observations[self.step_index]
        return obs if self.indexed else _legacy(obs)

    def step(self, actions):
        self.step_index += 1
        obs = self.observation()
        terminated = torch.zeros(3, dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        terminated[0] = self.step_index == 7
        truncated[1] = self.step_index == 13
        terminated[2] = self.step_index == 24
        terminal = deepcopy(obs)
        # Pre-reset state differs from the returned autoreset observation.
        terminal["proprio"] += (terminated | truncated)[:, None] * 2
        terminal["critic"] += (terminated | truncated)[:, None] * 3
        return obs["proprio"], obs["critic"], torch.tensor([1.0, 0.5, -0.1]), terminated | truncated, {
            "time_outs": truncated,
            "pie": {"terminal_proprio": terminal["proprio"],
                    "terminal_critic": terminal["critic"]},
        }


def test_frame_pool_reconstructs_both_channels_and_owns_immutable_images():
    observations, _ = _sequence()
    pool = DepthFramePool()
    stored, expected = [], []
    for obs in observations:
        expected.append(obs["depth"].clone())
        stored.append(clone_observation(obs, pool))
        # Simulate reusable environment buffers being overwritten immediately.
        obs["depth"].fill_(-999)
        obs["depth_frame_ids"].fill_(-999)
    images = pool.materialize()
    assert images.dtype == torch.float32 and not images.requires_grad
    assert images.shape[0] < len(stored) * 3 * 2
    assert all("depth" not in obs and "depth_frame_ids" not in obs for obs in stored)
    for obs, original in zip(stored, expected):
        recovered = PIERolloutStorage.observation(
            {"obs": obs}, depth_frames=images, restore_depth=True)
        torch.testing.assert_close(recovered["depth"], original, rtol=0, atol=0)
        subset = PIERolloutStorage.observation(
            {"obs": obs}, torch.tensor([2, 0]), images, restore_depth=True)
        torch.testing.assert_close(subset["depth"], original[[2, 0]], rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="sealed"):
        pool.add(expected[0], torch.arange(3).repeat(2, 1).t())


def test_async_visual_reuse_preserves_actions_hidden_probabilities_and_gradients():
    torch.manual_seed(42)
    reference = _model().train()
    cached = deepcopy(reference)
    observations, resets = _sequence()
    cache = PIEDepthFeatureCache()
    hidden_ref = reference.initial_state(3)
    hidden_cached = cached.initial_state(3)
    losses_ref, losses_cached = [], []
    for obs, reset in zip(observations[:24], resets[:24]):
        action = reference.policy_distribution(_legacy(obs), hidden_ref, reset)[0].mean.detach()
        plain = reference.evaluate_actions(_legacy(obs), hidden_ref, action, reset)
        visual = cache.get(cached, obs)
        reused = cached.evaluate_actions(obs, hidden_cached, action, reset, visual_features=visual)
        for actual, expected in zip(reused[:4], plain[:4]):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        for key in plain[4]:
            torch.testing.assert_close(reused[4][key], plain[4][key], rtol=1e-5, atol=1e-7)
        hidden_ref, hidden_cached = plain[3], reused[3]
        losses_ref.append(sum(value.square().mean() for value in plain[:4]))
        losses_cached.append(sum(value.square().mean() for value in reused[:4]))
    sum(losses_ref).backward()
    sum(losses_cached).backward()
    for (name, param_ref), (_, param_cached) in zip(
            reference.named_parameters(), cached.named_parameters()):
        if param_ref.grad is None:
            assert param_cached.grad is None, name
        else:
            torch.testing.assert_close(param_cached.grad, param_ref.grad,
                                       rtol=2e-5, atol=2e-6, msg=name)
    assert cached.depth_encoder[0].weight.grad.abs().sum() > 0
    assert cached.gru.weight_hh.grad.abs().sum() > 0
    assert cache.encoded_stacks < 24 * 3
    # The recurrent state changes even when every image ID is unchanged.
    assert torch.all(observations[1]["depth_frame_ids"] == observations[0]["depth_frame_ids"])
    assert not torch.equal(observations[1]["proprio"], observations[0]["proprio"])


@pytest.mark.parametrize("precision", [torch.float32, torch.float64], indirect=True)
def test_indexed_multi_epoch_ppo_matches_legacy_full_recurrent_updates(precision):
    torch.manual_seed(51)
    reference_model = _model()
    indexed_model = deepcopy(reference_model)
    cfg = PPOConfig(epochs=3, minibatches=2, schedule="fixed")
    plain = PIEPPO(reference_model, device="cpu", **cfg.as_native_kwargs())
    indexed = PIEPPO(indexed_model, device="cpu", **cfg.as_native_kwargs())
    batches, finals = [], []
    for algorithm, use_indices in ((plain, False), (indexed, True)):
        env = _RecordedEnvironment(use_indices)
        torch.manual_seed(71)
        batch, _ = collect_native(algorithm, env, steps=24)
        batches.append(batch)
        finals.append(algorithm.model.get_hidden_states()[0].clone())
    for key in ("advantages", "returns"):
        torch.testing.assert_close(batches[1][key], batches[0][key], rtol=1e-5, atol=1e-6)
    for legacy_frame, indexed_frame in zip(batches[0]["frames"], batches[1]["frames"]):
        for key in ("actions", "old_logp", "values", "rewards", "successor"):
            torch.testing.assert_close(indexed_frame[key], legacy_frame[key],
                                       rtol=1e-5, atol=1e-6)
        restored = PIERolloutStorage.observation(
            indexed_frame, depth_frames=batches[1]["depth_frames"], restore_depth=True)
        torch.testing.assert_close(restored["depth"], legacy_frame["obs"]["depth"],
                                   rtol=0, atol=0)
    reports = []
    for algorithm, batch in zip((plain, indexed), batches):
        torch.manual_seed(91)
        algorithm.update()
        reports.append(dict(algorithm.metrics))
    for key in ("velocity", "foot_clearance", "heightmap", "successor", "kl",
                "loss", "policy", "value", "grad_norm", "policy_kl"):
        assert reports[1][key] == pytest.approx(reports[0][key], rel=2e-5, abs=2e-6), key
    for (name, expected), (_, actual) in zip(
            reference_model.named_parameters(), indexed_model.named_parameters()):
        if precision == torch.float32 and name.endswith("self_attn.in_proj_bias"):
            # The attention key bias cancels in softmax and has a near-zero
            # gradient. FP32 accumulation order produces tiny residuals that
            # Adam amplifies relative to epsilon. Only this slice gets a
            # separate tolerance; all other parameters keep the strict check.
            torch.testing.assert_close(actual[:8], expected[:8], rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(actual[16:], expected[16:], rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(actual[8:16], expected[8:16], rtol=2e-5, atol=1e-5)
            print("FP32 attention key-bias max parameter difference:",
                  (actual[8:16] - expected[8:16]).abs().max().item(),
                  "max raw gradient:", expected.grad[8:16].abs().max().item(),
                  "max gradient difference:",
                  (actual.grad[8:16] - expected.grad[8:16]).abs().max().item())
        else:
            torch.testing.assert_close(actual, expected,
                                       rtol=1e-9 if precision == torch.float64 else 2e-5,
                                       atol=1e-11 if precision == torch.float64 else 2e-6,
                                       msg=name)
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad,
                                       rtol=1e-9 if precision == torch.float64 else 3e-5,
                                       atol=1e-11 if precision == torch.float64 else 2e-6,
                                       msg=name)
    assert reports[1]["cnn_encoded_stacks"] < reports[1]["cnn_dense_stacks"]
    assert reports[0]["cnn_encoded_stacks"] == reports[0]["cnn_dense_stacks"]
    assert indexed.model.depth_encoder[0].weight.grad.abs().sum() > 0
    assert indexed.model.gru.weight_hh.grad.abs().sum() > 0

    # refresh_hidden must use newly updated CNN weights, not collection features.
    refreshed = indexed.model.get_hidden_states()[0]
    mask = batches[1]["frames"][-1]["terminated"] | batches[1]["frames"][-1]["truncated"]
    with torch.no_grad():
        expected_hidden = batches[1]["hidden"].clone()
        for frame in batches[1]["frames"]:
            obs = PIERolloutStorage.observation(
                frame, depth_frames=batches[1]["depth_frames"], restore_depth=True)
            _, expected_hidden = indexed.model.encode(obs, expected_hidden, frame["reset"])
        expected_hidden[2] = 0
    torch.testing.assert_close(refreshed, expected_hidden, rtol=1e-5, atol=1e-6)
    assert mask.tolist() == [False, False, True]
    assert not refreshed.requires_grad
    assert not torch.equal(refreshed[1], finals[1][1])
    legacy_refreshed = plain.model.get_hidden_states()[0]
    legacy_mask = batches[0]["frames"][-1]["terminated"] | batches[0]["frames"][-1]["truncated"]
    torch.testing.assert_close(refreshed, legacy_refreshed, rtol=1e-5, atol=1e-6)
    assert torch.equal(mask, legacy_mask)


def test_reused_24_step_graph_retains_fewer_activations_and_cnn_inputs(monkeypatch):
    import rsl_rl.modules.actor_critic_pie as network

    torch.manual_seed(63)
    model = PIEActorCritic(ModelConfig()).cpu().train()
    observations, resets = _sequence(count=4, shape=(60, 80))
    for step, obs in enumerate(observations):
        history = torch.cos(torch.arange(4 * 10 * 45).reshape(4, 10, 45) / 37 + step / 9)
        obs["proprio_history"] = history
        obs["proprio"] = history[:, -1].clone()
        obs["critic"] = torch.zeros(4, 235)
    pool = DepthFramePool()
    indexed_frames = [{"obs": clone_observation(obs, pool)} for obs in observations[:24]]
    depth_frames = pool.materialize()
    parameter_storages = {parameter.untyped_storage().data_ptr()
                          for parameter in model.parameters()}
    measures = {}

    def measure(reuse, count=4):
        saved = {}
        references = []
        cache = PIEDepthFeatureCache()
        hidden = model.initial_state(count)
        ids = torch.arange(count)

        def pack(tensor):
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in parameter_storages:
                saved[storage.data_ptr()] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            for raw_obs, frame, reset in zip(observations[:24], indexed_frames, resets[:24]):
                if reuse:
                    obs = PIERolloutStorage.observation(frame, ids)
                    visual = cache.get(model, obs, depth_frames)
                else:
                    # Match the old trajectory update's environment indexing.
                    obs = {key: raw_obs[key][ids]
                           for key in ("proprio_history", "depth")}
                    visual = None
                estimates, hidden = model.encode(obs, hidden, reset[ids], visual_features=visual)
                references.append(hidden.square().mean() + estimates["velocity"].square().mean())
            loss = sum(references)
        result = {"saved_activation_bytes": sum(saved.values()),
                  "cnn_input_stacks": cache.encoded_stacks if reuse else 24 * count}
        # Verify all 24 steps really share a live, differentiable graph.
        model.zero_grad(set_to_none=True)
        loss.backward()
        assert model.depth_encoder[0].weight.grad.abs().sum() > 0
        assert model.gru.weight_hh.grad.abs().sum() > 0
        return result

    measures["dense_checkpoint"] = measure(False)
    measures["reuse_checkpoint"] = measure(True)
    measures["reuse_checkpoint_quarter_envs"] = measure(True, count=1)
    with monkeypatch.context() as direct:
        direct.setattr(network, "checkpoint", lambda function, *args, **kwargs: function(*args))
        measures["dense_no_checkpoint"] = measure(False)
        measures["reuse_no_checkpoint"] = measure(True)
        measures["reuse_no_checkpoint_quarter_envs"] = measure(True, count=1)
    print("PIE 24-step CPU retained tensors:", json.dumps(measures, sort_keys=True))
    assert measures["reuse_checkpoint"]["cnn_input_stacks"] < measures["dense_checkpoint"]["cnn_input_stacks"]
    assert measures["reuse_checkpoint"]["saved_activation_bytes"] < measures["dense_checkpoint"]["saved_activation_bytes"]
    assert measures["reuse_no_checkpoint"]["saved_activation_bytes"] < measures["dense_no_checkpoint"]["saved_activation_bytes"]
