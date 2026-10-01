"""CPU checks for observation privacy, recurrent replay, and time limits."""
import torch
from legged_gym.pie.models import ModelConfig, PIEActorCritic
from legged_gym.pie.learner import PPO, PPOConfig, gae

def make_obs(batch=2):
    torch.manual_seed(5)
    return {"proprio": torch.randn(batch, 45),
            "proprio_history": torch.randn(batch, 10, 45),
            "depth": torch.rand(batch, 2, 24, 32),
            "critic": torch.randn(batch, 59),
            "targets": {"velocity": torch.randn(batch, 3),
                        "foot_clearance": torch.rand(batch, 4),
                        "heightmap": torch.randn(batch, 11)}}

def make_model():
    return PIEActorCritic(ModelConfig(heightmap_dim=11, critic_dim=59,
                                     token_dim=16, gru_dim=16,
                                     latent_dim=4, map_latent_dim=8))

def test_privileged_targets_cannot_change_actor():
    model = make_model().eval()
    obs = make_obs(); hidden = model.initial_state(2)
    action1 = model.act(obs, hidden, deterministic=True)[0]
    obs["critic"].add_(1000)
    for target in obs["targets"].values():
        target.add_(1000)
    action2 = model.act(obs, hidden, deterministic=True)[0]
    torch.testing.assert_close(action1, action2, rtol=0, atol=0)

def test_recurrent_reset_and_probability_replay():
    model = make_model().eval(); obs = make_obs()
    hidden = model.initial_state(2)
    action, old_logp, _, _ = model.act(obs, hidden)
    new_logp = model.evaluate(obs, hidden, action)[0]
    torch.testing.assert_close(old_logp, new_logp, rtol=0, atol=0)
    dirty = torch.randn_like(hidden)
    reset = torch.ones(2, dtype=torch.bool)
    with_reset = model.act(obs, dirty, reset, deterministic=True)[0]
    fresh = model.act(obs, hidden, deterministic=True)[0]
    torch.testing.assert_close(with_reset, fresh, rtol=0, atol=0)

def test_time_limits_bootstrap_without_cross_episode_gae():
    rewards = torch.tensor([[1., 1.], [100., 100.]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[10., 10.], [0., 0.]])
    terminated = torch.tensor([[True, False], [False, False]])
    truncated = torch.tensor([[False, True], [False, False]])
    advantage, _ = gae(rewards, values, next_values, terminated, truncated, .9, .95)
    torch.testing.assert_close(advantage[0], torch.tensor([1., 10.]))

class ResetEnvironment:
    def __init__(self):
        self.obs = make_obs(); self.steps = 0
    def reset(self):
        return self.obs
    def step(self, actions):
        self.steps += 1
        final = make_obs()
        final["proprio"].fill_(7)
        final["critic"].fill_(7)
        reset_obs = make_obs(); reset_obs["proprio"].fill_(-99)
        reset_obs["critic"].fill_(-99)
        done = torch.ones(2, dtype=torch.bool)
        return reset_obs, torch.ones(2), ~done, done, {
            "terminal_proprio": final["proprio"],
            "terminal_observation": final}

def test_rollout_uses_pre_reset_successor_and_joint_loss_updates():
    torch.set_num_threads(1)
    model = make_model(); env = ResetEnvironment()
    algorithm = PPO(model, PPOConfig(epochs=1, minibatches=1))
    batch, *_ = algorithm.collect(env, env.reset(), model.initial_state(2),
                                 torch.ones(2, dtype=torch.bool), 3)
    assert all(torch.all(frame["successor"] == 7) for frame in batch["frames"])
    with torch.no_grad():
        final_obs = make_obs(); final_obs["critic"].fill_(7)
        expected_value = model.value(final_obs)
        reset_obs = make_obs(); reset_obs["critic"].fill_(-99)
        reset_value = model.value(reset_obs)
    assert not torch.equal(expected_value, reset_value)
    for frame in batch["frames"]:
        torch.testing.assert_close(frame["next_values"], expected_value)
    before_actor = model.actor[0].weight.detach().clone()
    before_encoder = model.depth_encoder[0].weight.detach().clone()
    report = algorithm.update(batch)
    assert all(torch.isfinite(torch.tensor(x)) for x in report.values())
    assert not torch.equal(before_actor, model.actor[0].weight)
    assert not torch.equal(before_encoder, model.depth_encoder[0].weight)

def test_invalid_successor_is_masked_without_nan():
    model = make_model(); obs = make_obs()
    _, _, est = model.policy_distribution(obs, model.initial_state(2))
    successor = obs["proprio"].clone(); successor[0].fill_(float("nan"))
    losses = model.auxiliary_losses(est, obs["targets"], successor, torch.tensor([False, True]))
    assert all(torch.isfinite(loss) for loss in losses.values())

def test_invalid_successor_keeps_current_labels_and_gradients():
    model = make_model(); obs = make_obs()
    _, _, estimates = model.policy_distribution(obs, model.initial_state(2))
    successor = torch.full_like(obs["proprio"], float("nan"))
    losses = model.auxiliary_losses(estimates, obs["targets"], successor,
                                   successor_valid=torch.zeros(2, dtype=torch.bool))
    assert losses["successor"].item() == 0
    assert all(torch.isfinite(loss) for loss in losses.values())
    sum(losses.values()).backward()
    assert model.velocity_head.weight.grad.abs().sum() > 0
    assert model.clearance_head.weight.grad.abs().sum() > 0
    assert model.height_decoder[0].weight.grad.abs().sum() > 0

def test_policy_gradients_train_estimator_without_privileged_supervision():
    model = make_model(); obs = make_obs()
    distribution, _, _ = model.policy_distribution(obs, model.initial_state(2))
    actions = distribution.mean.detach() + 0.1
    distribution.log_prob(actions).sum().backward()
    assert model.depth_encoder[0].weight.grad.abs().sum() > 0
    assert model.gru.weight_ih.grad.abs().sum() > 0
    assert model.velocity_head.weight.grad.abs().sum() > 0
    assert model.map_head.weight.grad.abs().sum() > 0
    assert model.mu_head.weight.grad.abs().sum() > 0
    # Posterior variance is trained by reconstruction/KL, not policy sampling.
    assert model.logvar_head.weight.grad is None
    assert model.critic[0].weight.grad is None

class MixedResetEnvironment:
    def __init__(self):
        self.steps = 0
    def reset(self):
        return make_obs(3)
    def step(self, actions):
        self.steps += 1
        obs = make_obs(3)
        terminated = torch.zeros(3, dtype=torch.bool)
        truncated = torch.zeros(3, dtype=torch.bool)
        if self.steps == 2:
            terminated[0] = True
        if self.steps == 3:
            truncated[1] = True
        return obs, torch.ones(3), terminated, truncated, {
            "terminal_observation": obs, "terminal_proprio": obs["proprio"]}

def test_updated_weight_replay_preserves_continuing_episode_memory():
    torch.set_num_threads(1)
    model = make_model(); env = MixedResetEnvironment()
    algorithm = PPO(model, PPOConfig(epochs=1, minibatches=1))
    batch, _, old_hidden, _ = algorithm.collect(
        env, env.reset(), model.initial_state(3),
        torch.ones(3, dtype=torch.bool), 3)
    algorithm.update(batch)
    refreshed, reset_mask = algorithm.refresh_hidden(batch)
    with torch.no_grad():
        expected = batch["hidden"].clone()
        for frame in batch["frames"]:
            _, expected = model.encode(frame["obs"], expected, frame["reset"])
        expected[1] = 0
    torch.testing.assert_close(refreshed, expected)
    assert reset_mask.tolist() == [False, True, False]
    assert torch.all(refreshed[1] == 0)
    assert refreshed[0].norm() > 0 and refreshed[2].norm() > 0
    assert not torch.equal(refreshed[2], old_hidden[2])
    assert not refreshed.requires_grad
