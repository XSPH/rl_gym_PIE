"""Checkpoint scheduling and recoverable file writes, without a simulator."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rsl_rl.runners.on_policy_runner_pie import PIERunnerCfg, PIEOnPolicyRunner


class _Algorithm:
    def __init__(self):
        self.actor_critic = torch.nn.Linear(1, 1)
        self.actor_critic.std = torch.ones(1)
        torch.nn.init.zeros_(self.actor_critic.weight)
        torch.nn.init.zeros_(self.actor_critic.bias)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=0.1)
        self.learning_rate = 0.1

    def collect(self, env, obs, hidden, reset_mask, steps):
        batch = {"frames": [{"rewards": torch.ones(1),
                             "terminated": torch.zeros(1, dtype=torch.bool),
                             "truncated": torch.zeros(1, dtype=torch.bool)} for _ in range(steps)]}
        return batch, obs, hidden, reset_mask

    def update(self, batch):
        self.optimizer.zero_grad()
        loss = (self.actor_critic(torch.ones(1, 1)) - 1).square().mean()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(), "value": 0.0, "policy": 0.0}

    def refresh_hidden(self, batch):
        return torch.zeros(1, 1), torch.zeros(1, dtype=torch.bool)


def _runner(output):
    runner = PIEOnPolicyRunner.__new__(PIEOnPolicyRunner)
    runner.pie_cfg = PIERunnerCfg(save_interval=500)
    runner.save_interval = 500
    runner.env = SimpleNamespace(num_envs=1)
    runner.device = "cpu"
    runner.alg = _Algorithm()
    runner.log_dir = str(output)
    runner.num_steps_per_env = 1
    runner.current_learning_iteration = 499
    runner.tot_timesteps = 0
    runner.tot_time = 0
    runner.writer = None
    runner._observation = {}
    runner._hidden = torch.zeros(1, 1)
    runner._reset_mask = torch.ones(1, dtype=torch.bool)
    runner.env.levels = torch.tensor([0, 1, 1, 2])
    return runner


def test_checkpoint_records_boundary_weights_and_cumulative_iteration(tmp_path):
    runner = _runner(tmp_path)
    runner.learn(2)
    boundary = torch.load(tmp_path / "model_500.pt", weights_only=True)
    assert boundary["iter"] == boundary["iterations"] == 500
    assert boundary["rsl_rl_base"] == "v1.0.2"
    assert all(value["step"].item() == 1 for value in boundary["optimizer_state_dict"]["state"].values())
    boundary_policy = torch.nn.Linear(1, 1)
    boundary_policy.load_state_dict(boundary["model_state_dict"])

    runner.learn(1)
    final = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert final["iter"] == 502
    assert all(value["step"].item() == 3 for value in final["optimizer_state_dict"]["state"].values())
    final_policy = torch.nn.Linear(1, 1)
    final_policy.load_state_dict(final["model_state_dict"])
    assert not torch.equal(boundary_policy(torch.ones(1, 1)), final_policy(torch.ones(1, 1)))
    assert torch.equal(final_policy(torch.ones(1, 1)), runner.alg.actor_critic(torch.ones(1, 1)))
    assert sorted(path.name for path in tmp_path.glob("model_*.pt")) == ["model_500.pt"]
    assert not list(tmp_path.glob("*.tmp"))
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [item["iteration"] for item in metrics] == [500, 501, 502]
    assert all(item["terrain_level"] == 1.0 for item in metrics)
    assert all(item["terrain_level_min"] == 0 and item["terrain_level_max"] == 2 for item in metrics)


def test_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    runner.save(checkpoint)
    previous = checkpoint.read_bytes()

    def interrupted_save(state, path):
        Path(path).write_bytes(b"incomplete checkpoint")
        raise OSError("interrupted write")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted write"):
        runner.save(checkpoint)
    assert checkpoint.read_bytes() == previous
    assert not (tmp_path / "checkpoint.pt.tmp").exists()


def test_episode_statistics_cross_rollouts_and_reset_only_finished_envs(tmp_path):
    runner = _runner(tmp_path)
    def frame(rewards, terminated, truncated):
        return {"rewards": torch.tensor(rewards),
                "terminated": torch.tensor(terminated), "truncated": torch.tensor(truncated)}
    runner._update_episode_statistics([
        frame([1.0, 2.0], [False, True], [False, False]),
        frame([3.0, 4.0], [False, False], [True, False]),
    ])
    assert list(runner._reward_buffer) == [2.0, 4.0]
    assert list(runner._length_buffer) == [1, 2]
    runner._update_episode_statistics([frame([5.0, 6.0], [False, True], [False, False])])
    assert list(runner._reward_buffer) == [2.0, 4.0, 10.0]
    assert list(runner._length_buffer) == [1, 2, 2]
    assert runner._episode_reward_sum.tolist() == [5.0, 0.0]


def test_reward_terms_use_native_episode_scaling_and_keep_continuing_envs(tmp_path):
    runner = _runner(tmp_path)
    runner.env.config = SimpleNamespace(policy_dt=0.02, episode_seconds=20.0, reward_scale_dt=True)
    def frame(tracking, collision, terminated, truncated):
        tracking, collision = torch.tensor(tracking), torch.tensor(collision)
        return {"rewards": (tracking+collision)*0.02,
                "terminated": torch.tensor(terminated), "truncated": torch.tensor(truncated),
                "reward_terms": {"tracking_linear": tracking, "collision": collision}}
    infos = runner._update_episode_statistics([
        frame([10.0, 20.0], [-2.0, -4.0], [False, True], [False, False]),
        frame([30.0, 40.0], [-6.0, -8.0], [False, False], [True, False]),
    ])
    assert len(infos) == 2
    assert infos[0]["rew_tracking_lin_vel"].item() == pytest.approx(0.02)
    assert infos[0]["rew_collision"].item() == pytest.approx(-0.004)
    assert infos[1]["rew_tracking_lin_vel"].item() == pytest.approx(0.04)
    assert infos[1]["rew_collision"].item() == pytest.approx(-0.008)
    infos = runner._update_episode_statistics([
        frame([50.0, 60.0], [-10.0, -12.0], [False, True], [False, False])])
    assert infos[0]["rew_tracking_lin_vel"].item() == pytest.approx(0.1)
    assert infos[0]["rew_collision"].item() == pytest.approx(-0.02)
    assert runner._reward_term_sums["tracking_linear"].tolist() == [1.0, 0.0]


def test_original_logger_prints_and_records_each_reward_without_double_counting(tmp_path, capsys):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter
    runner = _runner(tmp_path)
    runner.current_learning_iteration = 1
    runner.writer = SummaryWriter(log_dir=str(tmp_path))
    infos = runner._update_episode_statistics([{
        "rewards": torch.ones(1), "terminated": torch.ones(1, dtype=torch.bool),
        "truncated": torch.zeros(1, dtype=torch.bool),
        "reward_terms": {"tracking_linear": torch.tensor([2.0]), "collision": torch.tensor([-1.0])}}])
    metrics = {"iteration": 1, "value": 0.2, "policy": -0.1, "loss": 0.3,
               "collection_time": 0.3, "learning_time": 0.1, "mean_reward": 1.0,
               "velocity": 0.01, "foot_clearance": 0.02, "heightmap": 0.03,
               "successor": 0.04, "kl": 0.05, "policy_kl": 0.007, "grad_norm": 2.0,
               "learning_rate": 0.1, "transitions": 1,
               "terrain_level": 1.0, "terrain_level_min": 0, "terrain_level_max": 2}
    runner._log_pie_iteration(metrics, 15000, infos)
    text = capsys.readouterr().out
    assert "Learning iteration 0/15000" in text
    assert "Mean episode rew_tracking_lin_vel:" in text
    assert "Mean episode rew_collision:" in text
    assert "Surrogate loss:" in text
    for label in ("Total loss:", "Velocity estimation loss:", "Foot clearance loss:",
                  "Height map reconstruction loss:", "Successor reconstruction loss:",
                  "VAE KL loss:", "Policy KL divergence:", "Gradient norm before clipping:", "Learning rate:",
                  "Mean step reward:", "Transitions this iteration:",
                  "Mean terrain level:", "Min terrain level:", "Max terrain level:"):
        assert label in text
    assert text.index("Mean episode rew_collision:") < text.index("Velocity estimation loss:")
    assert text.index("Velocity estimation loss:") < text.index("Total timesteps:")
    assert runner.current_learning_iteration == 1
    assert runner.tot_timesteps == 1
    assert runner.tot_time == pytest.approx(0.4)
    runner.writer.close()
    events = EventAccumulator(str(tmp_path)).Reload()
    assert events.Scalars("Episode/rew_tracking_lin_vel")[0].value == 2.0
    assert events.Scalars("Episode/rew_collision")[0].value == -1.0
    assert events.Scalars("Train/mean_reward")[0].value == 1.0
    assert events.Scalars("Loss/value_function")[0].step == 0
    assert events.Scalars("PIE/velocity_loss")[0].value == pytest.approx(0.01)
    assert events.Scalars("PIE/vae_kl_loss")[0].value == pytest.approx(0.05)
    assert events.Scalars("PIE/policy_kl")[0].value == pytest.approx(0.007)
    assert events.Scalars("Episode/terrain_level")[0].value == 1.0
    assert events.Scalars("Terrain/min_level")[0].value == 0.0
    assert events.Scalars("Terrain/max_level")[0].value == 2.0


def test_resume_restores_adam_lr_iteration_and_learned_std_with_fresh_episodes(tmp_path):
    from dataclasses import replace
    from rsl_rl.algorithms.ppo_pie import PPOConfig
    from test_pie_ppo_schedule import _rollout, _TensorObservations

    torch.set_num_threads(1)
    algorithm, batch = _rollout(PPOConfig(epochs=1, minibatches=1))
    inputs = _TensorObservations()
    algorithm.update(batch)
    with torch.no_grad():
        algorithm.model.std.fill_(0.37)
    for group in algorithm.optimizer.param_groups:
        group["lr"] = 7e-4
    runner = _runner(tmp_path)
    runner.alg = algorithm
    runner.pie_cfg.model = replace(algorithm.model.cfg, initial_std=0.5)
    runner.current_learning_iteration = 500
    runner.tot_timesteps, runner.tot_time = 987, 4.0
    checkpoint = tmp_path / "resume.pt"
    runner.save(checkpoint)

    env = SimpleNamespace(device="cpu", num_envs=inputs.count, num_obs=3,
                          num_privileged_obs=4, num_actions=2,
                          config=SimpleNamespace(policy_dt=0.02, episode_seconds=20.0,
                                                 reward_scale_dt=True),
                          reset=inputs.observation, step=inputs.step)
    resumed = PIEOnPolicyRunner(env, PIERunnerCfg(
        model=replace(algorithm.model.cfg, initial_std=1.0),
        ppo=PPOConfig(epochs=1, minibatches=1), num_steps_per_env=3),
        log_dir=None, device="cpu")
    resumed._episode_reward_sum = torch.ones(inputs.count)
    resumed.load(checkpoint)
    assert resumed.current_learning_iteration == 500
    assert resumed.tot_timesteps == 987 and resumed.tot_time == 4.0
    assert resumed.alg.learning_rate == pytest.approx(7e-4)
    assert resumed.alg.optimizer.param_groups[0]["lr"] == pytest.approx(7e-4)
    torch.testing.assert_close(resumed.alg.model.std, torch.full((2,), 0.37))
    for key, value in algorithm.model.state_dict().items():
        torch.testing.assert_close(resumed.alg.model.state_dict()[key], value)
    actual = resumed.alg.optimizer.state_dict()["state"]
    for parameter, moments in algorithm.optimizer.state_dict()["state"].items():
        for name, value in moments.items():
            torch.testing.assert_close(actual[parameter][name], value)
    assert torch.count_nonzero(resumed._hidden) == 0
    assert resumed._reset_mask.all()
    assert not hasattr(resumed, "_episode_reward_sum")
    report = resumed.learn(1)
    assert report["iteration"] == 501
    assert resumed.tot_timesteps == 987 + inputs.count * 3


def test_resume_rejects_optimizer_from_old_parameterization(tmp_path):
    runner = _runner(tmp_path)
    checkpoint = tmp_path / "old.pt"
    runner.save(checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    saved["rsl_rl_base"] = "2.2.4"
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="requires a v1.0.2 PIE checkpoint"):
        runner.load(checkpoint)
