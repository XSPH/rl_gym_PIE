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
        return {"loss": loss.item()}

    def refresh_hidden(self, batch):
        return torch.zeros(1, 1), torch.zeros(1, dtype=torch.bool)


def _runner(output):
    runner = PIEOnPolicyRunner.__new__(PIEOnPolicyRunner)
    runner.pie_cfg = PIERunnerCfg(save_interval=500)
    runner.save_interval = 500
    runner.env = SimpleNamespace()
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
