"""New-schema checkpoint transactions and real native runner collection on CPU."""
import json
from pathlib import Path

import pytest
import torch

from native_rsl_helpers import algorithm_config, TensorEnvironment, train_config
from rsl_rl.runners.on_policy_runner_pie import PIEOnPolicyRunner


@pytest.fixture(scope="module", autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def runner(output=None, rollout=3):
    return PIEOnPolicyRunner(TensorEnvironment(), train_config(rollout=rollout),
                             log_dir=None if output is None else str(output), device="cpu")


def test_native_learning_loop_orders_act_step_process_returns_update(tmp_path, monkeypatch):
    learner = runner(tmp_path, rollout=2)
    events = []
    logged_views = []
    metrics = learner._iteration_metrics
    def collect_metrics(view):
        logged_views.append(set(view))
        return metrics(view)
    monkeypatch.setattr(learner, "_iteration_metrics", collect_metrics)
    for obj, name in ((learner.alg, "act"), (learner.env, "step"),
                      (learner.alg, "process_env_step"),
                      (learner.alg, "compute_returns"), (learner.alg, "update")):
        original = getattr(obj, name)
        def call(*args, _original=original, _name=name, **kwargs):
            events.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(obj, name, call)
    report = learner.learn(2)
    assert events == (["act", "step", "process_env_step"] * 2
                      + ["compute_returns", "update"]) * 2
    assert learner.current_learning_iteration == 2
    assert learner.tot_timesteps == 12
    assert report["iteration"] == 2 and report["transitions"] == 6
    assert all(not {"locs", "obs", "critic_obs"} & keys for keys in logged_views)
    learner.writer.close()


def test_boundary_checkpoint_matches_completed_updates_and_atomic_final(tmp_path):
    learner = runner(tmp_path)
    learner.current_learning_iteration = 499
    learner.learn(2)
    boundary = torch.load(tmp_path / "model_500.pt", weights_only=True)
    final = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert boundary["pie_checkpoint_version"] == final["pie_checkpoint_version"] == 4
    assert boundary["iter"] == 500 and final["iter"] == 501
    assert boundary["rsl_rl_base"] == "v1.0.2"
    assert all(value["step"].item() == 1
               for value in boundary["optimizer_state_dict"]["state"].values())
    assert all(value["step"].item() == 2
               for value in final["optimizer_state_dict"]["state"].values())
    assert any(not torch.equal(boundary["model_state_dict"][key], value)
               for key, value in final["model_state_dict"].items())
    for key, value in final["model_state_dict"].items():
        torch.testing.assert_close(value, learner.alg.actor_critic.state_dict()[key], rtol=0, atol=0)
    assert sorted(path.name for path in tmp_path.glob("model_*.pt")) == ["model_500.pt"]
    assert not list(tmp_path.glob("*.tmp"))
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [item["iteration"] for item in metrics] == [500, 501]
    learner.writer.close()


def test_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    learner = runner()
    path = tmp_path / "checkpoint.pt"
    learner.save(path)
    previous = path.read_bytes()
    def interrupted(state, file):
        Path(file).write_bytes(b"incomplete checkpoint")
        raise OSError("interrupted write")
    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError, match="interrupted write"):
        learner.save(path)
    assert path.read_bytes() == previous
    assert not (tmp_path / "checkpoint.pt.tmp").exists()


def test_schema4_resume_restores_adam_lr_iteration_and_learned_std(tmp_path):
    learner = runner(tmp_path)
    learner.learn(1)
    with torch.no_grad():
        learner.alg.actor_critic.std.fill_(.37)
    learner.alg.learning_rate = 7e-4
    for group in learner.alg.optimizer.param_groups:
        group["lr"] = 7e-4
    learner.current_learning_iteration = 500
    learner.tot_timesteps, learner.tot_time = 987, 4.
    checkpoint = tmp_path / "resume.pt"
    learner.save(checkpoint, infos={"label": "native"})
    resumed = runner()
    assert resumed.load(checkpoint) == {"label": "native"}
    assert resumed.current_learning_iteration == 500
    assert resumed.tot_timesteps == 987 and resumed.tot_time == 4.
    assert resumed.alg.learning_rate == pytest.approx(7e-4)
    assert resumed.alg.optimizer.param_groups[0]["lr"] == pytest.approx(7e-4)
    torch.testing.assert_close(resumed.alg.actor_critic.std, torch.full((2,), .37))
    for key, value in learner.alg.actor_critic.state_dict().items():
        torch.testing.assert_close(resumed.alg.actor_critic.state_dict()[key], value, rtol=0, atol=0)
    actual = resumed.alg.optimizer.state_dict()["state"]
    for parameter, moments in learner.alg.optimizer.state_dict()["state"].items():
        for name, value in moments.items():
            torch.testing.assert_close(actual[parameter][name], value, rtol=0, atol=0)
    hidden, _ = resumed.alg.actor_critic.get_hidden_states()
    assert hidden is None or torch.count_nonzero(hidden) == 0
    report = resumed.learn(1)
    assert report["iteration"] == 501
    assert resumed.tot_timesteps == 987 + 9
    learner.writer.close()


@pytest.mark.parametrize('version', [None, 1, 2, 3, 5])
@pytest.mark.parametrize('load_optimizer', [True, False])
def test_old_schema_is_rejected_explicitly(tmp_path, version, load_optimizer):
    learner = runner()
    checkpoint = tmp_path / "old.pt"
    learner.save(checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    saved["pie_checkpoint_version"] = version
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="(?i)(schema|version|checkpoint)"):
        learner.load(checkpoint, load_optimizer=load_optimizer)


def test_schema4_records_unified_config_effective_lr_and_restores_rng(tmp_path):
    from native_cpu_helpers import load_native_classes
    classes = load_native_classes()
    learner = runner()
    learner.env.cfg = classes.config()
    learner.alg.learning_rate = .0007
    learner.alg.optimizer.param_groups[0]['lr'] = .0007
    torch.manual_seed(912)
    checkpoint = tmp_path / 'schema4.pt'
    learner.save(checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved['pie_checkpoint_version'] == 4
    assert saved['learning_rate'] == .0007
    assert saved['environment_cfg'] == classes.helpers.class_to_dict(learner.env.cfg)
    assert 'pie' not in saved['environment_cfg']
    assert saved['environment_cfg']['camera']['history'] == 2
    assert saved['cuda_rng'] == []  # This test runs with CUDA disabled.
    expected = torch.rand(10)
    resumed = runner()
    resumed.env.cfg = classes.config()
    resumed.load(checkpoint)
    torch.testing.assert_close(torch.rand(10), expected, rtol=0, atol=0)
    assert resumed.alg.learning_rate == .0007


def test_checkpoint_activation_mismatch_is_rejected_before_loading_weights(tmp_path):
    learner = runner()
    checkpoint = tmp_path / 'activation.pt'
    learner.save(checkpoint)
    changed = train_config()
    changed['policy']['activation'] = 'tanh'
    resumed = PIEOnPolicyRunner(TensorEnvironment(), changed, device='cpu')
    with pytest.raises(ValueError, match='model configuration'):
        resumed.load(checkpoint)


def test_original_logs_include_native_rewards_and_pie_metrics_once(tmp_path, capsys, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    learner = runner(tmp_path)
    curriculum_metrics = {
        'terrain_level': 1.,
        'terrain_promotion_rate': .1,
        **{'cmd_' + group + '_' + field: .5
           for group in ('flat', 'omni', 'stairs', 'forward')
           for field in ('x', 'y', 'score')},
        **{'terrain_level_' + kind: .25
           for kind in ('flat', 'slope', 'stairs', 'gap', 'step', 'hurdle')},
    }
    original_step = learner.env.step
    def step_with_curriculum(actions):
        obs, critic, rewards, dones, infos = original_step(actions)
        infos['episode'].update(curriculum_metrics)
        return obs, critic, rewards, dones, infos
    monkeypatch.setattr(learner.env, 'step', step_with_curriculum)
    learner.learn(1)
    output = capsys.readouterr().out
    for label in ("Learning iteration", "Value function loss:", "Surrogate loss:",
                  "Mean action noise std:", "Mean episode rew_tracking_lin_vel:",
                  "Mean episode rew_collision:", "Velocity estimation loss:",
                  "Foot clearance loss:", "Height map reconstruction loss:",
                  "Successor reconstruction loss:", "VAE KL loss:",
                  "Policy KL divergence:", "Episode resets:", "Total timesteps:",
                  "Total loss:", "Gradient norm before clipping:", "Learning rate:",
                  "Mean episode terrain_level:"):
        assert output.count(label) == 1, label
    for key in curriculum_metrics:
        if key != 'terrain_level':
            assert 'Mean episode ' + key + ':' not in output
    for label in ("Mean step reward:", "Mean terrain level:", "Min terrain level:",
                  "Max terrain level:", "Depth frame pool (MiB):", "Equivalent dense depth (MiB):",
                  "CNN feature reuse fraction:", "Unique depth frames:", "CNN encoded stacks:",
                  "Equivalent dense CNN stacks:", "Reset failures:", "Reset timeouts:",
                  "Reset base_contact:", "Reset tilted:", "Reset nonfinite:", "Reset timeout:",
                  "Transitions this iteration:"):
        assert label not in output, label
    counts = learner.last_metrics['reset_counts']
    assert '{} failures, {} timeouts'.format(counts['failures'], counts['timeouts']) in output
    assert output.index("Mean episode rew_collision:") < output.index("Velocity estimation loss:")
    learner.writer.close()
    events = EventAccumulator(str(tmp_path)).Reload()
    assert len(events.Scalars("Episode/rew_tracking_lin_vel")) == 1
    assert len(events.Scalars("Episode/rew_collision")) == 1
    assert events.Scalars("Episode/rew_tracking_lin_vel")[0].value == pytest.approx(.7)
    assert events.Scalars("Episode/rew_collision")[0].value == pytest.approx(-.2)
    assert events.Scalars("PIE/velocity_loss")[0].value >= 0
    recorded = json.loads((tmp_path / 'metrics.jsonl').read_text())
    for key, value in curriculum_metrics.items():
        assert recorded['episode_rewards'][key] == pytest.approx(value)
        assert events.Scalars('Episode/' + key)[0].value == pytest.approx(value)
    for key, tag in (('terrain_level', 'Episode/terrain_level'),
                     ('mean_reward', 'Train/mean_step_reward'),
                     ('depth_pool_mib', 'PIE/depth_pool_mib'),
                     ('cnn_reuse_fraction', 'PIE/cnn_reuse_fraction')):
        assert recorded[key] == learner.last_metrics[key]
        assert events.Scalars(tag)[0].value == pytest.approx(recorded[key])
    assert recorded['reset_counts'] == counts
    assert events.Scalars('Reset/failures')[0].value == counts['failures']
    assert learner.tot_timesteps == 9


def test_stock_runner_still_trains_tensor_observations_without_pie_sensor_hooks(tmp_path):
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner
    cfg = train_config(rollout=3)
    cfg["runner"].update(policy_class_name="ActorCritic", algorithm_class_name="PPO")
    cfg["policy"] = {"actor_hidden_dims": [8], "critic_hidden_dims": [8],
                     "activation": "elu", "init_noise_std": 1.}
    del cfg["algorithm"]["estimation_weight"], cfg["algorithm"]["kl_weight"]
    env = TensorEnvironment()
    learner = OnPolicyRunner(env, cfg, log_dir=str(tmp_path), device="cpu")
    # A stock task has no multimodal API. Its native get_observations continues
    # to work through a saved direct callable, while sensor access raises.
    original = env.get_pie_observations
    env.get_observations = lambda: original()["proprio"]
    env.get_privileged_observations = lambda: original()["critic"]
    def stock_step(actions):
        env.step_index += 1
        observations = original()
        return (observations["proprio"], observations["critic"], torch.ones(env.num_envs),
                torch.zeros(env.num_envs, dtype=torch.bool), {})
    env.step = stock_step
    def no_visual_api():
        raise AssertionError("Stock runner must not request PIE side inputs")
    env.get_pie_observations = no_visual_api
    before = learner.alg.actor_critic.actor[0].weight.detach().clone()
    report = learner.learn(2)
    assert learner.current_learning_iteration == 2 and learner.tot_timesteps == 18
    assert not torch.equal(before, learner.alg.actor_critic.actor[0].weight)
    assert report is None
    learner.writer.close()


def test_real_task_registry_constructs_native_pie_runner_from_class_config():
    from types import SimpleNamespace
    from native_cpu_helpers import load_native_classes
    from native_rsl_helpers import model_config
    classes = load_native_classes()
    registry = classes.registry()
    env_cfg, training = classes.config(), classes.train_config()
    for name, value in model_config().items():
        setattr(training.policy, name, value)
    training.algorithm.num_learning_epochs = training.algorithm.num_mini_batches = 1
    training.runner.num_steps_per_env = 3
    registry.register("lite3_pie", classes.task, env_cfg, training)
    copied, copied_training = registry.get_cfgs("lite3_pie")
    copied.control.action_scale = 99
    assert registry.get_cfgs("lite3_pie")[0].control.action_scale == .25
    args = SimpleNamespace(seed=None, num_envs=None, max_iterations=None,
                           resume=False, experiment_name=None, run_name=None,
                           load_run=None, checkpoint=None, rl_device="cpu")
    native_runner, resolved = registry.make_alg_runner(
        TensorEnvironment(), name="lite3_pie", args=args, log_root=None)
    assert isinstance(native_runner, PIEOnPolicyRunner)
    assert native_runner.num_steps_per_env == 3
    assert native_runner.alg.num_learning_epochs == native_runner.alg.num_mini_batches == 1
    assert native_runner.learn(1)["iteration"] == 1


def test_parsed_native_config_numpy_fields_make_weights_only_safe_checkpoint(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from native_cpu_helpers import load_native_classes
    classes = load_native_classes()
    task = classes.task.__new__(classes.task)
    task.cfg = classes.config()
    task.cfg.seed = np.int64(3)
    task.cfg.domain_rand.friction_range = np.asarray([.2, 1.2], dtype=np.float64)
    task.cfg.domain_rand.randomize_pie = np.bool_(True)
    task.sim_params = SimpleNamespace(dt=.005)
    task._parse_cfg(task.cfg)
    assert isinstance(task.cfg.domain_rand.push_interval, np.float64)
    learner = runner()
    learner.env.cfg = task.cfg
    checkpoint = tmp_path / "parsed.pt"
    learner.save(checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    values = saved["environment_cfg"]
    assert type(values["seed"]) is int and values["seed"] == 3
    assert type(values["domain_rand"]["push_interval"]) is float
    assert values["domain_rand"]["push_interval"] == 750.
    assert values["domain_rand"]["friction_range"] == [.2, 1.2]
    assert type(values["domain_rand"]["randomize_pie"]) is bool


def test_default_network_one_env_checkpoint_load_and_recurrent_inference(tmp_path):
    from rsl_rl.modules.actor_critic_pie import PIEActorCritic
    class FullEnvironment(TensorEnvironment):
        def __init__(self, count):
            super().__init__(count=count)
            self.num_obs, self.num_privileged_obs, self.num_actions = 45, 235, 12
        def get_pie_observations(self):
            prop = torch.linspace(-.1, .1, self.num_obs).repeat(self.num_envs, 1)
            depth = torch.linspace(-.2, .2, 60*80).reshape(1, 1, 60, 80).repeat(self.num_envs, 2, 1, 1)
            velocity = torch.zeros(self.num_envs, 3)
            heightmap = torch.zeros(self.num_envs, 187)
            return {"proprio": prop, "proprio_history": prop[:, None].repeat(1, 10, 1),
                    "depth": depth, "depth_frame_ids": torch.arange(self.num_envs).repeat(2, 1).t(),
                    "critic": torch.cat((prop, velocity, heightmap), -1),
                    "targets": {"velocity": velocity, "foot_clearance": torch.zeros(self.num_envs, 4),
                                "heightmap": heightmap}}
    cfg = train_config(cfg=algorithm_config(), model=dict(num_actor_obs=45, num_critic_obs=235, num_actions=12), rollout=24)
    training = PIEOnPolicyRunner(FullEnvironment(4), cfg, log_dir=None, device="cpu")
    with torch.no_grad():
        training.alg.actor_critic.std.fill_(.64)
    checkpoint = tmp_path / "native_default.pt"
    training.save(checkpoint)
    inference = PIEOnPolicyRunner(FullEnvironment(1), cfg, log_dir=None, device="cpu")
    inference.load(checkpoint, load_optimizer=False)
    assert inference.alg.num_mini_batches == 4
    torch.testing.assert_close(inference.alg.actor_critic.std, torch.full((12,), .64))
    policy = inference.get_inference_policy()
    first = policy(inference.env.get_observations())
    second = policy(inference.env.get_observations())
    fresh = policy(inference.env.get_observations(), reset_mask=torch.ones(1, dtype=torch.bool))
    assert first.shape == second.shape == (1, 12)
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert not torch.equal(first, second)
    torch.testing.assert_close(first, fresh, rtol=0, atol=0)


def test_inference_env_count_does_not_silently_reduce_training_minibatches():
    from rsl_rl.algorithms.ppo_pie import PIEPPO
    from rsl_rl.modules.actor_critic_pie import PIEActorCritic
    from native_rsl_helpers import collect_native, model_config
    algorithm = PIEPPO(PIEActorCritic(**model_config()), device="cpu", **algorithm_config())
    environment = TensorEnvironment(count=1, resets={1: (0, False)})
    collect_native(algorithm, environment, steps=2)
    with pytest.raises(ValueError, match="num_envs >= minibatches"):
        algorithm.update()
