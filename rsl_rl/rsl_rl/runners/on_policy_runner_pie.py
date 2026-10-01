"""PIE runner built on the v1.0.2 OnPolicyRunner construction contract."""
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from collections import deque
import json
import random
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from .on_policy_runner import OnPolicyRunner
from rsl_rl.modules.actor_critic_pie import ModelConfig, PIEActorCritic
from rsl_rl.algorithms.ppo_pie import PPOConfig

@dataclass
class PIERunnerCfg:
    """Task-level network, joint PPO settings and checkpoint schedule."""
    seed: int = 0
    num_steps_per_env: int = 8
    max_iterations: int = 15000
    save_interval: int = 500
    experiment_name: str = "lite3_pie"
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class PIEOnPolicyRunner(OnPolicyRunner):
    """Native RSL factories/storage/optimizer with PIE dict observations and labels."""
    def __init__(self, env, train_cfg, log_dir="runs/minimal", device=None):
        device = torch.device(device or env.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device != torch.device(env.device):
            raise ValueError("PIE requires the RL device to match the simulation device.")
        if isinstance(train_cfg, dict):
            data = dict(train_cfg)
            if isinstance(data.get("model"), dict):
                data["model"] = ModelConfig(**data["model"])
            if isinstance(data.get("ppo"), dict):
                data["ppo"] = PPOConfig(**data["ppo"])
            train_cfg = PIERunnerCfg(**data)
        self.pie_cfg = train_cfg
        if train_cfg.num_steps_per_env < 1:
            raise ValueError("num_steps_per_env must be positive")
        if not isinstance(train_cfg.save_interval, int) or train_cfg.save_interval < 1:
            raise ValueError("save_interval must be a positive integer")
        seed_everything(train_cfg.seed)
        native_cfg = {
            "runner": {"policy_class_name": "PIEActorCritic",
                       "algorithm_class_name": "PIEPPO",
                       "num_steps_per_env": train_cfg.num_steps_per_env,
                       "save_interval": train_cfg.save_interval},
            "policy": {"model_config": asdict(train_cfg.model)},
            "algorithm": {"pie_config": asdict(train_cfg.ppo)},
        }
        super().__init__(env, native_cfg, log_dir, device=device or env.device)
        train_cfg.model.validate_observation(self._observation, env.num_actions)
        self._hidden = self.alg.actor_critic.initial_state(env.num_envs)
        self._reset_mask = torch.ones(env.num_envs, dtype=torch.bool, device=self.device)

    def _reset_env(self):
        # Stock VecEnv.reset returns a pair; PIE retains sensor history and labels.
        self._observation = self.env.reset()

    def learn(self, num_learning_iterations=None, init_at_random_ep_len=False):
        iterations = (self.pie_cfg.max_iterations if num_learning_iterations is None
                      else num_learning_iterations)
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if init_at_random_ep_len:
            counter = self.env.episode_steps
            limit = round(self.env.config.episode_seconds / self.env.config.policy_dt)
            counter.copy_(torch.randint(max(1, limit), counter.shape, device=counter.device))
        obs, hidden, reset_mask = self._observation, self._hidden, self._reset_mask
        output = Path(self.log_dir) if self.log_dir is not None else None
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            if self.writer is None:
                self.writer = SummaryWriter(log_dir=str(output), flush_secs=10)
        last = {}
        target_iteration = self.current_learning_iteration + iterations
        for _ in range(iterations):
            start = time.perf_counter()
            batch, obs, hidden, reset_mask = self.alg.collect(
                self.env, obs, hidden, reset_mask, self.num_steps_per_env)
            collected = time.perf_counter()
            last = self.alg.update(batch)
            hidden, reset_mask = self.alg.refresh_hidden(batch)
            self._observation, self._hidden, self._reset_mask = obs, hidden, reset_mask
            self.current_learning_iteration += 1
            transitions = len(batch["frames"]) * hidden.shape[0]
            self.tot_timesteps += transitions
            self._update_episode_statistics(batch["frames"])
            last.update(iteration=self.current_learning_iteration,
                        mean_reward=float(torch.stack([f["rewards"] for f in batch["frames"]]).mean()),
                        transitions=transitions)
            last.update(learning_rate=self.alg.optimizer.param_groups[0]["lr"],
                        mean_action_noise_std=float(self.alg.actor_critic.std.detach().mean()))
            stop = time.perf_counter()
            iteration_time = stop - start
            self.tot_time += iteration_time
            last.update(collection_time=collected-start, learning_time=stop-collected,
                        iteration_time=iteration_time, total_time=self.tot_time,
                        total_timesteps=self.tot_timesteps, fps=transitions/max(iteration_time, 1e-9))
            if self._reward_buffer:
                last.update(mean_episode_reward=sum(self._reward_buffer)/len(self._reward_buffer),
                            mean_episode_length=sum(self._length_buffer)/len(self._length_buffer))
            if output is not None:
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(last, allow_nan=False) + "\n")
            self._log_pie_iteration(last, target_iteration)
            if output is not None and self.current_learning_iteration % self.save_interval == 0:
                self.save(output / "model_{}.pt".format(self.current_learning_iteration))
        if output is not None:
            checkpoint = output / "checkpoint.pt"
            self.save(checkpoint)
            last = {"checkpoint": str(checkpoint), **last}
            self.writer.flush()
        return last

    def _update_episode_statistics(self, frames):
        # Like the native RSL runner, count only steps actually collected;
        # randomized timeout counters do not inflate logged episode lengths.
        if not hasattr(self, "_episode_reward_sum"):
            self._episode_reward_sum = torch.zeros_like(frames[0]["rewards"])
            self._episode_step_count = torch.zeros_like(frames[0]["rewards"], dtype=torch.long)
            self._reward_buffer, self._length_buffer = deque(maxlen=100), deque(maxlen=100)
        returns, lengths = [], []
        for frame in frames:
            self._episode_reward_sum += frame["rewards"]
            self._episode_step_count += 1
            done = frame["terminated"] | frame["truncated"]
            if done.any():
                returns.append(self._episode_reward_sum[done].clone())
                lengths.append(self._episode_step_count[done].clone())
                self._episode_reward_sum[done] = 0
                self._episode_step_count[done] = 0
        if returns:
            self._reward_buffer.extend(torch.cat(returns).cpu().tolist())
            self._length_buffer.extend(torch.cat(lengths).cpu().tolist())

    def _log_pie_iteration(self, metrics, target_iteration):
        """Native RSL-style console/TensorBoard with PIE auxiliary losses."""
        tags = {
            "Loss/value_function": "value", "Loss/surrogate": "policy",
            "Loss/total": "loss", "Loss/learning_rate": "learning_rate",
            "Policy/mean_noise_std": "mean_action_noise_std",
            "Perf/total_fps": "fps", "Perf/collection_time": "collection_time",
            "Perf/learning_time": "learning_time", "Train/mean_step_reward": "mean_reward",
            "Train/mean_reward": "mean_episode_reward",
            "Train/mean_episode_length": "mean_episode_length",
            "PIE/velocity_loss": "velocity", "PIE/foot_clearance_loss": "foot_clearance",
            "PIE/heightmap_loss": "heightmap", "PIE/successor_loss": "successor",
            "PIE/vae_kl_loss": "kl", "PIE/gradient_norm_before_clipping": "grad_norm",
        }
        if self.writer is not None:
            for tag, key in tags.items():
                if key in metrics:
                    self.writer.add_scalar(tag, metrics[key], metrics["iteration"])
        lines = ["#"*80, " Learning iteration {}/{} ".format(
            metrics["iteration"], target_iteration).center(80), "",
            "{:>35} {:.0f} steps/s (collection: {:.3f}s, learning: {:.3f}s)".format(
                "Computation:", metrics["fps"], metrics["collection_time"], metrics["learning_time"])]
        labels = [("Value function loss:", "value"), ("Surrogate loss:", "policy"),
                  ("Mean action noise std:", "mean_action_noise_std"),
                  ("Mean reward:", "mean_episode_reward"),
                  ("Mean episode length:", "mean_episode_length"),
                  ("Mean step reward:", "mean_reward"),
                  ("Velocity estimation loss:", "velocity"),
                  ("Foot clearance loss:", "foot_clearance"),
                  ("Height map reconstruction loss:", "heightmap"),
                  ("Successor reconstruction loss:", "successor"),
                  ("VAE KL loss:", "kl"), ("Total loss:", "loss"),
                  ("Gradient norm before clipping:", "grad_norm")]
        lines.extend("{:>35} {:.4f}".format(label, metrics[key])
                     for label, key in labels if key in metrics)
        remaining = target_iteration - metrics["iteration"]
        lines.extend(["-"*80,
                      "{:>35} {}".format("Total timesteps:", metrics["total_timesteps"]),
                      "{:>35} {:.2f}s".format("Iteration time:", metrics["iteration_time"]),
                      "{:>35} {:.2f}s".format("Total time:", metrics["total_time"]),
                      "{:>35} {:.1f}s".format("ETA:", self.tot_time / metrics["iteration"] * remaining)])
        print("\n".join(lines), flush=True)

    def save(self, path, infos=None):
        environment_cfg = getattr(self.env, "config", None)
        environment_cfg = asdict(environment_cfg) if is_dataclass(environment_cfg) else {}
        state, optimizer = self.alg.actor_critic.state_dict(), self.alg.optimizer.state_dict()
        # Native runner keys plus compatibility keys for existing bounded play.
        checkpoint = {
            "model_state_dict": state, "optimizer_state_dict": optimizer,
            "iter": self.current_learning_iteration, "infos": infos,
            "model": state, "optimizer": optimizer,
            "model_config": asdict(self.pie_cfg.model), "ppo_config": asdict(self.pie_cfg.ppo),
            "iterations": self.current_learning_iteration, "seed": self.pie_cfg.seed,
            "environment_config": environment_cfg, "torch_rng": torch.get_rng_state(),
            "rsl_rl_base": "v1.0.2", "pie_checkpoint_version": 2,
        }
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        try:
            torch.save(checkpoint, temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load(self, path, load_optimizer=True):
        if load_optimizer:
            raise NotImplementedError("PIE optimizer/episode resume is not implemented; use load_optimizer=False for weights.")
        saved = torch.load(path, map_location=self.device, weights_only=True)
        if asdict(ModelConfig(**saved["model_config"])) != asdict(self.pie_cfg.model):
            raise ValueError("Checkpoint model configuration does not match this task")
        self.alg.actor_critic.load_state_dict(
            saved["model_state_dict"] if "model_state_dict" in saved else saved["model"])
        self._reset_env()
        self._hidden = self.alg.actor_critic.initial_state(self.env.num_envs)
        self._reset_mask.fill_(True)
        return saved.get("infos")

    def get_inference_policy(self, device=None):
        model = self.alg.actor_critic
        model.eval()
        if device is not None:
            model.to(device)
        hidden = None
        @torch.no_grad()
        def policy(obs, reset_mask=None):
            nonlocal hidden
            if hidden is None or hidden.shape[0] != obs["proprio"].shape[0]:
                hidden = model.initial_state(obs["proprio"].shape[0])
            actions, hidden = model.act_inference(obs, hidden, reset_mask)
            return actions
        return policy

    def add_git_repo_to_log(self, source_file):
        import subprocess
        if self.log_dir is None:
            return
        output = Path(self.log_dir)
        output.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "-C", str(Path(source_file).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True)
        (output / "upstream_commit.txt").write_text(result.stdout)
        # The published rsl_rl directory is vendored; its nearest Git HEAD is
        # the PIE repository, so retain the pinned dependency baseline explicitly.
        (output / "rsl_rl_base_commit.txt").write_text(
            "2ad79cf0caa85b91721abfe358105f869a784121\n")


def train(env, iterations=1, rollout_steps=8, output_dir="runs/minimal",
          seed=0, model_config=None, ppo_config=None, init_at_random_ep_len=False):
    """Bounded convenience entry; uses the same native-derived PIE runner."""
    cfg = PIERunnerCfg(seed=seed, max_iterations=iterations,
                       num_steps_per_env=rollout_steps,
                       model=model_config or ModelConfig(),
                       ppo=ppo_config or PPOConfig())
    runner = PIEOnPolicyRunner(env, cfg, output_dir, device=env.device)
    return runner.learn(iterations, init_at_random_ep_len)

@torch.no_grad()
def evaluate(env, checkpoint, steps=20):
    if steps < 1:
        raise ValueError("steps must be positive")
    obs = env.reset()
    saved = torch.load(checkpoint, map_location=obs["proprio"].device, weights_only=True)
    cfg = ModelConfig(**saved["model_config"]).validate_observation(obs, env.num_actions)
    model = PIEActorCritic(cfg).to(obs["proprio"].device)
    model.load_state_dict(saved["model"]); model.eval()
    hidden = model.initial_state(obs["proprio"].shape[0])
    reset_mask = torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device)
    rewards, terminated_count, timeout_count = [], 0, 0
    for _ in range(steps):
        actions, _, _, hidden = model.act(obs, hidden, reset_mask, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(actions)
        rewards.append(float(reward.mean()))
        terminated_count += int(terminated.sum()); timeout_count += int(truncated.sum())
        reset_mask = terminated | truncated
    return {"steps": steps, "mean_reward": sum(rewards)/len(rewards),
            "terminations": terminated_count, "timeouts": timeout_count,
            "claim": "untrained/minimally trained plumbing check; not paper performance"}
