"""PIE runner built on the v1.0.2 OnPolicyRunner construction contract."""
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
import json
import random
import torch
from .on_policy_runner import OnPolicyRunner
from rsl_rl.modules.actor_critic_pie import ModelConfig, PIEActorCritic
from rsl_rl.algorithms.ppo_pie import PPOConfig

@dataclass
class PIERunnerCfg:
    """Task-level network and joint PPO settings, with a minimal default budget."""
    seed: int = 0
    num_steps_per_env: int = 8
    max_iterations: int = 1
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
        seed_everything(train_cfg.seed)
        native_cfg = {
            "runner": {"policy_class_name": "PIEActorCritic",
                       "algorithm_class_name": "PIEPPO",
                       "num_steps_per_env": train_cfg.num_steps_per_env,
                       "save_interval": 1},
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
        last = {}
        for _ in range(iterations):
            batch, obs, hidden, reset_mask = self.alg.collect(
                self.env, obs, hidden, reset_mask, self.num_steps_per_env)
            last = self.alg.update(batch)
            self.current_learning_iteration += 1
            transitions = len(batch["frames"]) * hidden.shape[0]
            self.tot_timesteps += transitions
            last.update(iteration=self.current_learning_iteration,
                        mean_reward=float(torch.stack([f["rewards"] for f in batch["frames"]]).mean()),
                        transitions=transitions)
            if output is not None:
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(last, allow_nan=False) + "\n")
            print(json.dumps(last, allow_nan=False), flush=True)
            hidden, reset_mask = self.alg.refresh_hidden(batch)
            self._observation, self._hidden, self._reset_mask = obs, hidden, reset_mask
        if output is not None:
            checkpoint = output / "checkpoint.pt"
            self.save(checkpoint)
            last = {"checkpoint": str(checkpoint), **last}
        return last

    def save(self, path, infos=None):
        environment_cfg = getattr(self.env, "config", None)
        environment_cfg = asdict(environment_cfg) if is_dataclass(environment_cfg) else {}
        state, optimizer = self.alg.actor_critic.state_dict(), self.alg.optimizer.state_dict()
        # Native runner keys plus compatibility keys for existing bounded play.
        torch.save({
            "model_state_dict": state, "optimizer_state_dict": optimizer,
            "iter": self.current_learning_iteration, "infos": infos,
            "model": state, "optimizer": optimizer,
            "model_config": asdict(self.pie_cfg.model), "ppo_config": asdict(self.pie_cfg.ppo),
            "iterations": self.current_learning_iteration, "seed": self.pie_cfg.seed,
            "environment_config": environment_cfg, "torch_rng": torch.get_rng_state(),
            "rsl_rl_base": "v1.0.2", "pie_checkpoint_version": 2,
        }, path)

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
