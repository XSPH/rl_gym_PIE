"""PIE hooks on the original RSL-RL v1.0.2 runner; no second learn loop."""
from dataclasses import asdict, is_dataclass
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
import torch
from .on_policy_runner import OnPolicyRunner
from rsl_rl.modules.actor_critic_pie import ModelConfig


def plain_config(value):
    """Serialize configuration fields, including nested BaseConfig classes."""
    if is_dataclass(value) and not isinstance(value, type):
        return plain_config(asdict(value))
    if isinstance(value, dict):
        return {key: plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_config(item) for item in value]
    # NumPy scalar subclasses (notably np.float64) can pass isinstance(float)
    # but their pickles require unsafe globals under torch weights_only loading.
    if isinstance(value, np.generic):
        return plain_config(value.item())
    if isinstance(value, np.ndarray):
        return plain_config(value.tolist())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, type) or hasattr(value, "__dict__"):
        result = {}
        for name in dir(value):
            if name.startswith('_'):
                continue
            item = getattr(value, name)
            if callable(item) and not isinstance(item, type):
                continue
            result[name] = plain_config(item)
        return result
    raise TypeError("Unsupported configuration value: {}".format(type(value).__name__))


class PIEOnPolicyRunner(OnPolicyRunner):
    """Sensor preparation, metrics, and checkpoint hooks; learn() is inherited."""
    def __init__(self, env, train_cfg, log_dir=None, device=None):
        if not isinstance(train_cfg, dict) or not {"runner", "policy", "algorithm"}.issubset(train_cfg):
            raise TypeError("PIE runner requires the native runner/policy/algorithm train_cfg dictionary")
        device = torch.device(device or env.device)
        sim_device = torch.device(env.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if sim_device.type == "cuda" and sim_device.index is None:
            sim_device = torch.device("cuda", torch.cuda.current_device())
        if device != sim_device:
            raise ValueError("PIE requires matching simulation and RL devices")
        self.train_cfg = deepcopy(train_cfg)
        self.train_cfg["runner"]["completed_iteration_numbering"] = True
        runner = self.train_cfg["runner"]
        if runner["num_steps_per_env"] < 1 or runner["save_interval"] < 1:
            raise ValueError("PIE rollout length and save interval must be positive")
        self._rollout_reward_sum = None
        self._reset_counts = {}
        super().__init__(env, self.train_cfg, log_dir, device)
        self.model_cfg = self.alg.actor_critic.cfg
        self.model_cfg.validate_observation(env.get_pie_observations(), env.num_actions)
        print("PIE training: envs={}, rollout={}, epochs={}, minibatches={}, lr={}, "
              "schedule={}, desired_kl={}, initial_std={}, target_iterations={}, save_interval={}".format(
                  env.num_envs, self.num_steps_per_env, self.alg.num_learning_epochs,
                  self.alg.num_mini_batches, self.alg.learning_rate, self.alg.schedule,
                  self.alg.desired_kl, self.model_cfg.initial_std,
                  runner.get("max_iterations", 15000), self.save_interval))

    def _actor_observations(self, observations):
        return self.env.get_pie_observations()

    def _rollout_context(self):
        # Camera tensors allocated during collection must remain usable by
        # autograd; inference-mode tensors cannot be saved for CNN backward.
        return torch.no_grad()

    def _begin_rollout(self):
        self.alg.begin_rollout()
        self._rollout_reward_sum = torch.zeros((), device=self.device)
        self._reset_counts = {}

    def _on_env_step(self, rewards, dones, infos):
        self._rollout_reward_sum += rewards.sum()
        side = infos.get("pie", {})
        timeout = infos.get("time_outs", torch.zeros_like(dones)).bool()
        counts = {"timeouts": timeout.sum(), "failures": (dones.bool() & ~timeout).sum()}
        for name, flags in side.get("termination_reasons", {}).items():
            counts[name] = flags.bool().sum()
        for name, count in counts.items():
            if name not in self._reset_counts:
                self._reset_counts[name] = count.detach().clone()
            else:
                self._reset_counts[name] += count

    def _iteration_metrics(self, locs):
        metrics = super()._iteration_metrics(locs)
        metrics.update(learning_rate=self.alg.learning_rate,
                       mean_action_noise_std=float(self.alg.actor_critic.std.detach().mean()),
                       mean_reward=float(self._rollout_reward_sum) / (self.num_steps_per_env * self.env.num_envs))
        levels = getattr(self.env, "terrain_levels", None)
        if levels is not None:
            metrics.update(terrain_level=float(levels.float().mean()),
                           terrain_level_min=int(levels.min()), terrain_level_max=int(levels.max()))
        metrics["reset_counts"] = {name: int(value) for name, value in self._reset_counts.items()}
        if locs["rewbuffer"]:
            metrics["mean_episode_reward"] = sum(locs["rewbuffer"]) / len(locs["rewbuffer"])
            metrics["mean_episode_length"] = sum(locs["lenbuffer"]) / len(locs["lenbuffer"])
        ep_infos = locs["ep_infos"]
        if ep_infos:
            metrics["episode_rewards"] = {
                name: float(torch.cat([torch.as_tensor(info[name], device=self.device).flatten()
                                      for info in ep_infos if name in info]).float().mean())
                for name in ep_infos[0]}
        return metrics

    def _extra_log_string(self, metrics):
        fields = [
            ("Total loss:", "loss"), ("Velocity estimation loss:", "velocity"),
            ("Foot clearance loss:", "foot_clearance"), ("Height map reconstruction loss:", "heightmap"),
            ("Successor reconstruction loss:", "successor"), ("VAE KL loss:", "kl"),
            ("Policy KL divergence:", "policy_kl"), ("Gradient norm before clipping:", "grad_norm"),
            ("Learning rate:", "learning_rate"), ("Mean step reward:", "mean_reward"),
            ("Mean terrain level:", "terrain_level"), ("Depth frame pool (MiB):", "depth_pool_mib"),
            ("Equivalent dense depth (MiB):", "depth_dense_mib"), ("CNN feature reuse fraction:", "cnn_reuse_fraction"),
        ]
        result = ''.join("{:>35} {:.6f}\n".format(label, metrics[key])
                         for label, key in fields if key in metrics)
        for label, key in (("Min terrain level:", "terrain_level_min"), ("Max terrain level:", "terrain_level_max"),
                           ("Unique depth frames:", "depth_unique_frames"), ("CNN encoded stacks:", "cnn_encoded_stacks"),
                           ("Equivalent dense CNN stacks:", "cnn_dense_stacks")):
            if key in metrics:
                result += "{:>35} {}\n".format(label, int(metrics[key]))
        for name, count in metrics.get("reset_counts", {}).items():
            result += "{:>35} {}\n".format("Reset {}:".format(name), count)
        result += "{:>35} {}\n".format("Transitions this iteration:", self.num_steps_per_env * self.env.num_envs)
        return result

    def _after_iteration(self, locs, metrics):
        if self.log_dir is None:
            return
        output = Path(self.log_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        iteration = metrics["iteration"]
        tags = {
            "loss": "Loss/total", "velocity": "PIE/velocity_loss", "foot_clearance": "PIE/foot_clearance_loss",
            "heightmap": "PIE/heightmap_loss", "successor": "PIE/successor_loss", "kl": "PIE/vae_kl_loss",
            "policy_kl": "PIE/policy_kl", "grad_norm": "PIE/gradient_norm_before_clipping",
            "terrain_level": "Episode/terrain_level", "terrain_level_min": "Terrain/min_level",
            "terrain_level_max": "Terrain/max_level", "mean_reward": "Train/mean_step_reward",
        }
        for name, value in metrics.items():
            if name.startswith(("depth_", "cnn_")):
                tags[name] = "PIE/" + name
        for name, tag in tags.items():
            if name in metrics:
                self.writer.add_scalar(tag, metrics[name], iteration)
        for name, value in metrics["reset_counts"].items():
            self.writer.add_scalar("Reset/" + name, value, iteration)

    def _save_final(self):
        self.save(Path(self.log_dir) / "checkpoint.pt")

    def save(self, path, infos=None):
        checkpoint = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration, "infos": infos,
            "learning_rate": float(self.alg.learning_rate),
            "model_config": asdict(self.model_cfg), "train_config": plain_config(self.train_cfg),
            "environment_cfg": plain_config(getattr(self.env, "cfg", {})),
            "seed": self.train_cfg.get("seed", 1), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "total_timesteps": self.tot_timesteps, "total_time": self.tot_time,
            "rsl_rl_base": "v1.0.2", "pie_checkpoint_version": 4,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            torch.save(checkpoint, temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load(self, path, load_optimizer=True):
        saved = torch.load(path, map_location=self.device, weights_only=True)
        if saved.get("pie_checkpoint_version") != 4 or saved.get("rsl_rl_base") != "v1.0.2":
            raise ValueError("This branch accepts only native PIE version-4 checkpoints")
        stored_config, current_config = asdict(ModelConfig(**saved["model_config"])), asdict(self.model_cfg)
        stored_config.pop("initial_std")
        current_config.pop("initial_std")
        if stored_config != current_config:
            raise ValueError("Checkpoint model configuration does not match this task")
        self.alg.actor_critic.load_state_dict(saved["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(saved["optimizer_state_dict"])
            self.alg.learning_rate = float(saved['learning_rate'])
            for group in self.alg.optimizer.param_groups:
                group['lr'] = self.alg.learning_rate
        self.current_learning_iteration = int(saved["iter"])
        self.tot_timesteps = int(saved.get("total_timesteps", 0))
        self.tot_time = float(saved.get("total_time", 0.0))
        torch.set_rng_state(saved["torch_rng"].cpu())
        if saved.get("cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        # Simulator episodes restart, as in native RSL resume.
        self._reset_env()
        self.alg.storage.clear()
        self.alg.actor_critic.reset()
        self.alg._reset_mask.fill_(True)
        return saved.get("infos")

    def get_inference_policy(self, device=None):
        model = self.alg.actor_critic
        if device is not None and torch.device(device) != torch.device(self.device):
            raise ValueError("PIE inference must use the environment device")
        model.eval()
        model.reset()
        @torch.no_grad()
        def policy(observations, reset_mask=None):
            if reset_mask is not None:
                model.reset(reset_mask)
            obs = observations if isinstance(observations, dict) else self.env.get_pie_observations()
            return model.act_inference(obs)
        return policy

    def add_git_repo_to_log(self, source_file):
        import subprocess
        if self.log_dir is None:
            return
        output = Path(self.log_dir)
        output.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(["git", "-C", str(Path(source_file).resolve().parent), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
        (output / "upstream_commit.txt").write_text(result.stdout)
        (output / "rsl_rl_base_commit.txt").write_text("2ad79cf0caa85b91721abfe358105f869a784121\n")
