"""CPU-only native VecEnv fixture and collection using actual RSL entry points."""

import torch

from rsl_rl.algorithms.ppo_pie import PIEPPO
from rsl_rl.modules.actor_critic_pie import PIEActorCritic


def algorithm_config(**overrides):
    """Use the real registered task's native algorithm fields, with CPU overrides."""
    from native_cpu_helpers import load_native_classes
    classes = load_native_classes()
    values = classes.helpers.class_to_dict(classes.train_config().algorithm)
    values.update(overrides)
    return values


def model_config(**overrides):
    values = dict(
        initial_std=.5, proprio_dim=3, proprio_history=2, depth_history=2,
        action_dim=2, heightmap_dim=3, critic_dim=4, token_dim=8, gru_dim=8,
        latent_dim=2, map_latent_dim=2, transformer_heads=2,
        proprio_hidden_dims=(8,), cnn_hidden_channels=(2, 2),
        cnn_kernel_sizes=(3, 3, 3), cnn_strides=(1, 1, 1),
        cnn_paddings=(1, 1, 1), visual_grid=(1, 1),
        actor_hidden_dims=(8,), critic_hidden_dims=(8,),
        successor_hidden_dims=(8,), height_decoder_hidden_dims=(8,))
    values.update(overrides)
    names = dict(proprio_dim="num_actor_obs", critic_dim="num_critic_obs",
                 action_dim="num_actions", initial_std="init_noise_std")
    return {names.get(name, name): value for name, value in values.items()}


def train_config(cfg=None, model=None, rollout=3, save_interval=500):
    return {"runner": {"policy_class_name": "PIEActorCritic",
                       "algorithm_class_name": "PIEPPO",
                       "num_steps_per_env": rollout, "save_interval": save_interval,
                       "max_iterations": 15000},
            "policy": dict(model or model_config()),
            "algorithm": (cfg or algorithm_config(num_learning_epochs=1, num_mini_batches=1))}


class TensorEnvironment:
    """Time-indexed inputs; terminal tensors intentionally differ from autoreset."""
    def __init__(self, count=3, resets=None, indexed=True):
        self.num_envs = self.count = count
        self.num_obs, self.num_privileged_obs, self.num_actions = 3, 4, 2
        self.device = "cpu"
        self.step_index = 0
        self.indexed = indexed
        self.resets = resets or {1: (0, False), 2: (1, True)}
        self.episode_length_buf = torch.zeros(count, dtype=torch.long)
        self.max_episode_length = 1000
        self.terrain_levels = torch.arange(count, dtype=torch.long)
        self.cfg = type("EnvironmentConfig", (), {"seed": 1})()

    def get_pie_observations(self):
        prop = torch.arange(self.count * 3, dtype=torch.get_default_dtype()).reshape(self.count, 3)
        prop = prop / 10 + self.step_index / 5
        capture = (self.step_index // 5) * self.count
        ids = capture + torch.arange(self.count).repeat(2, 1).t()
        obs = {"proprio": prop,
               "proprio_history": torch.stack((prop - .1, prop), dim=1),
               "depth": (ids[:, :, None, None].to(prop.dtype).expand(-1, -1, 8, 8).clone() / 30
                         + torch.linspace(.01, .2, 64).reshape(8, 8)),
               "critic": torch.cat((prop, prop[:, :1]), dim=-1),
               "targets": {"velocity": prop.clone(),
                           "foot_clearance": torch.zeros(self.count, 4),
                           "heightmap": prop / 2}}
        if self.indexed:
            obs["depth_frame_ids"] = ids
        return obs

    def get_observations(self):
        return self.get_pie_observations()["proprio"]

    def get_privileged_observations(self):
        return self.get_pie_observations()["critic"]

    def reset(self):
        self.step_index = 0
        self.episode_length_buf.zero_()
        return self.get_observations(), self.get_privileged_observations()

    def step(self, actions):
        self.step_index += 1
        self.episode_length_buf += 1
        obs = self.get_pie_observations()
        dones = torch.zeros(self.count, dtype=torch.bool)
        timeouts = torch.zeros_like(dones)
        selection = self.resets.get(self.step_index)
        if selection is not None:
            index, timeout = selection
            dones[index] = True
            timeouts[index] = timeout
            self.episode_length_buf[index] = 0
        terminal_prop = obs["proprio"].clone()
        terminal_critic = obs["critic"].clone()
        terminal_prop[dones] += 7
        terminal_critic[dones] += 11
        extras = {"time_outs": timeouts,
                  "pie": {"terminal_proprio": terminal_prop,
                          "terminal_critic": terminal_critic},
                  "episode": {"rew_tracking_lin_vel": torch.tensor(.7),
                              "rew_collision": torch.tensor(-.2)}}
        rewards = torch.arange(self.count, dtype=torch.get_default_dtype()) + 1
        return obs["proprio"], obs["critic"], rewards, dones, extras


def collect_native(algorithm, environment, steps=3, initial_hidden=None):
    algorithm.init_storage(environment.num_envs, steps, [environment.num_obs],
                           [environment.num_privileged_obs], [environment.num_actions])
    if initial_hidden is not None:
        algorithm.actor_critic.set_hidden_states(initial_hidden.clone())
    algorithm.begin_rollout()
    transitions = []
    with torch.no_grad():
        for _ in range(steps):
            sensor = environment.get_pie_observations()
            actions = algorithm.act(sensor, sensor["critic"])
            obs, critic, reward, done, info = environment.step(actions)
            transitions.append((reward.clone(), done.clone(), info))
            algorithm.process_env_step(reward, done, info)
        algorithm.compute_returns(critic)
    return storage_snapshot(algorithm.storage), transitions


def rollout(cfg=None, steps=3, indexed=True):
    torch.manual_seed(41)
    model = PIEActorCritic(**model_config())
    algorithm = PIEPPO(model, device="cpu",
                       **(cfg or algorithm_config(num_learning_epochs=1, num_mini_batches=1)))
    environment = TensorEnvironment(count=max(3, algorithm.num_mini_batches), indexed=indexed)
    batch, _ = collect_native(algorithm, environment, steps)
    return algorithm, batch


def storage_snapshot(storage):
    """Test-only views for independent dense replay and transition assertions."""
    frames = []
    for step in range(storage.step):
        frame = {"obs": storage.observation(step), "reset": storage.reset_mask[step],
                 "targets": {name: value[step] for name, value in storage.targets.items()}}
        for name in ("successor", "successor_valid", "valid", "raw_rewards", "terminated", "truncated"):
            frame[name] = getattr(storage, name)[step]
        for name, tensor in (("actions", storage.actions), ("old_logp", storage.actions_log_prob),
                             ("values", storage.values), ("rewards", storage.rewards),
                             ("old_mu", storage.mu), ("old_sigma", storage.sigma)):
            frame[name] = tensor[step] if name in ("actions", "old_mu", "old_sigma") else tensor[step, :, 0]
        frames.append(frame)
    return {"frames": frames, "hidden": storage.initial_hidden,
            "depth_frames": storage.depth_pool.materialize(),
            "advantages": storage.advantages[:, :, 0], "returns": storage.returns[:, :, 0]}


def restore_observation(frame, ids=None, depth_frames=None, restore_depth=False):
    obs = {name: value if ids is None else value[ids] for name, value in frame["obs"].items()}
    if restore_depth and "depth_indices" in obs:
        obs["depth"] = depth_frames[obs["depth_indices"]]
    return obs
