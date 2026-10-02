from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.pie.config import EnvConfig


class Lite3PIECfg(LeggedRobotCfg):
    def __init__(self):
        super().__init__()
        self.pie = EnvConfig()
        self.env.num_envs = 4096
        self.env.num_observations = 45
        self.env.num_privileged_obs = 235
        self.env.num_actions = 12
        self.init_state.pos = [0.0, 0.0, self.pie.robot.base_height]
        self.init_state.default_joint_angles = dict(zip(self.pie.robot.joint_names, self.pie.robot.stand_angles))
        self.asset.file = str(self.pie.resolve_urdf())
        self.asset.name = 'lite3'
        self.asset.foot_name = 'FOOT'
        self.asset.collapse_fixed_joints = False
        self.asset.self_collisions = 1
        self.asset.penalize_contacts_on = ['HIP', 'THIGH', 'SHANK']
        self.asset.terminate_after_contacts_on = ['TORSO']
        self.asset.default_dof_drive_mode = 3
        self.control.stiffness = {'joint': self.pie.robot.kp}
        self.control.damping = {'joint': self.pie.robot.kd}
        self.control.action_scale = self.pie.robot.action_scale
        self.control.decimation = self.pie.decimation
        self.pie.observation_noise = self.noise.add_noise
        self.pie.command_seconds = self.commands.resampling_time
        self.pie.robot.action_clip = self.normalization.clip_actions
        self.pie.terrain.initial_max_level = self.terrain.max_init_terrain_level
        self.domain_rand.randomize_friction = True
        self.domain_rand.friction_range = [0.2, 1.2]
        self.domain_rand.randomize_base_mass = True
        self.domain_rand.added_mass_range = [-1.0, 2.0]
        self.rewards.soft_dof_pos_limit = 1.0
        self.commands.heading_command = False
        self.commands.num_commands = 3
        self.commands.ranges.lin_vel_x = [0.0, 1.5]
        self.commands.ranges.lin_vel_y = [0.0, 0.0]
        self.commands.ranges.ang_vel_yaw = [-1.2, 1.2]


class Lite3PIECfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'PIEOnPolicyRunner'
    def __init__(self):
        super().__init__()
        from rsl_rl.modules import ModelConfig
        from rsl_rl.algorithms import PPOConfig
        self.pie_model = ModelConfig()
        self.pie_ppo = PPOConfig()
        self.sync_pie_config()

    def sync_pie_config(self):
        """Use the original policy/algorithm fields as the training authority."""
        self.pie_model.initial_std = self.policy.init_noise_std
        self.pie_model.actor_hidden_dims = tuple(self.policy.actor_hidden_dims)
        self.pie_model.critic_hidden_dims = tuple(self.policy.critic_hidden_dims)
        fields = {
            'learning_rate': 'learning_rate', 'gamma': 'gamma',
            'gae_lambda': 'lam', 'clip': 'clip_param',
            'epochs': 'num_learning_epochs', 'minibatches': 'num_mini_batches',
            'entropy_weight': 'entropy_coef', 'value_weight': 'value_loss_coef',
            'max_grad_norm': 'max_grad_norm', 'schedule': 'schedule',
            'desired_kl': 'desired_kl',
        }
        for pie_name, native_name in fields.items():
            setattr(self.pie_ppo, pie_name, getattr(self.algorithm, native_name))

    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = 'lite3_pie'
        num_steps_per_env = 24
        max_iterations = 15000
        save_interval = 500
