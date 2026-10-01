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
        self.noise.add_noise = False
        self.domain_rand.randomize_friction = True
        self.domain_rand.friction_range = [0.2, 1.2]
        self.domain_rand.randomize_base_mass = True
        self.domain_rand.added_mass_range = [-1.0, 2.0]
        self.domain_rand.push_robots = False
        self.rewards.soft_dof_pos_limit = 1.0
        self.rewards.only_positive_rewards = False
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
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = 'lite3_pie'
        num_steps_per_env = 8
        max_iterations = 15000
        save_interval = 500
