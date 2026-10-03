"""Lite3 task additions; original config fields own training and rewards."""
from dataclasses import asdict

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
        self.asset.flip_visual_attachments = False
        self.asset.penalize_contacts_on = ['TORSO', 'HIP', 'THIGH', 'SHANK']
        self.asset.terminate_after_contacts_on = ['TORSO']
        self.asset.default_dof_drive_mode = 3
        self.control.stiffness = {'joint': self.pie.robot.kp}
        self.control.damping = {'joint': self.pie.robot.kd}
        self.control.action_scale = self.pie.robot.action_scale
        self.control.decimation = 4
        self.control.control_type = 'P'
        self.domain_rand.randomize_friction = True
        self.domain_rand.friction_range = [0.2, 1.2]
        self.domain_rand.randomize_base_mass = True
        self.domain_rand.added_mass_range = [-1.0, 2.0]
        self.commands.heading_command = False
        self.commands.curriculum = False
        self.commands.num_commands = 3
        self.commands.resampling_time = 10.0
        self.commands.ranges.lin_vel_x = [0.0, 1.5]
        self.commands.ranges.lin_vel_y = [0.0, 0.0]
        self.commands.ranges.ang_vel_yaw = [-1.2, 1.2]
        self.terrain.mesh_type = 'trimesh'
        self.terrain.curriculum = True
        self.terrain.num_rows = 10
        self.terrain.num_cols = 20
        self.terrain.terrain_length = 8.0
        self.terrain.terrain_width = 8.0
        self.terrain.terrain_proportions = [0.2] * 5
        self.terrain.max_init_terrain_level = 5
        self.rewards.soft_dof_pos_limit = 1.0
        # Original reward registration and dt scaling remain the only pipeline.
        scales = {'tracking_lin_vel': 1.5, 'tracking_ang_vel': 0.5,
                  'lin_vel_z': -1.0, 'ang_vel_xy': -0.05, 'orientation': -1.0,
                  'dof_acc': -2.5e-7, 'joint_power': -2e-5, 'collision': -10.0,
                  'action_rate': -0.01, 'smoothness': -0.01}
        for name in dir(self.rewards.scales):
            if not name.startswith('_'):
                setattr(self.rewards.scales, name, 0.0)
        for name, value in scales.items():
            setattr(self.rewards.scales, name, value)


class Lite3PIECfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'PIEOnPolicyRunner'

    def __init__(self):
        super().__init__()
        from rsl_rl.modules import ModelConfig
        model = asdict(ModelConfig())
        for native_owned in ('initial_std', 'actor_hidden_dims', 'critic_hidden_dims'):
            model.pop(native_owned)
        self.policy.model_config = model
        self.algorithm.estimation_weight = 1.0
        self.algorithm.kl_weight = 1.0

    class policy(LeggedRobotCfgPPO.policy):
        # Native policy fields own actor/critic widths, activation and std.
        # PIE adds CNN/Transformer/GRU and estimator dimensions only.
        model_config = None

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'PIEActorCritic'
        algorithm_class_name = 'PIEPPO'
        experiment_name = 'lite3_pie_native'
        num_steps_per_env = 24
        max_iterations = 15000
        save_interval = 500
        completed_iteration_numbering = True
