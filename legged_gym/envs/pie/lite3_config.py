"""Lite3 PIE settings in the original nested task configuration."""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Lite3PIECfg(LeggedRobotCfg):
    seed = 1

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 45
        num_privileged_obs = 235
        num_actions = 12
        proprio_history = 10

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/lite3/urdf/Lite3.urdf'
        name = 'lite3'
        joint_names = [leg + '_' + joint + '_joint'
                       for leg in ('FL', 'FR', 'HL', 'HR')
                       for joint in ('HipX', 'HipY', 'Knee')]
        foot_names = [leg + '_FOOT' for leg in ('FL', 'FR', 'HL', 'HR')]
        base_name = 'TORSO'
        foot_radius = 0.022
        foot_name = 'FOOT'
        collapse_fixed_joints = False
        self_collisions = 1
        flip_visual_attachments = False
        penalize_contacts_on = ['TORSO', 'HIP', 'THIGH', 'SHANK']
        terminate_after_contacts_on = ['TORSO']
        default_dof_drive_mode = 3

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.31]
        default_joint_angles = {leg + '_' + joint + '_joint': angle
                                for leg in ('FL', 'FR', 'HL', 'HR')
                                for joint, angle in (('HipX', 0.0), ('HipY', -0.8), ('Knee', 1.6))}

    class control(LeggedRobotCfg.control):
        stiffness = {'joint': 20.0}
        damping = {'joint': 0.5}
        action_scale = 0.25
        decimation = 4
        control_type = 'P'

    class commands(LeggedRobotCfg.commands):
        # WMP Go1 terrain groups, adapted to six terrain kinds including slope.
        heading_command = True
        curriculum = True
        num_commands = 4  # The policy still observes only vx/vy/yaw.
        resampling_time = 10.0
        initial_limit = 0.5
        curriculum_threshold = 0.8
        curriculum_increment = 0.2
        group_maxima = {'flat': [2.0, 1.0], 'omni': [1.5, 1.0], 'stairs': [1.0, 0.8],
                        'forward': [1.0, 0.0]}
        heading_gain = 2.0
        zero_command_probability = 0.05
        small_command_threshold = 0.2

        class ranges(LeggedRobotCfg.commands.ranges):
            # Envelope for grouped commands; group_maxima controls sampling.
            lin_vel_x = [-2.0, 2.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.2, 1.2]
            heading = [-3.141592653589793, 3.141592653589793]

    class terrain(LeggedRobotCfg.terrain):
        geometry_version = 3
        mesh_type = 'trimesh'
        curriculum = True
        num_rows = 10
        num_cols = 20
        terrain_length = 8.0
        terrain_width = 8.0
        terrain_proportions = [1.0 / 6.0] * 6
        max_init_terrain_level = 0
        kinds = ['flat', 'slope', 'stairs', 'gap', 'step', 'hurdle']
        # WMP Go1 heightfields; slope/stairs use level / num_rows, while
        # gap/climb dimensions use level / (num_rows - 1).
        min_gap = 0.1
        max_gap = 0.9
        gap_floor_units = -1000
        gap_channel_width = [1.0, 2.0]
        min_step = 0.15
        max_step = 0.60
        step_length = [0.9, 1.1]
        min_hurdle = 0.05
        max_hurdle = 0.75
        hurdle_length = [0.2, 0.5]
        min_stair = 0.05
        max_stair = 0.23  # .05 + .18 * .9 = .212 m at row 9.
        stair_width = [0.30, 0.34]
        max_slope = 0.5  # Row 9 uses .45; nominal rise/run, about 24.2 degrees.
        platform_size = 3.0
        spawn_xy_range = 1.0
        obstacle_spawn_range = 0.5
        forward_spawn_x = 1.0
        obstacle_start_x = 3.3
        roughness = 0.05
        roughness_step = 0.005
        roughness_downsample = 0.2
        slope_threshold = 0.75
        heightmap_offset = 0.5

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.2]
        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]
        randomize_pie = True
        randomize_camera = True
        com_shift = 0.05
        gain_factor = [0.9, 1.1]
        motor_factor = [0.9, 1.1]
        max_delay_seconds = 0.015
        camera_position = 0.01
        camera_pitch_degrees = 1.0
        camera_hfov_degrees = [86.0, 88.0]

    class camera:
        # Render and encode real depth for the policy; 'zero' selects the blind ablation.
        input_mode = 'depth'  # 'depth' or 'zero'
        render_for_debug = False  # play --show_depth enables real captures in zero mode.
        height = 60
        width = 80
        history = 2
        update_every = 5
        latency_frames = 1
        near = 0.1
        far = 3.0
        hfov_degrees = 87.0
        position = [0.25, 0.0, 0.06]
        pitch_degrees = 30.0
        noise_std = 0.0
        salt_pepper_probability = 0.0
        # Network input: (optical-axis depth-near)/(far-near)-0.5.
        normalize = True

    class rewards(LeggedRobotCfg.rewards):
        lin_vel_clip = 0.1  # WMP obstacle groups permit temporary overspeed.
        soft_dof_pos_limit = 0.9
        base_height_target = 0.3

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_acc = -2.5e-7
            joint_power = -2e-5
            collision = -10.0
            action_rate = -0.01
            smoothness = -0.01
            torques = -1e-4
            dof_vel = 0.0
            base_height = -1.
            hip_default = -0.5
            feet_regulation = -0.05
            feet_air_time = 0.0
            feet_stumble = 0.0
            stand_still = 0.0


class Lite3PIECfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'PIEOnPolicyRunner'

    class policy(LeggedRobotCfgPPO.policy):
        proprio_history = 10
        depth_history = 2
        heightmap_dim = 187
        token_dim = 128
        gru_dim = 128
        latent_dim = 16
        map_latent_dim = 32
        transformer_heads = 4
        transformer_layers = 1
        proprio_hidden_dims = (512, 256)
        cnn_hidden_channels = (32, 64)
        cnn_kernel_sizes = (5, 3, 3)
        cnn_strides = (2, 2, 2)
        cnn_paddings = (2, 1, 1)
        visual_grid = (4, 4)
        transformer_ffn_multiplier = 2
        transformer_dropout = 0.0
        successor_hidden_dims = (128, 128)
        height_decoder_hidden_dims = (128, 128)

    class algorithm(LeggedRobotCfgPPO.algorithm):
        estimation_weight = 1.0
        kl_weight = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'PIEActorCritic'
        algorithm_class_name = 'PIEPPO'
        experiment_name = 'lite3_pie'
        resume = False
        num_steps_per_env = 24
        max_iterations = 15000
        save_interval = 500
        completed_iteration_numbering = True
