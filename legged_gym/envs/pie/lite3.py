"""PIE extension of Unitree's real LeggedRobot initialization and actors."""
from isaacgym import gymapi
import numpy as np
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.pie.sensors_and_rollout import PIESensorsAndRollout
from legged_gym.pie.terrain import build_atlas, TerrainSampler
from legged_gym.pie.warp_camera import WarpDepthCamera
from legged_gym.pie.kinematics import UrdfKinematics


class Lite3PIE(LeggedRobot, PIESensorsAndRollout):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.config = cfg.pie
        self.config.num_envs = cfg.env.num_envs
        self.config.device, self.config.headless = sim_device, headless
        self.config.seed = cfg.seed
        self.config.episode_seconds = cfg.env.episode_length_s
        self.config.validate()
        self.urdf = self.config.resolve_urdf()
        # Resolve the final PIE settings before the base class creates actors.
        # The simulation asset and reset-time FK must use the same URDF.
        cfg.asset.file = str(self.urdf)
        cfg.init_state.pos = [0.0, 0.0, self.config.robot.base_height]
        cfg.init_state.default_joint_angles = dict(
            zip(self.config.robot.joint_names, self.config.robot.stand_angles))
        cfg.control.stiffness = {'joint': self.config.robot.kp}
        cfg.control.damping = {'joint': self.config.robot.kd}
        cfg.control.action_scale = self.config.robot.action_scale
        cfg.control.decimation = self.config.decimation
        randomization = self.config.randomization
        cfg.domain_rand.randomize_friction = randomization.enabled
        cfg.domain_rand.friction_range = list(randomization.friction)
        cfg.domain_rand.randomize_base_mass = randomization.enabled
        cfg.domain_rand.added_mass_range = list(randomization.payload)
        sim_params.dt = self.config.physics_dt
        cfg.env.num_observations = 45
        cfg.env.num_privileged_obs = 45 + 3 + self.config.map_size
        cfg.env.num_actions = 12
        self.rng = np.random.default_rng(self.config.seed)
        self.atlas = build_atlas(self.config.terrain, self.config.seed)
        self.terrain = TerrainSampler(self.atlas, sim_device)
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
        try:
            super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
            self.env_handles = self.envs
            self.asset_joint_names = self.dof_names
            self.body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
            order = [self.dof_names.index(name) for name in self.config.robot.joint_names]
            self.joint_order = torch.tensor(order, device=self.device, dtype=torch.long)
            self.foot_indices = torch.tensor([self.body_names.index(name) for name in self.config.robot.foot_names], device=self.device)
            self.base_index = self.body_names.index(self.config.robot.base_name)
            self.collision_indices = torch.tensor([i for i, name in enumerate(self.body_names) if name not in self.config.robot.foot_names], device=self.device)
            self.joint_lower = self.dof_pos_limits[self.joint_order, 0].clone()
            self.joint_upper = self.dof_pos_limits[self.joint_order, 1].clone()
            self.torque_limits = self.torque_limits[self.joint_order].clone().clamp(max=self.config.robot.torque_limit)
            self.actor_indices = torch.tensor([self.gym.get_actor_index(e, a, gymapi.DOMAIN_SIM) for e, a in zip(self.envs, self.actor_handles)], dtype=torch.int32, device=self.device)
            self._acquire_buffers()
            self.camera = WarpDepthCamera(self.atlas, self.num_envs, self.config.camera, self.device)
            self.fk = UrdfKinematics(self.urdf, self.config.robot.joint_names, self.config.robot.base_name, self.device)
            self.metadata = {'base': 'unitreerobotics/unitree_rl_gym', 'class': 'Lite3PIE(LeggedRobot)', 'robot': 'lite3', 'depth': 'Warp optical-axis metres'}
            self.reset()
        except Exception:
            self.close()
            raise

    def _get_env_origins(self):
        self.levels = torch.randint(0, min(self.config.terrain.initial_max_level + 1, self.config.terrain.levels), (self.num_envs,), device=self.device)
        self.columns = torch.arange(self.num_envs, device=self.device) % len(self.atlas.kinds)
        self.env_origins = self.terrain.origins[self.levels, self.columns].clone()
        self.env_origins[:, 2] += self.config.robot.base_height
        self.custom_origins = True

    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)
        if self.config.randomization.enabled:
            shift = self.rng.uniform(-self.config.randomization.com_shift, self.config.randomization.com_shift, 3)
            props[0].com.x += float(shift[0])
            props[0].com.y += float(shift[1])
            props[0].com.z += float(shift[2])
        return props

    def get_observations(self):
        return self._observations(self.proprio_history[:, -1])

    _create_ground_plane = PIESensorsAndRollout._create_ground
    step = PIESensorsAndRollout.step
    reset = PIESensorsAndRollout.reset
    close = PIESensorsAndRollout.close
