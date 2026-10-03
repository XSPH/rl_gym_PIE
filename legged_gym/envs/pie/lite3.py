"""PIE observations and supervision on the original LeggedRobot lifecycle."""
from isaacgym import gymapi
from pathlib import Path

import numpy as np
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.pie.sensors_and_rollout import PIESensorsAndRollout
from legged_gym.pie.terrain import PIETerrain, TerrainSampler
from legged_gym.pie.kinematics import UrdfKinematics


class Lite3PIE(LeggedRobot, PIESensorsAndRollout):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.config = cfg.pie
        # Native configuration fields own simulation, control and commands.
        self.config.num_envs = cfg.env.num_envs
        self.config.device, self.config.headless = sim_device, headless
        self.config.seed = cfg.seed
        # Serialized convenience values mirror native cfg; they never override it.
        self.config.episode_seconds = cfg.env.episode_length_s
        self.config.physics_dt = sim_params.dt
        self.config.decimation = cfg.control.decimation
        self.config.command_seconds = cfg.commands.resampling_time
        self.config.forward_velocity = list(cfg.commands.ranges.lin_vel_x)
        self.config.yaw_velocity = list(cfg.commands.ranges.ang_vel_yaw)
        self.config.angular_velocity_scale = cfg.normalization.obs_scales.ang_vel
        self.config.joint_velocity_scale = cfg.normalization.obs_scales.dof_vel
        self.config.observation_noise = cfg.noise.add_noise
        self.config.terrain.levels = cfg.terrain.num_rows
        self.config.terrain.variants = max(1, cfg.terrain.num_cols // len(self.config.terrain.kinds))
        self.config.terrain.length = cfg.terrain.terrain_length
        self.config.terrain.width = cfg.terrain.terrain_width
        self.config.terrain.spacing = 0.0
        self.config.terrain.resolution = cfg.terrain.horizontal_scale
        self.config.terrain.initial_max_level = cfg.terrain.max_init_terrain_level
        self.config.terrain.curriculum = cfg.terrain.curriculum
        self.config.terrain.scan_x = list(cfg.terrain.measured_points_x)
        self.config.terrain.scan_y = list(cfg.terrain.measured_points_y)
        self.config.robot.base_height = cfg.init_state.pos[2]
        self.config.robot.stand_angles = [cfg.init_state.default_joint_angles[name]
                                         for name in self.config.robot.joint_names]
        self.config.robot.action_scale = cfg.control.action_scale
        self.config.robot.action_clip = cfg.normalization.clip_actions
        self.config.randomization.friction = list(cfg.domain_rand.friction_range)
        self.config.randomization.payload = list(cfg.domain_rand.added_mass_range)
        self.config.validate()
        asset = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(Path(__file__).resolve().parents[3]))
        self.urdf = Path(asset).expanduser().resolve()
        if not self.urdf.is_file():
            raise FileNotFoundError("Robot URDF does not exist: {}".format(self.urdf))
        self.config.robot.urdf = str(self.urdf)
        self.rng = np.random.default_rng(cfg.seed)
        self._pie_ready = False
        try:
            super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
            self.metadata = {'base': 'unitreerobotics/unitree_rl_gym',
                             'class': 'Lite3PIE(LeggedRobot)', 'robot': 'lite3',
                             'depth': 'Warp optical-axis metres',
                             'training_flow': 'native-v1'}
            # The native runner calls BaseTask.reset once before collection.
        except Exception:
            self.close()
            raise

    def _create_ground_plane(self):
        # Unitree's base create_sim calls this hook even for non-plane tasks.
        self.terrain = PIETerrain(self.cfg.terrain, self.num_envs,
                                  self.config.terrain, self.cfg.seed)
        self.atlas = self.terrain.atlas
        self.terrain_sampler = TerrainSampler(self.atlas, self.device)
        self.height_samples = self.terrain_sampler.heights
        params = gymapi.TriangleMeshParams()
        params.nb_vertices, params.nb_triangles = len(self.atlas.vertices), len(self.atlas.triangles)
        params.static_friction = self.cfg.terrain.static_friction
        params.dynamic_friction = self.cfg.terrain.dynamic_friction
        params.restitution = self.cfg.terrain.restitution
        # Atlas vertices are already in world coordinates, including -border.
        self.gym.add_triangle_mesh(self.sim, self.atlas.vertices.reshape(-1),
                                   self.atlas.triangles.reshape(-1), params)

    def _get_env_origins(self):
        cfg = self.cfg.terrain
        maximum = min(cfg.max_init_terrain_level, cfg.num_rows - 1)
        self.terrain_levels = torch.randint(0, maximum + 1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                       self.num_envs / cfg.num_cols, rounding_mode='floor').long()
        self.terrain_origins = self.terrain_sampler.origins
        self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types].clone()
        # Origins contain terrain elevation only; _reset_root_states adds init z.
        self.custom_origins = True

    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)
        if self.config.randomization.enabled:
            shift = self.rng.uniform(-self.config.randomization.com_shift,
                                     self.config.randomization.com_shift, 3)
            props[0].com.x += float(shift[0])
            props[0].com.y += float(shift[1])
            props[0].com.z += float(shift[2])
        return props

    def _init_buffers(self):
        super()._init_buffers()
        self.body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        self.joint_order = torch.tensor([self.dof_names.index(name) for name in self.config.robot.joint_names],
                                        device=self.device, dtype=torch.long)
        self.inverse_joint_order = torch.argsort(self.joint_order)
        self.foot_indices = torch.tensor([self.body_names.index(name) for name in self.config.robot.foot_names],
                                        device=self.device, dtype=torch.long)
        self.fk = UrdfKinematics(self.urdf, self.config.robot.joint_names,
                                self.config.robot.base_name, self.device)
        self._init_pie_buffers()
        # Import lazily: CPU lifecycle tests need no Warp initialization.
        from legged_gym.pie.warp_camera import WarpDepthCamera
        self.camera = WarpDepthCamera(self.atlas, self.num_envs, self.config.camera, self.device)

    def _get_noise_scale_vec(self, cfg):
        self.add_noise = cfg.noise.add_noise
        noise, level, scales = cfg.noise.noise_scales, cfg.noise.noise_level, self.obs_scales
        vector = torch.zeros(self.num_obs, device=self.device)
        vector[:3] = noise.ang_vel * level * scales.ang_vel
        vector[3:6] = noise.gravity * level
        vector[9:21] = noise.dof_pos * level * scales.dof_pos
        vector[21:33] = noise.dof_vel * level * scales.dof_vel
        return vector

    def _compute_torques(self, actions):
        """PD through the native hook, with PIE delay and actuator randomization."""
        self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
        self.action_queue[:, 0] = actions
        delayed = self.action_queue[torch.arange(self.num_envs, device=self.device), self.delay_steps]
        delayed_native = delayed[:, self.inverse_joint_order]
        if self.cfg.control.control_type != 'P':
            return super()._compute_torques(delayed_native)
        target = (self.default_dof_pos + self.cfg.control.action_scale * delayed_native).clamp(
            self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
        # Random factors are defined in policy joint order; output is URDF order.
        kp = self.p_gains * self.kp_factors[:, self.inverse_joint_order]
        kd = self.d_gains * self.kd_factors[:, self.inverse_joint_order]
        torque = (kp * (target - self.dof_pos) - kd * self.dof_vel)
        torque *= self.motor_factors[:, self.inverse_joint_order]
        return torque.clamp(-self.torque_limits, self.torque_limits)

    def check_termination(self):
        super().check_termination()
        contact = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices], dim=-1) > 1, dim=1)
        tilted = (self.rpy[:, 1].abs() > 1.0) | (self.rpy[:, 0].abs() > 0.8)
        nonfinite = (~torch.isfinite(self.root_states).all(-1)
                     | ~torch.isfinite(self.dof_pos).all(-1)
                     | ~torch.isfinite(self.dof_vel).all(-1))
        failure = contact | tilted | nonfinite
        # A simultaneous failure at the time limit is a terminal, not timeout.
        self.time_out_buf &= ~failure
        self.reset_buf |= nonfinite
        self._termination_reasons = {'base_contact': contact, 'tilted': tilted,
                                     'nonfinite': nonfinite, 'timeout': self.time_out_buf.clone()}

    def _before_reset(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        finite = (torch.isfinite(self.root_states).all(-1)
                  & torch.isfinite(self.dof_pos).all(-1)
                  & torch.isfinite(self.dof_vel).all(-1)
                  & torch.isfinite(self.rigid_body_states).flatten(1).all(-1)
                  & torch.isfinite(self.rew_buf))
        if not bool(finite.all()):
            ids = (~finite).nonzero(as_tuple=False).flatten().tolist()
            raise FloatingPointError("Nonfinite physics/reward state in PIE environments {}".format(ids[:16]))
        if self.common_step_counter % self.config.camera.update_every == 0:
            self._render_depth()
        self._pre_reset_proprio = self._proprio()
        targets = self._terrain_targets()
        terminal_critic = self._critic_observation(self._pre_reset_proprio, targets)
        self.extras.pop('episode', None)
        self.extras['time_outs'] = self.time_out_buf.clone()
        self.extras['pie'] = {
            'terminal_proprio': self._pre_reset_proprio.clone(),
            'terminal_critic': terminal_critic,
            'reset_mask': self.reset_buf.bool().clone(),
            'terrain_level': self.terrain_levels.clone(),
            'termination_reasons': {key: value.clone() for key, value in self._termination_reasons.items()},
        }
        # Record a_{t-1} before native post_physics_step overwrites last_actions.
        self.last_last_actions.copy_(self.last_actions)

    def compute_observations(self):
        self._sync_base_quantities()
        proprio = self._proprio() if self._pre_reset_proprio is None else self._pre_reset_proprio.clone()
        ids = self._pending_reset.nonzero(as_tuple=False).flatten()
        if len(ids):
            proprio[ids] = self._proprio()[ids]
        self.obs_buf.copy_(proprio)
        self.proprio_history[:, :-1] = self.proprio_history[:, 1:].clone()
        self.proprio_history[:, -1] = proprio
        if len(ids):
            self.proprio_history[ids] = proprio[ids, None]
        self._pending_reset.zero_()
        self._pre_reset_proprio = None
        self._current_targets = self._terrain_targets()
        self.privileged_obs_buf.copy_(self._critic_observation(proprio, self._current_targets))

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if self._pie_ready and self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        super().reset_idx(env_ids)
        if self._pie_ready:
            self._reset_pie_sensors(env_ids)
            self.extras['episode']['terrain_level'] = self.terrain_levels.float().mean()

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return
        # An explicit initial reset after construction must not downgrade rows.
        active = self.episode_length_buf[env_ids] > 0
        ids = env_ids[active]
        if len(ids) == 0:
            return
        distance = torch.norm(self.root_states[ids, :2] - self.env_origins[ids, :2], dim=1)
        up = distance > self.terrain.env_length / 2
        down = ((distance < torch.norm(self.commands[ids, :2], dim=1)
                 * self.max_episode_length_s * 0.5) & ~up)
        levels = self.terrain_levels[ids] + up.long() - down.long()
        levels = torch.where(levels >= self.cfg.terrain.num_rows,
                             torch.randint_like(levels, self.cfg.terrain.num_rows), levels.clamp_min(0))
        self.terrain_levels[ids] = levels
        self.env_origins[ids] = self.terrain_origins[levels, self.terrain_types[ids]]

    def _reward_joint_power(self):
        return (self.torques.abs() * self.dof_vel.abs()).sum(-1)

    def _reward_smoothness(self):
        return (self.actions - 2 * self.last_actions + self.last_last_actions).square().sum(-1)
