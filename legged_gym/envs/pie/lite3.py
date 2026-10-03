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
        self._validate_config(cfg, sim_params, sim_device)
        asset = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(Path(__file__).resolve().parents[3]))
        self.urdf = Path(asset).expanduser().resolve()
        if not self.urdf.is_file():
            raise FileNotFoundError("Robot URDF does not exist: {}".format(self.urdf))
        # The native actor loader and reset-time FK read the same asset entry.
        cfg.asset.file = str(self.urdf)
        cfg.env.num_privileged_obs = 48 + len(cfg.terrain.measured_points_x) * len(cfg.terrain.measured_points_y)
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

    @staticmethod
    def _validate_config(cfg, sim_params, sim_device):
        if cfg.env.num_envs < 1 or not sim_device.startswith("cuda:"):
            raise ValueError("Isaac Gym GPU pipeline requires num_envs >= 1 and cuda:<index>.")
        if cfg.env.num_actions != 12 or cfg.env.num_observations != 45:
            raise ValueError("PIE requires 12 actions and 45 proprio observations.")
        joints, feet = cfg.asset.joint_names, cfg.asset.foot_names
        if len(joints) != 12 or len(set(joints)) != 12:
            raise ValueError("Exactly 12 distinct policy joint names are required.")
        if len(feet) != 4 or len(set(feet)) != 4 or any(
                name not in cfg.init_state.default_joint_angles for name in joints):
            raise ValueError("Expected 4 distinct feet and 12 configured stand angles.")
        if cfg.control.decimation < 1 or sim_params.dt <= 0 or cfg.env.proprio_history < 1:
            raise ValueError("Invalid simulation timing or proprio history.")
        r, c, terrain = cfg.domain_rand, cfg.camera, cfg.terrain
        if r.max_delay_seconds < 0:
            raise ValueError("Action delay must be nonnegative.")
        for name in ("friction_range", "added_mass_range", "gain_factor", "motor_factor",
                     "camera_hfov_degrees"):
            bounds = getattr(r, name)
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError("Invalid randomization range: " + name)
        if c.history != 2 or c.update_every < 1 or c.latency_frames < 0:
            raise ValueError("PIE requires depth history 2 and positive update_every.")
        if not (0 < c.near < c.far and c.height > 1 and c.width > 1):
            raise ValueError("Invalid depth clipping or image resolution.")
        if terrain.num_rows < 1 or terrain.num_cols < 1:
            raise ValueError("Terrain rows and columns must be positive.")
        if terrain.terrain_length < 7 or terrain.terrain_width < 1:
            raise ValueError("Course length >= 7 m and width >= 1 m are required.")
        if not terrain.kinds or set(terrain.kinds) - {"flat", "gap", "step", "hurdle", "stairs"}:
            raise ValueError("Unsupported terrain kinds.")
        if not terrain.measured_points_x or not terrain.measured_points_y:
            raise ValueError("Height scan cannot be empty.")

    def _create_ground_plane(self):
        # Unitree's base create_sim calls this hook even for non-plane tasks.
        self.terrain = PIETerrain(self.cfg.terrain, self.num_envs, self.cfg.seed)
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
        if self.cfg.domain_rand.randomize_pie:
            shift = self.rng.uniform(-self.cfg.domain_rand.com_shift,
                                     self.cfg.domain_rand.com_shift, 3)
            props[0].com.x += float(shift[0])
            props[0].com.y += float(shift[1])
            props[0].com.z += float(shift[2])
        return props

    def _init_buffers(self):
        super()._init_buffers()
        self.body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        self.joint_order = torch.tensor([self.dof_names.index(name) for name in self.cfg.asset.joint_names],
                                        device=self.device, dtype=torch.long)
        self.inverse_joint_order = torch.argsort(self.joint_order)
        self.foot_indices = torch.tensor([self.body_names.index(name) for name in self.cfg.asset.foot_names],
                                        device=self.device, dtype=torch.long)
        self.fk = UrdfKinematics(self.urdf, self.cfg.asset.joint_names,
                                self.cfg.asset.base_name, self.device)
        self._init_pie_buffers()
        # Import lazily: CPU lifecycle tests need no Warp initialization.
        from legged_gym.pie.warp_camera import WarpDepthCamera
        self.camera = WarpDepthCamera(self.atlas, self.num_envs, self.cfg.camera, self.device)

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
        if self.common_step_counter % self.cfg.camera.update_every == 0:
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
