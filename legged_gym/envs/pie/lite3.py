"""PIE observations and supervision on the original LeggedRobot lifecycle."""
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_mul, quat_apply as quat_rotate, quat_conjugate
from pathlib import Path
import math

import numpy as np
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import PIETerrain, TerrainSampler
from legged_gym.utils.kinematics import UrdfKinematics
from legged_gym.utils.math import quat_yaw


class Lite3PIE(LeggedRobot):
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
        from legged_gym.utils.warp_camera import WarpDepthCamera
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

    @property
    def action_dim(self):
        return self.num_actions

    def _init_pie_buffers(self):
        n, c = self.num_envs, self.cfg.camera
        self.rigid_body_states = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)).view(n, self.num_bodies, 13)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.proprio_history = torch.zeros((n, self.cfg.env.proprio_history, self.num_obs), device=self.device)
        fill = 0.5 if c.normalize else c.far
        self.depth_history = torch.full((n, c.history, c.height, c.width), fill, device=self.device)
        self.depth_queue = torch.full((n, c.latency_frames + 1, c.height, c.width), fill, device=self.device)
        self.depth_frame_ids = torch.full((n, c.history), -1, device=self.device, dtype=torch.long)
        self.depth_queue_frame_ids = torch.full((n, c.latency_frames + 1), -1,
                                               device=self.device, dtype=torch.long)
        self.camera_capture_serial = 0
        self.kp_factors = torch.ones_like(self.actions)
        self.kd_factors = torch.ones_like(self.actions)
        self.motor_factors = torch.ones_like(self.actions)
        capacity = int(math.floor(self.cfg.domain_rand.max_delay_seconds / self.sim_params.dt + 1e-9)) + 1
        self.action_queue = torch.zeros((n, max(1, capacity), self.num_actions), device=self.device)
        self.delay_steps = torch.zeros(n, device=self.device, dtype=torch.long)
        self.camera_offsets = torch.tensor(c.position, device=self.device).expand(n, -1).clone()
        self.camera_pitch = torch.full((n,), math.radians(c.pitch_degrees), device=self.device)
        self.camera_fov = torch.full((n,), c.hfov_degrees, device=self.device)
        xs = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device)
        ys = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device)
        xx, yy = torch.meshgrid(xs, ys, indexing='ij')
        self.scan_points = torch.stack((xx.flatten(), yy.flatten(), torch.zeros_like(xx.flatten())), -1)
        self._pending_reset = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._pre_reset_proprio = None
        self._current_targets = None
        self._pie_ready = True

    def _uniform(self, shape, bounds):
        return torch.rand(shape, device=self.device) * (bounds[1] - bounds[0]) + bounds[0]

    def _sync_base_quantities(self):
        """Indexed reset/push setters do not update derived base velocities."""
        self.base_lin_vel[:] = quat_rotate(quat_conjugate(self.base_quat), self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate(quat_conjugate(self.base_quat), self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate(quat_conjugate(self.base_quat), self.gravity_vec)

    def _proprio(self):
        proprio = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                             self.projected_gravity,
                             self.commands[:, :3] * self.commands_scale,
                             (self.dof_pos[:, self.joint_order] - self.default_dof_pos[:, self.joint_order])
                             * self.obs_scales.dof_pos,
                             self.dof_vel[:, self.joint_order] * self.obs_scales.dof_vel,
                             self.actions), dim=-1)
        if self.add_noise:
            proprio = proprio + (2 * torch.rand_like(proprio) - 1) * self.noise_scale_vec
        clip = self.cfg.normalization.clip_observations
        return proprio.clamp(-clip, clip)

    def _terrain_targets(self):
        yaw = quat_yaw(self.base_quat)
        scan = self.scan_points[None].expand(self.num_envs, -1, -1).clone()
        x, y = scan[..., 0].clone(), scan[..., 1].clone()
        scan[..., 0] = torch.cos(yaw[:, None]) * x - torch.sin(yaw[:, None]) * y
        scan[..., 1] = torch.sin(yaw[:, None]) * x + torch.cos(yaw[:, None]) * y
        scan += self.root_states[:, None, :3]
        ground = self.terrain_sampler.sample(scan)
        heightmap = (self.root_states[:, 2:3] - ground - self.cfg.terrain.heightmap_offset).clamp(-1.0, 1.0)
        feet = self.rigid_body_states[:, self.foot_indices, :3]
        clearance = (feet[..., 2] - self.terrain_sampler.sample(feet)
                     - self.cfg.asset.foot_radius).clamp(0.0, 2.0)
        return {'velocity': self.base_lin_vel.clone(), 'foot_clearance': clearance,
                'heightmap': heightmap}

    def _critic_observation(self, proprio, targets):
        clip = self.cfg.normalization.clip_observations
        return torch.cat((proprio, targets['velocity'], targets['heightmap']), -1).clamp(-clip, clip)

    def _render_depth(self, ids=None, reset=False):
        q = self.base_quat
        self.camera.positions.copy_(self.root_states[:, :3] + quat_rotate(q, self.camera_offsets))
        pitch = self.camera_pitch / 2
        cq = torch.stack((torch.zeros_like(pitch), torch.sin(pitch),
                          torch.zeros_like(pitch), torch.cos(pitch)), -1)
        self.camera.orientations.copy_(quat_mul(q, cq))
        self.camera.focal.copy_(self.cfg.camera.width
                               / (2 * torch.tan(self.camera_fov * (math.pi / 360))))
        image = self.camera.encode(self.camera.render(ids))
        self.camera_capture_serial += 1
        captured = self.camera_capture_serial * self.num_envs + torch.arange(self.num_envs, device=self.device)
        if reset:
            self.depth_history[ids] = image[ids, None]
            self.depth_queue[ids] = image[ids, None]
            self.depth_frame_ids[ids] = captured[ids, None]
            self.depth_queue_frame_ids[ids] = captured[ids, None]
        else:
            self.depth_queue[:, 1:] = self.depth_queue[:, :-1].clone()
            self.depth_queue[:, 0] = image
            self.depth_queue_frame_ids[:, 1:] = self.depth_queue_frame_ids[:, :-1].clone()
            self.depth_queue_frame_ids[:, 0] = captured
            self.depth_history[:, :-1] = self.depth_history[:, 1:].clone()
            self.depth_history[:, -1] = self.depth_queue[:, -1]
            self.depth_frame_ids[:, :-1] = self.depth_frame_ids[:, 1:].clone()
            self.depth_frame_ids[:, -1] = self.depth_queue_frame_ids[:, -1]

    def _reset_pie_sensors(self, ids):
        count, r = len(ids), self.cfg.domain_rand
        self.last_last_actions[ids] = 0
        self.action_queue[ids] = 0
        self.torques[ids] = 0
        self.last_contacts[ids] = False
        self.last_root_vel[ids] = 0
        c = self.cfg.camera
        if r.randomize_pie:
            self.kp_factors[ids] = self._uniform((count, self.num_actions), r.gain_factor)
            self.kd_factors[ids] = self._uniform((count, self.num_actions), r.gain_factor)
            self.motor_factors[ids] = self._uniform((count, self.num_actions), r.motor_factor)
            self.delay_steps[ids] = torch.randint(self.action_queue.shape[1], (count,), device=self.device)
            self.camera_offsets[ids] = torch.tensor(c.position, device=self.device) + self._uniform((count, 3), [-r.camera_position, r.camera_position])
            delta = math.radians(r.camera_pitch_degrees)
            self.camera_pitch[ids] = math.radians(c.pitch_degrees) + self._uniform((count,), [-delta, delta])
            self.camera_fov[ids] = self._uniform((count,), r.camera_hfov_degrees)
        else:
            self.kp_factors[ids] = self.kd_factors[ids] = self.motor_factors[ids] = 1
            self.delay_steps[ids] = 0
            self.camera_offsets[ids] = torch.tensor(c.position, device=self.device)
            self.camera_pitch[ids] = math.radians(c.pitch_degrees)
            self.camera_fov[ids] = c.hfov_degrees
        # PhysX rigid-body tensors still contain the old episode immediately
        # after indexed setters. FK updates selected rows only, without stepping.
        states = self.fk.forward(self.root_states[ids, :3], self.base_quat[ids],
                                 self.dof_pos[ids][:, self.joint_order])
        self.rigid_body_states[ids] = 0
        for index, name in enumerate(self.body_names):
            if name in states:
                position, orientation = states[name]
                self.rigid_body_states[ids, index, :3] = position
                self.rigid_body_states[ids, index, 3:7] = orientation
        self.contact_forces[ids] = 0
        self._sync_base_quantities()
        self._render_depth(ids, reset=True)
        self._pending_reset[ids] = True

    def get_pie_observations(self):
        """Current-state input and privileged labels; never called by base tasks."""
        if self._current_targets is None:
            self.compute_observations()
        return {'proprio': self.obs_buf.clone(),
                'proprio_history': self.proprio_history.clone(),
                'depth': self.depth_history.clone(),
                'depth_frame_ids': self.depth_frame_ids.clone(),
                'critic': self.privileged_obs_buf.clone(),
                'targets': {key: value.clone() for key, value in self._current_targets.items()}}

    def close(self):
        if getattr(self, 'viewer', None) is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None
        if getattr(self, 'sim', None) is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
