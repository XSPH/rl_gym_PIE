"""PIE sensor and supervision helpers; simulation stays in LeggedRobot.

This component deliberately has no step/reset/reward implementation. The task
adds observations around the native lifecycle, including a pre-reset snapshot.
"""
import math

from isaacgym import gymtorch
import torch

from .math_utils import quat_mul, quat_rotate, quat_rotate_inverse, quat_yaw


class PIESensorsAndRollout:
    @property
    def action_dim(self):
        return self.num_actions

    def _init_pie_buffers(self):
        n, c = self.num_envs, self.config.camera
        self.rigid_body_states = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)).view(n, self.num_bodies, 13)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.proprio_history = torch.zeros((n, self.config.proprio_history, self.num_obs), device=self.device)
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
        capacity = int(math.floor(self.config.randomization.max_delay_seconds / self.sim_params.dt + 1e-9)) + 1
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
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

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
        heightmap = (self.root_states[:, 2:3] - ground - self.config.heightmap_offset).clamp(-1.0, 1.0)
        feet = self.rigid_body_states[:, self.foot_indices, :3]
        clearance = (feet[..., 2] - self.terrain_sampler.sample(feet)
                     - self.config.robot.foot_radius).clamp(0.0, 2.0)
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
        self.camera.focal.copy_(self.config.camera.width
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
        count, r = len(ids), self.config.randomization
        self.last_last_actions[ids] = 0
        self.action_queue[ids] = 0
        self.torques[ids] = 0
        self.last_contacts[ids] = False
        self.last_root_vel[ids] = 0
        c = self.config.camera
        if r.enabled:
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
