"""Batched PhysX task with deployment observations and training-only targets."""
# Isaac Gym explicitly requires its native bindings to be imported before torch.
try:
    from isaacgym import gymapi, gymtorch
except ImportError as error:
    raise ImportError("Install Isaac Gym Preview 4 in a separate Python 3.8 environment before loading this backend.") from error

import math
from pathlib import Path

import numpy as np
import torch

from .config import EnvConfig
from .kinematics import UrdfKinematics
from .math_utils import quat_mul, quat_rotate, quat_rotate_inverse, quat_yaw
from .terrain import TerrainSampler, build_atlas
from .warp_camera import WarpDepthCamera


class PIESensorsAndRollout:
    """Gymnasium-style 5-tuple API, automatically resetting completed episodes.

    ``info['terminal_observation']`` contains the pre-reset observation for all
    environments, selected by ``reset_mask``. ``terminal_proprio`` is the real
    successor label for *every* transition, including terminal transitions.
    Current-state targets live under observation['targets'] and are never
    included in the actor's proprio/history/depth inputs.
    """


    @property
    def action_dim(self):
        return 12

    def _create_ground(self):
        params = gymapi.TriangleMeshParams()
        params.nb_vertices = len(self.atlas.vertices)
        params.nb_triangles = len(self.atlas.triangles)
        params.static_friction = 1.0
        params.dynamic_friction = 1.0
        params.restitution = 0.0
        # TerrainAtlas includes the floor, so a second plane is unnecessary.
        self.gym.add_triangle_mesh(self.sim, self.atlas.vertices.reshape(-1),
                                   self.atlas.triangles.reshape(-1), params)


    def _acquire_buffers(self):
        n = self.num_envs
        self.root = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).view(n, 13)
        self.dof = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(n, 12, 2)
        self.body = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(n, self.num_bodies, 13)
        self.contacts = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).view(n, self.num_bodies, 3)
        self._refresh()
        self.stand = torch.tensor(self.config.robot.stand_angles, device=self.device)
        self.gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(n, -1)
        self.actions = torch.zeros((n, 12), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.previous_previous_actions = torch.zeros_like(self.actions)
        self.previous_velocity = torch.zeros_like(self.actions)
        self.torques = torch.zeros_like(self.actions)
        self.effort = torch.zeros_like(self.actions)
        self.commands = torch.zeros((n, 3), device=self.device)
        # Original LeggedRobot amplitudes, mapped to PIE's 45-D ordering
        # (base velocity is estimated, so it is absent from the actor input).
        noise = self.cfg.noise.noise_scales
        level = self.cfg.noise.noise_level
        self.proprio_noise_scale = torch.zeros(45, device=self.device)
        self.proprio_noise_scale[:3] = noise.ang_vel * level * self.config.angular_velocity_scale
        self.proprio_noise_scale[3:6] = noise.gravity * level
        self.proprio_noise_scale[9:21] = noise.dof_pos * level * self.obs_scales.dof_pos
        self.proprio_noise_scale[21:33] = noise.dof_vel * level * self.config.joint_velocity_scale
        self.episode_steps = torch.zeros(n, device=self.device, dtype=torch.long)
        self.episode_returns = torch.zeros(n, device=self.device)
        self.proprio_history = torch.zeros((n, self.config.proprio_history, 45), device=self.device)
        c = self.config.camera
        self.depth_history = torch.full((n, c.history, c.height, c.width), 0.5 if c.normalize else c.far, device=self.device)
        self.depth_queue = torch.full((n, c.latency_frames + 1, c.height, c.width),
                                      0.5 if c.normalize else c.far, device=self.device)
        self.kp_factors = torch.ones((n, 12), device=self.device)
        self.kd_factors = torch.ones_like(self.kp_factors)
        self.motor_factors = torch.ones_like(self.kp_factors)
        self.delay_steps = torch.zeros(n, device=self.device, dtype=torch.long)
        delay_capacity = int(math.floor(
            self.config.randomization.max_delay_seconds / self.config.physics_dt + 1e-9)) + 1
        self.action_queue = torch.zeros((n, max(1, delay_capacity), 12), device=self.device)
        self.camera_offsets = torch.tensor(c.position, device=self.device).expand(n, -1).clone()
        self.camera_pitch = torch.full((n,), math.radians(c.pitch_degrees), device=self.device)
        self.camera_fov = torch.full((n,), c.hfov_degrees, device=self.device)
        xs = torch.tensor(self.config.terrain.scan_x, device=self.device)
        ys = torch.tensor(self.config.terrain.scan_y, device=self.device)
        xx, yy = torch.meshgrid(xs, ys, indexing="ij")
        self.scan_points = torch.stack((xx.flatten(), yy.flatten(), torch.zeros_like(xx.flatten())), -1)
        self.control_steps = 0

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _uniform(self, shape, bounds):
        return torch.rand(shape, device=self.device) * (bounds[1] - bounds[0]) + bounds[0]

    def _sample_commands(self, ids):
        self.commands[ids, 0] = self._uniform((len(ids),), self.config.forward_velocity)
        self.commands[ids, 1] = 0.0
        self.commands[ids, 2] = self._uniform((len(ids),), self.config.yaw_velocity)
        self.commands[ids, :2] *= (self.commands[ids, :2].norm(dim=-1) > 0.2).unsqueeze(-1)

    def _update_goal_commands(self):
        if self.config.command_mode == "goal":
            goals = self.terrain.goals[self.levels, self.columns]
            delta = goals[:, :2] - self.root[:, :2]
            error = torch.atan2(delta[:, 1], delta[:, 0]) - quat_yaw(self.root[:, 3:7])
            error = torch.atan2(torch.sin(error), torch.cos(error))
            self.commands[:, 2] = (2 * error).clamp(*self.config.yaw_velocity)

    def _reset_indices(self, ids, curriculum=False, success=None):
        if ids.numel() == 0:
            return
        cfg = self.config
        if curriculum and cfg.terrain.curriculum:
            successful = success[ids]
            reached_x = self.root[ids, 0] - self.levels[ids] * (cfg.terrain.length + cfg.terrain.spacing)
            failed_early = ~successful & (reached_x < cfg.terrain.length / 2)
            self.levels[ids] = (self.levels[ids] + successful.long() - failed_early.long()).clamp(0, cfg.terrain.levels - 1)
        count = len(ids)
        self.root[ids] = 0.0
        self.root[ids, :3] = self.terrain.origins[self.levels[ids], self.columns[ids]]
        self.root[ids, 2] += cfg.robot.base_height
        if cfg.randomization.enabled:
            self.root[ids, 1] += self._uniform((count,), [-0.1, 0.1])
            # Restore the native reset-time linear/angular velocity variation.
            self.root[ids, 7:13] = self._uniform((count, 6), [-0.5, 0.5])
        self.root[ids, 6] = 1.0
        factor = self._uniform((count, 12), cfg.randomization.joint_position_factor) if cfg.randomization.enabled else 1.0
        q = (self.stand.expand(count, -1) * factor).clamp(self.joint_lower, self.joint_upper)
        self.dof[ids[:, None], self.joint_order[None, :], 0] = q
        self.dof[ids, :, 1] = 0.0
        selected = self.actor_indices[ids].contiguous()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root),
                                                    gymtorch.unwrap_tensor(selected), len(ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof),
                                             gymtorch.unwrap_tensor(selected), len(ids))
        # FK produces valid reset-time labels without advancing other envs.
        states = self.fk.forward(self.root[ids, :3], self.root[ids, 3:7], q)
        self.body[ids] = 0.0
        for index, name in enumerate(self.body_names):
            if name in states:
                position, orientation = states[name]
                self.body[ids, index, :3] = position
                self.body[ids, index, 3:7] = orientation
        self.contacts[ids] = 0.0
        self.actions[ids] = 0.0
        self.previous_actions[ids] = 0.0
        self.previous_previous_actions[ids] = 0.0
        self.previous_velocity[ids] = 0.0
        self.torques[ids] = 0.0
        self.action_queue[ids] = 0.0
        self.episode_steps[ids] = 0
        self.episode_returns[ids] = 0.0
        self._sample_commands(ids)
        if cfg.randomization.enabled:
            r = cfg.randomization
            self.kp_factors[ids] = self._uniform((count, 12), r.gain_factor)
            self.kd_factors[ids] = self._uniform((count, 12), r.gain_factor)
            self.motor_factors[ids] = self._uniform((count, 12), r.motor_factor)
            self.delay_steps[ids] = torch.randint(0, self.action_queue.shape[1], (count,), device=self.device)
            self.camera_offsets[ids] = torch.tensor(cfg.camera.position, device=self.device) + self._uniform((count, 3), [-r.camera_position, r.camera_position])
            self.camera_pitch[ids] = math.radians(cfg.camera.pitch_degrees) + self._uniform((count,), [-math.radians(r.camera_pitch_degrees), math.radians(r.camera_pitch_degrees)])
            self.camera_fov[ids] = self._uniform((count,), r.camera_hfov_degrees)
        else:
            self.kp_factors[ids] = 1.0
            self.kd_factors[ids] = 1.0
            self.motor_factors[ids] = 1.0
            self.delay_steps[ids] = 0
            self.camera_offsets[ids] = torch.tensor(cfg.camera.position, device=self.device)
            self.camera_pitch[ids] = math.radians(cfg.camera.pitch_degrees)
            self.camera_fov[ids] = cfg.camera.hfov_degrees
        self._update_goal_commands()
        self._render_depth(ids, reset=True)
        proprio = self._proprio()
        self.proprio_history[ids] = proprio[ids, None, :]

    def reset(self):
        ids = torch.arange(self.num_envs, device=self.device)
        self._reset_indices(ids)
        return self._observations(self.proprio_history[:, -1])

    def _render_depth(self, ids=None, reset=False):
        q = self.root[:, 3:7]
        self.camera.positions.copy_(self.root[:, :3] + quat_rotate(q, self.camera_offsets))
        pitch = self.camera_pitch / 2
        cq = torch.stack((torch.zeros_like(pitch), torch.sin(pitch), torch.zeros_like(pitch), torch.cos(pitch)), -1)
        self.camera.orientations.copy_(quat_mul(q, cq))
        self.camera.focal.copy_(self.config.camera.width / (2 * torch.tan(self.camera_fov * (math.pi / 360))))
        image = self.camera.encode(self.camera.render(ids))
        if reset:
            self.depth_history[ids] = image[ids, None]
            self.depth_queue[ids] = image[ids, None]
        else:
            self.depth_queue[:, 1:] = self.depth_queue[:, :-1].clone()
            self.depth_queue[:, 0] = image
            self.depth_history[:, :-1] = self.depth_history[:, 1:].clone()
            self.depth_history[:, -1] = self.depth_queue[:, -1]

    def _base_quantities(self):
        q = self.root[:, 3:7]
        return (quat_rotate_inverse(q, self.root[:, 7:10]),
                quat_rotate_inverse(q, self.root[:, 10:13]), quat_rotate_inverse(q, self.gravity))

    def _apply_pushes(self):
        """Apply the original indexed velocity impulse to the due environments."""
        if not (self.config.randomization.enabled and self.cfg.domain_rand.push_robots):
            return
        interval = max(1, int(self.cfg.domain_rand.push_interval))
        ids = torch.nonzero(self.episode_steps % interval == 0).flatten()
        if ids.numel() == 0:
            return
        maximum = self.cfg.domain_rand.max_push_vel_xy
        # Updating only the selected rows also keeps cached states of other
        # robots consistent with PhysX (the stock helper modifies all rows).
        self.root[ids, 7:9] = self._uniform((len(ids), 2), [-maximum, maximum])
        selected = self.actor_indices[ids].contiguous()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root),
            gymtorch.unwrap_tensor(selected), len(ids))

    def _proprio(self):
        _, omega, gravity = self._base_quantities()
        joints = self.dof[:, self.joint_order]
        proprio = torch.cat((omega * self.config.angular_velocity_scale, gravity,
            self.commands * self.commands_scale,
            (joints[..., 0] - self.stand) * self.obs_scales.dof_pos,
            joints[..., 1] * self.config.joint_velocity_scale, self.actions), -1)
        if self.config.observation_noise and self.cfg.noise.add_noise:
            proprio = proprio + (2 * torch.rand_like(proprio) - 1) * self.proprio_noise_scale
        clip = self.cfg.normalization.clip_observations
        return proprio.clamp(-clip, clip)

    def _observations(self, proprio):
        velocity, _, _ = self._base_quantities()
        yaw = quat_yaw(self.root[:, 3:7])
        scan = self.scan_points[None].expand(self.num_envs, -1, -1).clone()
        x, y = scan[..., 0].clone(), scan[..., 1].clone()
        scan[..., 0] = torch.cos(yaw[:, None]) * x - torch.sin(yaw[:, None]) * y
        scan[..., 1] = torch.sin(yaw[:, None]) * x + torch.cos(yaw[:, None]) * y
        scan += self.root[:, None, :3]
        terrain_heights = self.terrain.sample(scan, self.levels, self.columns)
        heightmap = (self.root[:, 2:3] - terrain_heights - self.config.heightmap_offset).clamp(-1.0, 1.0)
        feet = self.body[:, self.foot_indices, :3]
        foot_clearance = (feet[..., 2] - self.terrain.sample(feet, self.levels, self.columns) - self.config.robot.foot_radius).clamp(0.0, 2.0)
        return {"proprio": proprio.clone(), "proprio_history": self.proprio_history.clone(),
                "depth": self.depth_history.clone(),
                "critic": torch.cat((proprio, velocity, heightmap), -1).clamp(
                    -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations),
                "targets": {"velocity": velocity.clone(), "foot_clearance": foot_clearance.clone(),
                            "heightmap": heightmap.clone()}}

    def _reward(self, action, prior, prior_prior, old_qd):
        velocity, omega, gravity = self._base_quantities()
        qd = self.dof[:, self.joint_order, 1]
        collisions = (torch.linalg.vector_norm(self.contacts[:, self.collision_indices], dim=-1) > 1.0).sum(-1)
        parts = {
            "tracking_linear": 1.5 * torch.exp(-4 * (self.commands[:, :2] - velocity[:, :2]).square().sum(-1)),
            "tracking_yaw": 0.5 * torch.exp(-4 * (self.commands[:, 2] - omega[:, 2]).square()),
            "vertical_velocity": -velocity[:, 2].square(),
            "angular_velocity": -0.05 * omega[:, :2].square().sum(-1),
            "orientation": -gravity[:, :2].square().sum(-1),
            "joint_acceleration": -2.5e-7 * ((qd - old_qd) / self.config.policy_dt).square().sum(-1),
            "joint_power": -2e-5 * (self.torques.abs() * qd.abs()).sum(-1),
            "collision": -10.0 * collisions,
            "action_rate": -0.01 * (action - prior).square().sum(-1),
            "smoothness": -0.01 * (action - 2 * prior + prior_prior).square().sum(-1),
        }
        reward = sum(parts.values())
        if self.cfg.rewards.only_positive_rewards:
            reward = reward.clamp_min(0.0)
        if self.config.reward_scale_dt:
            reward = reward * self.config.policy_dt
        return reward, parts

    @torch.no_grad()
    def step(self, action):
        cfg = self.config
        if action.shape != (self.num_envs, 12):
            raise ValueError("Expected action shape ({},12).".format(self.num_envs))
        action = action.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(action).all():
            raise ValueError("Actions contain nonfinite values.")
        action = action.clamp(-cfg.robot.action_clip, cfg.robot.action_clip)
        prior, prior_prior = self.actions.clone(), self.previous_actions.clone()
        old_qd = self.dof[:, self.joint_order, 1].clone()
        self.previous_previous_actions.copy_(prior_prior)
        self.previous_actions.copy_(prior)
        self.actions.copy_(action)
        for _ in range(cfg.decimation):
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = action
            delayed = self.action_queue[torch.arange(self.num_envs, device=self.device), self.delay_steps]
            target = (self.stand + cfg.robot.action_scale * delayed).clamp(self.joint_lower, self.joint_upper)
            q, qd = self.dof[:, self.joint_order, 0], self.dof[:, self.joint_order, 1]
            torque = ((target - q) * cfg.robot.kp * self.kp_factors - qd * cfg.robot.kd * self.kd_factors) * self.motor_factors
            self.torques.copy_(torque.clamp(-self.torque_limits, self.torque_limits))
            self.effort[:, self.joint_order] = self.torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.effort))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self._refresh()
        self.control_steps += 1
        self.episode_steps += 1
        reward, parts = self._reward(action, prior, prior_prior, old_qd)
        finite = (torch.isfinite(self.root).all(-1)
                  & torch.isfinite(self.dof).flatten(1).all(-1)
                  & torch.isfinite(self.body).flatten(1).all(-1)
                  & torch.isfinite(reward))
        # Invalid physics states terminate below. Their NaN/Inf reward must not
        # poison returns for the other trajectories before they reset.
        reward = torch.where(finite, reward, torch.zeros_like(reward))
        self.episode_returns += reward
        origins = self.terrain.origins[self.levels, self.columns]
        goals = self.terrain.goals[self.levels, self.columns]
        success = self.root[:, 0] >= goals[:, 0]
        _, _, gravity = self._base_quantities()
        base_contact = torch.linalg.vector_norm(self.contacts[:, self.base_index], dim=-1) > 1.0
        outside = (self.root[:, 1] - origins[:, 1]).abs() > cfg.terrain.width / 2
        outside |= self.root[:, 0] < origins[:, 0] - 0.6
        terminated = base_contact | (gravity[:, 2] > -0.3) | (self.root[:, 2] < cfg.terrain.floor_height + 0.15) | outside | ~finite | success
        truncated = (self.episode_steps >= math.ceil(cfg.episode_seconds / cfg.policy_dt)) & ~terminated
        done = terminated | truncated
        resample = torch.nonzero((self.episode_steps % max(1, round(cfg.command_seconds / cfg.policy_dt))) == 0).flatten()
        self._sample_commands(resample)
        self._update_goal_commands()
        self._apply_pushes()
        if self.control_steps % cfg.camera.update_every == 0:
            self._render_depth()
        proprio = self._proprio()
        self.proprio_history[:, :-1] = self.proprio_history[:, 1:].clone()
        self.proprio_history[:, -1] = proprio
        terminal = self._observations(proprio)
        ids = torch.nonzero(done).flatten()
        info = {"terminal_observation": terminal, "terminal_proprio": proprio.clone(),
                "reset_mask": done.clone(), "time_outs": truncated.clone(), "success": success.clone(),
                "reward_terms": parts, "terrain_level": self.levels.clone(),
                "episode": {"indices": ids.clone(), "return": self.episode_returns[ids].clone(),
                            "length": self.episode_steps[ids].clone()}}
        self._reset_indices(ids, curriculum=True, success=success)
        obs = self._observations(self.proprio_history[:, -1]) if ids.numel() else terminal
        if self.viewer is not None:
            if self.gym.query_viewer_has_closed(self.viewer):
                raise KeyboardInterrupt("Viewer closed")
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
        return obs, reward, terminated, truncated, info

    def close(self):
        if getattr(self, "viewer", None) is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None
        if getattr(self, "sim", None) is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
