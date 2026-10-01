"""GPU optical-axis depth from the same static terrain mesh used by PhysX.

Camera convention is x forward, y left, z up; pixels are u right, v down.
This is an original implementation based on Warp's documented mesh query API.
The robot itself and other dynamic bodies are not part of the camera mesh.
"""
import math

import torch
import warp as wp


@wp.kernel
def _raycast(mesh: wp.uint64,
             env_ids: wp.array(dtype=wp.int32),
             positions: wp.array(dtype=wp.vec3),
             orientations: wp.array(dtype=wp.quat),
             focal: wp.array(dtype=wp.float32),
             width: int, height: int, near: float, far: float,
             depth: wp.array(dtype=wp.float32, ndim=3)):
    slot, row, column = wp.tid()
    env = env_ids[slot]
    local = wp.vec3(1.0, -(float(column) + 0.5 - float(width) * 0.5) / focal[env],
                    -(float(row) + 0.5 - float(height) * 0.5) / focal[env])
    ray = wp.normalize(local)
    direction = wp.quat_rotate(orientations[env], ray)
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3(0.0, 0.0, 0.0)
    face = int(0)
    value = far
    # far is optical-axis depth; oblique rays need a larger range limit.
    if wp.mesh_query_ray(mesh, positions[env], direction, far / ray[0],
                         t, u, v, sign, normal, face):
        value = wp.clamp(t * ray[0], near, far)
    depth[env, row, column] = value


class WarpDepthCamera:
    def __init__(self, atlas, count, config, device):
        wp.init()
        self.config, self.device, self.count = config, device, count
        self.positions = torch.zeros((count, 3), dtype=torch.float32, device=device)
        self.orientations = torch.zeros((count, 4), dtype=torch.float32, device=device)
        self.orientations[:, 3] = 1.0
        self.focal = torch.full((count,), config.width / (2 * math.tan(math.radians(config.hfov_degrees) / 2)), device=device)
        self.depth = torch.full((count, config.height, config.width), config.far, device=device)
        # Keep all Warp arrays alive as long as the mesh references them.
        self.vertices = wp.array(atlas.vertices, dtype=wp.vec3, device=device)
        self.indices = wp.array(atlas.triangles.astype("int32").reshape(-1), dtype=wp.int32, device=device)
        self.mesh = wp.Mesh(points=self.vertices, indices=self.indices)
        self.all_ids = torch.arange(count, dtype=torch.int32, device=device)
        self._positions = wp.from_torch(self.positions, dtype=wp.vec3)
        self._orientations = wp.from_torch(self.orientations, dtype=wp.quat)
        self._focal = wp.from_torch(self.focal)
        self._depth = wp.from_torch(self.depth)

    def render(self, env_ids=None):
        ids = self.all_ids if env_ids is None else env_ids.to(
            device=self.device, dtype=torch.int32).contiguous()
        if ids.numel() == 0:
            return self.depth
        # Sharing the current PyTorch stream establishes ordering for pose writes
        # and image reads, without a device-wide synchronize on every frame.
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        wp.launch(_raycast, dim=(ids.numel(), self.config.height, self.config.width),
                  inputs=[self.mesh.id, wp.from_torch(ids), self._positions,
                          self._orientations, self._focal, self.config.width,
                          self.config.height, self.config.near, self.config.far, self._depth],
                  device=self.device, stream=stream)
        return self.depth

    def encode(self, image):
        cfg = self.config
        image = image.clamp(cfg.near, cfg.far)
        if cfg.noise_std:
            image = (image + torch.randn_like(image) * cfg.noise_std).clamp(cfg.near, cfg.far)
        if cfg.salt_pepper_probability:
            probability = torch.rand_like(image)
            image = torch.where(probability < cfg.salt_pepper_probability / 2, cfg.near, image)
            image = torch.where(probability > 1 - cfg.salt_pepper_probability / 2, cfg.far, image)
        return (image - cfg.near) / (cfg.far - cfg.near) - 0.5 if cfg.normalize else image
