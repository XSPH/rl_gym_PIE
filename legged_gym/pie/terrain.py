"""PIE terrain types in the original legged_gym Terrain grid.

The map/origin construction is inherited from Terrain. PIE cliffs use constant
height cells and explicit vertical faces, so collision, camera and ground-truth
sampling agree even at a gap edge. No lane boundary or goal is used.
"""
from copy import copy
from dataclasses import dataclass

import numpy as np
from isaacgym import terrain_utils

from legged_gym.utils.terrain import Terrain


@dataclass
class TerrainAtlas:
    vertices: np.ndarray
    triangles: np.ndarray
    heights: np.ndarray
    origins: np.ndarray
    kinds: list
    config: object
    border_size: float
    resolution: float
    vertical_scale: float


def _quad(vertices, triangles, points, reverse=False):
    base = len(vertices)
    vertices.extend(points)
    faces = [(0, 1, 2), (0, 2, 3)]
    if reverse:
        faces = [tuple(reversed(face)) for face in faces]
    triangles.extend([tuple(base + i for i in face) for face in faces])


class PIETerrain(Terrain):
    """Original 8 m grid and central origins with five PIE terrain families."""
    def __init__(self, cfg, num_robots, seed=1):
        self.rng = np.random.default_rng(seed)
        # Reuse native generation and add_terrain_to_map without first allocating
        # its dense sloped mesh; our vertical-face mesh is built below.
        generation_cfg = copy(cfg)
        generation_cfg.mesh_type = "heightfield"
        super().__init__(generation_cfg, num_robots)
        self.cfg = cfg
        self.type = cfg.mesh_type
        if self.type != "trimesh":
            raise ValueError("PIE native training requires terrain.mesh_type='trimesh'.")
        self.kinds = [self._kind_for_choice(j / cfg.num_cols + 0.001)
                      for j in range(cfg.num_cols)]
        self.vertices, self.triangles = self._vertical_mesh()
        self.atlas = TerrainAtlas(self.vertices, self.triangles,
                                  self.height_field_raw, self.env_origins,
                                  self.kinds, cfg, cfg.border_size,
                                  cfg.horizontal_scale, cfg.vertical_scale)

    def _kind_for_choice(self, choice):
        if len(self.cfg.kinds) == 1:
            return self.cfg.kinds[0]
        weights = np.asarray(self.cfg.terrain_proportions, dtype=float)
        if len(weights) != len(self.cfg.kinds) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("terrain_proportions must match PIE kinds and have positive total weight.")
        index = np.searchsorted(np.cumsum(weights / weights.sum()), choice, side='right')
        return self.cfg.kinds[min(index, len(self.cfg.kinds) - 1)]

    def curiculum(self):
        for column in range(self.cfg.num_cols):
            choice = column / self.cfg.num_cols + 0.001
            for row in range(self.cfg.num_rows):
                # Preserve the existing PIE nonzero easiest obstacle level.
                difficulty = (row + 1) / self.cfg.num_rows
                self.add_terrain_to_map(self.make_terrain(choice, difficulty), row, column)

    def make_terrain(self, choice, difficulty):
        cfg = self.cfg
        terrain = terrain_utils.SubTerrain("pie", width=self.width_per_env_pixels,
                                           length=self.length_per_env_pixels,
                                           vertical_scale=cfg.vertical_scale,
                                           horizontal_scale=cfg.horizontal_scale)
        kind = self._kind_for_choice(choice)
        # Square pyramid bands match native Terrain's center-platform layout.
        # A yaw command cannot bypass all obstacles through an empty y strip.
        center_x, center_y = self.env_length / 2, self.env_width / 2
        xs = (np.arange(self.length_per_env_pixels) + 0.5) * cfg.horizontal_scale
        ys = (np.arange(self.width_per_env_pixels) + 0.5) * cfg.horizontal_scale
        radius = np.maximum(np.abs(xs[:, None] - center_x), np.abs(ys[None, :] - center_y))
        surface = np.zeros_like(radius, dtype=np.float32)
        def fill(a, b, z):
            surface[(radius >= a) & (radius < b)] = z
        if kind == "gap":
            width = max(cfg.horizontal_scale, self.cfg.max_gap * difficulty
                        * self.rng.uniform(0.85, 1.0))
            fill(2.4 - width / 2, 2.4 + width / 2, -self.rng.uniform(0.4, 1.5))
        elif kind == "step":
            h = self.cfg.max_step * difficulty * self.rng.uniform(0.85, 1.0)
            fill(1.8, min(center_x, center_y) - 0.2, h)
        elif kind == "hurdle":
            h = self.cfg.max_hurdle * difficulty * self.rng.uniform(0.85, 1.0)
            half = self.rng.uniform(0.10, 0.25)
            fill(2.4 - half, 2.4 + half, h)
        elif kind == "stairs":
            h = self.cfg.max_stair * difficulty * self.rng.uniform(0.85, 1.0)
            for stair in range(5):
                fill(1.8 + stair * 0.35, 1.8 + (stair + 1) * 0.35, (stair + 1) * h)
        # +/-1m native root randomization plus feet fit on this 3.2m platform.
        surface[radius < 1.6] = 0
        terrain.height_field_raw[:] = np.rint(surface / cfg.vertical_scale).astype(np.int16)
        return terrain

    def _vertical_mesh(self):
        """Merged top faces, exposed cliffs, and outer closure only.

        Equal-height cell boundaries have no buried contact triangles. A cliff
        wall spans exactly its neighboring two top elevations, with its normal
        pointing from the higher solid terrain toward the lower surface.
        """
        heights = self.height_field_raw
        changes = np.r_[0, np.flatnonzero(np.any(heights[1:] != heights[:-1], axis=1)) + 1,
                        heights.shape[0]]
        vertices, triangles = [], []
        scale, offset, vertical = self.cfg.horizontal_scale, self.cfg.border_size, self.cfg.vertical_scale
        for first, last in zip(changes[:-1], changes[1:]):
            row = heights[first]
            runs = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
            x0, x1 = first * scale - offset, last * scale - offset
            for start, end in zip(runs[:-1], runs[1:]):
                y0, y1 = start * scale - offset, end * scale - offset
                z = float(row[start]) * vertical
                _quad(vertices, triangles, [(x0, y0, z), (x1, y0, z),
                                            (x1, y1, z), (x0, y1, z)])
        bottom_units = int(heights.min()) - int(np.ceil(0.2 / vertical))
        def walls(before, after, boundary, axis):
            if np.array_equal(before, after):
                return
            differences = (before[1:] != before[:-1]) | (after[1:] != after[:-1])
            runs = np.r_[0, np.flatnonzero(differences) + 1, len(before)]
            coordinate = boundary * scale - offset
            for start, end in zip(runs[:-1], runs[1:]):
                a, b = int(before[start]), int(after[start])
                if a == b:
                    continue
                low, high = min(a, b) * vertical, max(a, b) * vertical
                first, last = start * scale - offset, end * scale - offset
                if axis == 0:
                    points = [(coordinate, first, low), (coordinate, last, low),
                              (coordinate, last, high), (coordinate, first, high)]
                else:
                    points = [(first, coordinate, low), (first, coordinate, high),
                              (last, coordinate, high), (last, coordinate, low)]
                _quad(vertices, triangles, points, reverse=(a < b))
        outside_x = np.full(heights.shape[1], bottom_units, dtype=np.int32)
        for boundary in range(heights.shape[0] + 1):
            before = outside_x if boundary == 0 else heights[boundary - 1]
            after = outside_x if boundary == heights.shape[0] else heights[boundary]
            walls(before, after, boundary, axis=0)
        outside_y = np.full(heights.shape[0], bottom_units, dtype=np.int32)
        for boundary in range(heights.shape[1] + 1):
            before = outside_y if boundary == 0 else heights[:, boundary - 1]
            after = outside_y if boundary == heights.shape[1] else heights[:, boundary]
            walls(before, after, boundary, axis=1)
        x0, y0 = -offset, -offset
        x1 = heights.shape[0] * scale - offset
        y1 = heights.shape[1] * scale - offset
        bottom = bottom_units * vertical
        _quad(vertices, triangles, [(x0, y0, bottom), (x1, y0, bottom),
                                    (x1, y1, bottom), (x0, y1, bottom)], reverse=True)
        return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.uint32)


class TerrainSampler:
    def __init__(self, atlas, device):
        import torch
        self.torch = torch
        self.atlas = atlas
        self.heights = torch.as_tensor(atlas.heights, device=device)
        self.origins = torch.as_tensor(atlas.origins, device=device, dtype=torch.float32)

    def sample(self, points, levels=None, columns=None):
        """World points, including neighboring tiles; half-open cells at edges."""
        torch, atlas = self.torch, self.atlas
        ix = torch.floor((points[..., 0] + atlas.border_size) / atlas.resolution).long()
        iy = torch.floor((points[..., 1] + atlas.border_size) / atlas.resolution).long()
        inside = ((ix >= 0) & (ix < self.heights.shape[0])
                  & (iy >= 0) & (iy < self.heights.shape[1]))
        height = self.heights[ix.clamp(0, self.heights.shape[0] - 1),
                              iy.clamp(0, self.heights.shape[1] - 1)].float() * atlas.vertical_scale
        # Match native height-label boundary clamping: its 25m border is flat.
        return torch.where(inside, height, torch.zeros_like(height))
