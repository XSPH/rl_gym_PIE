"""Original course generator with matching triangle geometry and height labels.

Each lane is a sequence of rectangular plateaus. Explicit vertical faces avoid
the sloped cliff artefact of triangulated height fields. No upstream code copied.
"""
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TerrainAtlas:
    vertices: np.ndarray
    triangles: np.ndarray
    heights: np.ndarray
    origins: np.ndarray
    goals: np.ndarray
    kinds: List[str]
    config: object


def _box(vertices, triangles, x0, x1, y0, y1, z0, z1):
    base = len(vertices)
    vertices.extend([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                     (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])
    # Consistent outward winding, including the upper surface.
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    triangles.extend([tuple(base + i for i in face) for face in faces])


def build_atlas(cfg, seed=1):
    rng = np.random.default_rng(seed)
    nx = int(round(cfg.length / cfg.resolution))
    if abs(nx * cfg.resolution - cfg.length) > 1e-6:
        raise ValueError("Terrain length must be an integer multiple of resolution.")
    columns = len(cfg.kinds) * cfg.variants
    heights = np.zeros((cfg.levels, columns, nx), dtype=np.float32)
    origins = np.zeros((cfg.levels, columns, 3), dtype=np.float32)
    goals = np.zeros_like(origins)
    vertices, triangles, kind_names = [], [], []
    for col in range(columns):
        kind = cfg.kinds[col % len(cfg.kinds)]
        kind_names.append(kind)
        for level in range(cfg.levels):
            d = (level + 1) / cfg.levels
            profile = heights[level, col]

            def fill(a, b, z):
                start = max(0, int(round(a / cfg.resolution)))
                end = min(nx, int(round(b / cfg.resolution)))
                profile[start:end] = z

            if kind == "gap":
                width = max(cfg.resolution, cfg.max_gap * d * rng.uniform(0.85, 1.0))
                for center in (2.5, 5.3):
                    fill(center - width / 2, center + width / 2, -rng.uniform(0.4, 1.5))
            elif kind == "step":
                h = cfg.max_step * d * rng.uniform(0.85, 1.0)
                fill(2.0, 3.0 + rng.uniform(0.2, 0.8), h)
                fill(4.8, 6.0, h * rng.uniform(0.7, 1.0))
            elif kind == "hurdle":
                h = cfg.max_hurdle * d * rng.uniform(0.85, 1.0)
                for center in (2.5, 5.3):
                    half = rng.uniform(0.10, 0.25)
                    fill(center - half, center + half, h)
            elif kind == "stairs":
                h = cfg.max_stair * d * rng.uniform(0.85, 1.0)
                for stair in range(5):
                    fill(1.8 + stair * 0.35, 1.8 + (stair + 1) * 0.35, (stair + 1) * h)
                    fill(3.55 + stair * 0.35, 3.55 + (stair + 1) * 0.35, (4 - stair) * h)
            # Start and finish remain flat at every difficulty level.
            fill(0.0, 1.5, 0.0)
            fill(cfg.length - 1.2, cfg.length, 0.0)
            x0 = level * (cfg.length + cfg.spacing)
            y0 = col * (cfg.width + cfg.spacing)
            origins[level, col] = (x0 + 0.7, y0 + cfg.width / 2, 0.0)
            goals[level, col] = (x0 + cfg.length - 0.7, y0 + cfg.width / 2, 0.0)
            changes = np.r_[0, np.flatnonzero(np.diff(profile)) + 1, nx]
            for start, end in zip(changes[:-1], changes[1:]):
                _box(vertices, triangles, x0 + start * cfg.resolution,
                     x0 + end * cfg.resolution, y0, y0 + cfg.width,
                     cfg.floor_height - 0.1, float(profile[start]))
    # Visible floor matches the global PhysX plane at floor_height.
    xmax = cfg.levels * (cfg.length + cfg.spacing)
    ymax = columns * (cfg.width + cfg.spacing)
    _box(vertices, triangles, -cfg.spacing, xmax, -cfg.spacing, ymax,
         cfg.floor_height - 0.2, cfg.floor_height)
    return TerrainAtlas(np.asarray(vertices, dtype=np.float32),
                        np.asarray(triangles, dtype=np.uint32), heights, origins, goals,
                        kind_names, cfg)


class TerrainSampler:
    def __init__(self, atlas, device):
        import torch
        self.torch = torch
        self.cfg = atlas.config
        self.heights = torch.as_tensor(atlas.heights, device=device)
        self.origins = torch.as_tensor(atlas.origins, device=device)
        self.goals = torch.as_tensor(atlas.goals, device=device)

    def sample(self, points, levels, columns):
        """points[N,P,3], levels/columns[N]; outside lanes equals floor."""
        torch, cfg = self.torch, self.cfg
        local_x = points[..., 0] - levels[:, None] * (cfg.length + cfg.spacing)
        local_y = points[..., 1] - columns[:, None] * (cfg.width + cfg.spacing)
        index = torch.floor(local_x / cfg.resolution).long()
        inside = (local_x >= 0) & (local_x < cfg.length) & (local_y >= 0) & (local_y <= cfg.width)
        value = self.heights[levels[:, None], columns[:, None], index.clamp(0, self.heights.shape[-1] - 1)]
        return torch.where(inside, value, torch.full_like(value, cfg.floor_height))
