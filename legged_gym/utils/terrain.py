from copy import copy
from dataclasses import dataclass

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


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
