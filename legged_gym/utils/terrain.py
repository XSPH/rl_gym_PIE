# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

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


class PIETerrain(Terrain):
    """WMP Go1 heightfields restricted to PIE's six terrain families.

    Layout reference: haozhang04/wmp_go1, commit 65cf1af,
    legged_gym/utils/terrain.py. SDK generators and mesh conversion are shared;
    random noise uses a local RNG and supported bilinear SciPy interpolation.
    """
    def __init__(self, cfg, num_robots, seed=1):
        self._validate_geometry(cfg)
        self.rng = np.random.default_rng(seed)
        generation_cfg = copy(cfg)
        generation_cfg.mesh_type = "heightfield"
        # Freezing runtime progression does not randomize the graded map.
        generation_cfg.curriculum = True
        super().__init__(generation_cfg, num_robots)
        self.cfg = cfg
        self.type = cfg.mesh_type
        if self.type != "trimesh":
            raise ValueError("PIE native training requires terrain.mesh_type='trimesh'.")
        self.kinds = [self._kind_for_choice(j / cfg.num_cols + 0.001)
                      for j in range(cfg.num_cols)]
        # WMP's converter uses the SDK's vertex shift / triangle construction.
        # Both PhysX and Warp receive this one mesh in world coordinates.
        self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
            self.height_field_raw, cfg.horizontal_scale, cfg.vertical_scale,
            cfg.slope_threshold)
        self.vertices[:, :2] -= cfg.border_size
        self.atlas = TerrainAtlas(self.vertices, self.triangles,
                                  self.height_field_raw, self.env_origins,
                                  self.kinds, cfg, cfg.border_size,
                                  cfg.horizontal_scale, cfg.vertical_scale)

    @staticmethod
    def _validate_geometry(cfg):
        if cfg.geometry_version != 3:
            raise ValueError('PIE requires current terrain geometry version 3')
        if (not cfg.kinds or len(set(cfg.kinds)) != len(cfg.kinds)
                or set(cfg.kinds) - {'flat', 'slope', 'stairs', 'gap', 'step', 'hurdle'}):
            raise ValueError('Unsupported or duplicate PIE terrain kinds')
        weights = np.asarray(cfg.terrain_proportions, dtype=float)
        if (len(weights) != len(cfg.kinds) or not np.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError('Terrain proportions must match kinds and have positive total weight')
        values = [cfg.terrain_length, cfg.terrain_width, cfg.horizontal_scale,
                  cfg.vertical_scale, cfg.platform_size, cfg.spawn_xy_range,
                  cfg.obstacle_spawn_range, cfg.forward_spawn_x, cfg.obstacle_start_x,
                  cfg.roughness, cfg.roughness_step, cfg.roughness_downsample,
                  cfg.slope_threshold, cfg.max_slope, cfg.gap_floor_units, cfg.border_size]
        if not np.isfinite(values).all():
            raise ValueError('Terrain geometry parameters must be finite')
        if (cfg.num_rows < 1 or cfg.num_cols < 1 or cfg.horizontal_scale <= 0
                or cfg.vertical_scale <= 0 or cfg.border_size < 0
                or min(cfg.terrain_length, cfg.terrain_width) < 7):
            raise ValueError('Invalid PIE terrain grid')
        if (cfg.platform_size <= 2 * (cfg.spawn_xy_range + 0.4)
                or cfg.platform_size >= min(cfg.terrain_length, cfg.terrain_width)
                or min(cfg.spawn_xy_range, cfg.obstacle_spawn_range, cfg.roughness, cfg.max_slope) < 0
                or cfg.slope_threshold <= 0 or cfg.roughness_step < cfg.vertical_scale
                or cfg.roughness_downsample < cfg.horizontal_scale
                or cfg.roughness_downsample > min(cfg.terrain_length, cfg.terrain_width) / 2
                or cfg.gap_floor_units != int(cfg.gap_floor_units)
                or not -32768 <= cfg.gap_floor_units < 0):
            raise ValueError('Invalid WMP terrain platform, roughness, or pit parameters')
        for name in ('gap', 'step', 'hurdle', 'stair'):
            low, high = getattr(cfg, 'min_' + name), getattr(cfg, 'max_' + name)
            if not np.isfinite([low, high]).all() or not 0 < low <= high:
                raise ValueError('Invalid terrain size range: ' + name)
        for name in ('gap_channel_width', 'step_length', 'hurdle_length', 'stair_width'):
            bounds = getattr(cfg, name)
            if len(bounds) != 2 or not np.isfinite(bounds).all() or not 0 < bounds[0] <= bounds[1]:
                raise ValueError('Invalid terrain geometry range: ' + name)
        if (cfg.gap_channel_width[1] >= cfg.terrain_width
                or cfg.max_gap >= cfg.terrain_length / 2 - 2
                or cfg.stair_width[0] < cfg.horizontal_scale
                or cfg.forward_spawn_x - cfg.obstacle_spawn_range < 0.4
                or cfg.forward_spawn_x + cfg.obstacle_spawn_range + 0.8 >= cfg.obstacle_start_x
                or cfg.obstacle_start_x + cfg.step_length[1] + 0.8
                   >= cfg.terrain_length - cfg.step_length[1]
                or cfg.obstacle_start_x + cfg.hurdle_length[1] + 0.8 >= cfg.terrain_length):
            raise ValueError('WMP courses must contain spawn and separated obstacle zones')

    def _kind_for_choice(self, choice):
        weights = np.asarray(self.cfg.terrain_proportions, dtype=float)
        index = np.searchsorted(np.cumsum(weights / weights.sum()), choice, side='right')
        return self.cfg.kinds[min(index, len(self.cfg.kinds) - 1)]

    def curiculum(self):
        for column in range(self.cfg.num_cols):
            choice = column / self.cfg.num_cols + 0.001
            for row in range(self.cfg.num_rows):
                # WMP's ordinary slope/stair schedule deliberately tops at .9.
                difficulty = row / self.cfg.num_rows
                self.add_terrain_to_map(self.make_terrain(choice, difficulty), row, column)

    def add_terrain_to_map(self, terrain, row, col):
        nx, ny = self.length_per_env_pixels, self.width_per_env_pixels
        x, y = self.border + row * nx, self.border + col * ny
        self.height_field_raw[x:x+nx, y:y+ny] = terrain.height_field_raw
        origin_x = getattr(terrain, 'spawn_x', self.env_length / 2)
        origin_z = getattr(terrain, 'spawn_z', None)
        if origin_z is None:
            # Same central 2x2 m maximum as WMP (including roughness).
            scale = self.cfg.horizontal_scale
            x1, x2 = int((self.env_length / 2 - 1) / scale), int((self.env_length / 2 + 1) / scale)
            y1, y2 = int((self.env_width / 2 - 1) / scale), int((self.env_width / 2 + 1) / scale)
            origin_z = float(terrain.height_field_raw[x1:x2, y1:y2].max()) * self.cfg.vertical_scale
        self.env_origins[row, col] = [row * self.env_length + origin_x,
                                      col * self.env_width + self.env_width / 2, origin_z]

    def _direction(self, choice, kind):
        weights = np.asarray(self.cfg.terrain_proportions, dtype=float)
        weights /= weights.sum()
        index = self.cfg.kinds.index(kind)
        fraction = (choice - weights[:index].sum()) / weights[index]
        return -1.0 if fraction < 0.5 else 1.0

    def _roughen(self, terrain):
        """WMP's uniform +/-5 cm noise on a .2 m grid, bilinearly upsampled.

        RectBivariateSpline(kx=ky=1) keeps the SDK's linear interpolation
        semantics without its removed scipy.interpolate.interp2d dependency.
        """
        cfg = self.cfg
        amplitude = int(cfg.roughness / cfg.vertical_scale)
        step = int(cfg.roughness_step / cfg.vertical_scale)
        values = np.arange(-amplitude, amplitude + step, step)
        nx, ny = terrain.height_field_raw.shape
        coarse = self.rng.choice(values, (int(nx * cfg.horizontal_scale / cfg.roughness_downsample),
                                          int(ny * cfg.horizontal_scale / cfg.roughness_downsample)))
        x = np.linspace(0, nx * cfg.horizontal_scale, coarse.shape[0])
        y = np.linspace(0, ny * cfg.horizontal_scale, coarse.shape[1])
        noise = interpolate.RectBivariateSpline(x, y, coarse, kx=1, ky=1)(
            np.linspace(0, nx * cfg.horizontal_scale, nx),
            np.linspace(0, ny * cfg.horizontal_scale, ny))
        terrain.height_field_raw += np.rint(noise).astype(np.int16)

    def make_terrain(self, choice, difficulty):
        cfg = self.cfg
        terrain = terrain_utils.SubTerrain("pie", width=self.length_per_env_pixels,
                                           length=self.width_per_env_pixels,
                                           vertical_scale=cfg.vertical_scale,
                                           horizontal_scale=cfg.horizontal_scale)
        kind = self._kind_for_choice(choice)
        normalized = min(difficulty * cfg.num_rows / max(cfg.num_rows - 1, 1), 1.0)
        def size(name, progress=normalized):
            low, high = getattr(cfg, 'min_' + name), getattr(cfg, 'max_' + name)
            return low + progress * (high - low)
        scale = cfg.horizontal_scale
        nx, ny = terrain.height_field_raw.shape
        if kind == 'slope':
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=cfg.max_slope * difficulty * self._direction(choice, kind),
                platform_size=cfg.platform_size)
        elif kind == 'stairs':
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=self.rng.uniform(*cfg.stair_width),
                step_height=size('stair', difficulty) * self._direction(choice, kind),
                platform_size=cfg.platform_size)
        elif kind == 'gap':
            # WMP gap: central [3,6] m platform, gaps on both sides, narrow
            # [1,2] m channel, with the surrounding field at -1000 units.
            gap = int(size('gap') / scale)
            x1, x2 = int(nx // 2 - 1 / scale), int(nx // 2 + 2 / scale)
            half_width = self.rng.uniform(*cfg.gap_channel_width) / 2
            y1, y2 = int(ny // 2 - half_width / scale), int(ny // 2 + half_width / scale)
            terrain.height_field_raw[:, :] = cfg.gap_floor_units
            terrain.height_field_raw[gap:x1-gap, y1:y2] = 0
            terrain.height_field_raw[x1:x2, y1:y2] = 0
            terrain.height_field_raw[x2+gap:, y1:y2] = 0
        elif kind == 'step':
            # WMP climb has two raised blocks; the second reaches the tile end.
            lengths = [int(round(self.rng.uniform(*cfg.step_length) / scale)) for _ in range(2)]
            starts = [int(round(cfg.obstacle_start_x / scale)), nx - lengths[1]]
            height = int(round(size('step') / cfg.vertical_scale))
            for start, length in zip(starts, lengths):
                terrain.height_field_raw[start:start+length, :] = height
            terrain.spawn_x, terrain.spawn_z = cfg.forward_spawn_x, 0.0
        elif kind == 'hurdle':
            # PIE's narrow wall has no exact WMP counterpart.
            start = int(round(cfg.obstacle_start_x / scale))
            length = int(round(self.rng.uniform(*cfg.hurdle_length) / scale))
            terrain.height_field_raw[start:start+length, :] = int(round(size('hurdle') / cfg.vertical_scale))
            terrain.spawn_x, terrain.spawn_z = cfg.forward_spawn_x, 0.0
        if kind in ('flat', 'slope', 'gap', 'step'):
            self._roughen(terrain)
        return terrain


class TerrainSampler:
    def __init__(self, atlas, device):
        import torch
        self.torch = torch
        self.atlas = atlas
        self.heights = torch.as_tensor(atlas.heights, device=device)
        self.origins = torch.as_tensor(atlas.origins, device=device, dtype=torch.float32)

    def sample(self, points, levels=None, columns=None):
        """WMP/native conservative labels: minimum of three adjacent samples.

        These are heightfield labels, not exact ray intersections on the mesh's
        shifted cliff vertices. PhysX and the depth camera use the mesh itself.
        """
        torch, atlas = self.torch, self.atlas
        ix = torch.floor((points[..., 0] + atlas.border_size) / atlas.resolution).long()
        iy = torch.floor((points[..., 1] + atlas.border_size) / atlas.resolution).long()
        inside = ((ix >= 0) & (ix < self.heights.shape[0] - 1)
                  & (iy >= 0) & (iy < self.heights.shape[1] - 1))
        x, y = ix.clamp(0, self.heights.shape[0] - 2), iy.clamp(0, self.heights.shape[1] - 2)
        height = torch.minimum(torch.minimum(self.heights[x, y], self.heights[x+1, y]),
                               self.heights[x, y+1]).float() * atlas.vertical_scale
        return torch.where(inside, height, torch.zeros_like(height))
