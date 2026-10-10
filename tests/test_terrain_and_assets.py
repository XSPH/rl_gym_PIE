import xml.etree.ElementTree as ET

import numpy as np
import torch

from pathlib import Path
from native_cpu_helpers import load_native_classes, original_visual_config
from test_native_environment import state


def test_native_grid_plateau_collision_vertices_and_labels_agree():
    classes = load_native_classes()
    cfg = original_visual_config(classes)
    terrain = classes.terrain(cfg.terrain, 4096, seed=4)
    atlas = terrain.atlas
    assert atlas.triangles.max() < len(atlas.vertices)
    sampler = classes.sampler(atlas, "cpu")
    assert terrain.env_origins.shape == (10, cfg.terrain.num_cols, 3)
    assert len(set(atlas.kinds)) == 5
    for level in (0, cfg.terrain.num_rows - 1):
        for column, kind in enumerate(atlas.kinds):
            count = terrain.length_per_env_pixels
            x = (torch.arange(count) + .5) * cfg.terrain.horizontal_scale
            points = torch.zeros(1, count, 3)
            points[0, :, 0] = x + level * cfg.terrain.terrain_length
            points[0, :, 1] = column * cfg.terrain.terrain_width + cfg.terrain.terrain_width / 2
            actual = sampler.sample(points)
            first = terrain.border + level * count
            col = terrain.border + column * terrain.width_per_env_pixels + terrain.width_per_env_pixels // 2
            expected = np.minimum(np.minimum(atlas.heights[first:first+count, col],
                                             atlas.heights[first+1:first+count+1, col]),
                                  atlas.heights[first:first+count, col+1]).astype(np.float32) * np.float32(cfg.terrain.vertical_scale)
            np.testing.assert_allclose(actual.numpy()[0], expected, rtol=0, atol=1e-7)
            origin = torch.tensor(terrain.env_origins[level, column], dtype=torch.float32).view(1, 1, 3)
            # WMP climb explicitly keeps origin Z=0 despite ground noise.
            allowance = cfg.terrain.roughness if kind == 'step' else 0.0
            assert sampler.sample(origin).item() <= origin[0, 0, 2].item() + allowance + 1e-6
            mesh_z = np.unique(atlas.vertices[:, 2])
            assert all(np.isclose(value, mesh_z, rtol=0, atol=1e-6).any()
                       for value in np.unique(expected))
    outside = torch.tensor([[[-1000., -1000., 0.]]])
    assert sampler.sample(outside).item() == 0


def test_mesh_uses_native_heightfield_vertices_and_triangle_construction():
    classes = load_native_classes()
    cfg = original_visual_config(classes)
    terrain = classes.terrain(cfg.terrain, 4096, seed=4)
    vertices, triangles, heights = terrain.vertices, terrain.triangles, terrain.height_field_raw
    nx, ny = heights.shape
    assert vertices.shape == (nx * ny, 3)
    assert triangles.shape == (2 * (nx-1) * (ny-1), 3)
    np.testing.assert_allclose(vertices[:, 2], heights.flatten() * cfg.terrain.vertical_scale,
                               rtol=0, atol=1e-6)
    np.testing.assert_array_equal(triangles[:2], [[0, ny+1, 1], [0, ny, ny+1]])
    # Native slope-threshold correction shifts XY vertices by whole cells.
    base_x, base_y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    shift_x = (vertices[:, 0] + cfg.terrain.border_size) / cfg.terrain.horizontal_scale - base_x.flatten()
    shift_y = (vertices[:, 1] + cfg.terrain.border_size) / cfg.terrain.horizontal_scale - base_y.flatten()
    np.testing.assert_allclose(shift_x, np.rint(shift_x), atol=1e-3, rtol=0)
    np.testing.assert_allclose(shift_y, np.rint(shift_y), atol=1e-3, rtol=0)
    assert np.abs(shift_x).max() > 0.9


def test_progress_curriculum_updates_origins_and_skips_initial_reset():
    classes = load_native_classes()
    task = state(classes.task, count=4)
    task.cfg = classes.config()
    task.init_done = True
    task.terrain = type("Terrain", (), {"env_length": 8.})()
    task.terrain_levels = torch.tensor([2, 2, 0, 9])
    task.terrain_types = torch.zeros(4, dtype=torch.long)
    task.terrain_origins = torch.zeros(10, 1, 3)
    task.terrain_origins[:, 0, 0] = torch.arange(10) * 8. + 4.
    task.env_origins = task.terrain_origins[task.terrain_levels, task.terrain_types].clone()
    task.root_states[:, :3] = task.env_origins
    task.root_states[:, 0] += torch.tensor([4.5, 1., 0., 4.5])
    task.commands[:, 0] = .5
    task.episode_length_buf[:] = torch.tensor([10, 10, 0, 10])
    task._episode_steps = task.episode_length_buf.clone()
    task._update_terrain_curriculum(torch.arange(4))
    assert task.terrain_levels[:3].tolist() == [3, 1, 0]
    assert 0 <= task.terrain_levels[3] < 10
    torch.testing.assert_close(task.env_origins,
                               task.terrain_origins[task.terrain_levels, task.terrain_types])


def test_default_parkour_map_has_easy_and_hard_obstacles_with_wmp_dimensions():
    classes = load_native_classes()
    cfg = classes.config()
    terrain = classes.terrain(cfg.terrain, 4096, seed=4)
    size = terrain.length_per_env_pixels
    for column, kind in enumerate(terrain.atlas.kinds):
        patches = []
        for row in (0, cfg.terrain.num_rows - 1):
            x = terrain.border + row * size
            y = terrain.border + column * terrain.width_per_env_pixels
            patches.append(terrain.height_field_raw[x:x+size, y:y+size]
                           * cfg.terrain.vertical_scale)
        easy, hard = patches
        if kind == 'flat':
            assert np.abs(easy).max() <= cfg.terrain.roughness
            assert np.abs(hard).max() <= cfg.terrain.roughness
        elif kind == 'gap':
            floor = cfg.terrain.gap_floor_units * cfg.terrain.vertical_scale
            # Count gaps along the channel, excluding the surrounding pit.
            assert (easy[:, size//2] < floor/2).sum() < (hard[:, size//2] < floor/2).sum()
            assert abs(hard.min() - floor) <= cfg.terrain.roughness + 1e-6
            np.testing.assert_allclose(terrain.env_origins[:, column, 0], np.arange(10) * 8 + 4)
        elif kind == 'slope':
            assert np.abs(easy).max() <= cfg.terrain.roughness
            assert np.abs(hard).max() > np.abs(easy).max()
        elif kind == 'stairs':
            # Native pyramid stairs derive the number of risers from tile size.
            assert len(np.unique(np.abs(easy))) > 5
            assert np.abs(hard).max() > np.abs(easy).max()
            assert np.diff(np.unique(np.abs(hard))).max() <= .215 + 1e-6
        elif kind == 'step':
            assert abs(easy.max() - cfg.terrain.min_step) <= cfg.terrain.roughness + 1e-6
            assert abs(hard.max() - cfg.terrain.max_step) <= cfg.terrain.roughness + 1e-6
            assert hard[-1, size//2] >= cfg.terrain.max_step - cfg.terrain.roughness
            np.testing.assert_allclose(terrain.env_origins[:, column, 0], np.arange(10) * 8 + 1)
        else:
            np.testing.assert_allclose(easy.max(), cfg.terrain.min_hurdle, atol=1e-6)
            np.testing.assert_allclose(hard.max(), cfg.terrain.max_hurdle, atol=1e-6)


def test_native_root_reset_adds_base_height_exactly_once():
    classes = load_native_classes()
    task = state(classes.base)
    task.base_init_state = torch.tensor([0., 0., .3, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
    task.custom_origins = True
    task.env_origins = torch.tensor([[4., 4., 0.], [12., 4., .75], [20., 4., 0.]])
    recorded = []
    task.gym = type("Gym", (), {})()
    task.gym.set_actor_root_state_tensor_indexed = lambda sim, root, ids, count: recorded.append(ids.clone())
    task._reset_root_states(torch.tensor([0, 1]))
    torch.testing.assert_close(task.root_states[:2, 2], torch.tensor([.3, 1.05]))
    assert recorded[0].dtype == torch.int32 and recorded[0].tolist() == [0, 1]


def test_base_height_reward_excludes_neighboring_pits_for_all_terrain_groups():
    classes = load_native_classes()
    task = state(classes.task)
    task.cfg = classes.config()
    task.env_origins = torch.zeros(3, 3)
    task.root_states[:, 2] = .3
    # Include a robot originally from a flat column and a scan fully over a pit.
    task._scan_ground_heights = lambda: torch.tensor([[0., -5., -4.95], [0., 0., 0.], [-5., -5., -5.]])
    torch.testing.assert_close(task._reward_base_height(), torch.zeros(3), atol=1e-7, rtol=0)


def test_asset_complete_joint_order_and_stand_pose():
    classes = load_native_classes()
    cfg = classes.config()
    path = Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(Path(__file__).resolve().parents[1])))
    root = ET.parse(str(path)).getroot()
    movable = [j.attrib["name"] for j in root.findall("joint") if j.attrib["type"] != "fixed"]
    assert movable == cfg.asset.joint_names
    for mesh in root.findall(".//mesh"):
        assert (path.parent / mesh.attrib["filename"]).resolve().is_file()
    fk = classes.kinematics.UrdfKinematics(path, cfg.asset.joint_names, cfg.asset.base_name, "cpu")
    states = fk.forward(torch.tensor([[0.0, 0.0, cfg.init_state.pos[2]]]),
                        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                        torch.tensor([[cfg.init_state.default_joint_angles[name] for name in cfg.asset.joint_names]]))
    feet = torch.cat([states[name][0] for name in cfg.asset.foot_names])
    assert torch.all(feet[:, 2] > 0.0)
    assert torch.all(feet[:, 2] < 0.1)
    assert torch.all(feet[:2, 0] > feet[2:, 0])
