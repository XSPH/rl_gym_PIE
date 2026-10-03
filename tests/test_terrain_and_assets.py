import xml.etree.ElementTree as ET

import numpy as np
import torch

from legged_gym.pie.config import EnvConfig
from legged_gym.pie.kinematics import UrdfKinematics
from native_cpu_helpers import load_native_classes
from test_native_environment import state


def test_native_grid_plateau_collision_vertices_and_labels_agree():
    classes = load_native_classes()
    cfg = classes.config()
    terrain = classes.terrain(cfg.terrain, 4096, cfg.pie.terrain, seed=4)
    atlas = terrain.atlas
    assert atlas.triangles.max() < len(atlas.vertices)
    sampler = classes.sampler(atlas, "cpu")
    assert terrain.env_origins.shape == (10, 20, 3)
    np.testing.assert_array_equal(terrain.env_origins[:, :, 2], 0)
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
            expected = atlas.heights[first:first+count, col] * cfg.terrain.vertical_scale
            np.testing.assert_allclose(actual.numpy()[0], expected, rtol=0, atol=1e-7)
            assert np.all(expected[24:56] == 0)  # 3.2m central spawn platform
            mesh_z = np.unique(atlas.vertices[:, 2])
            assert all(np.isclose(value, mesh_z, rtol=0, atol=1e-7).any()
                       for value in np.unique(expected))
    outside = torch.tensor([[[-1000., -1000., 0.]]])
    assert sampler.sample(outside).item() == 0


def test_mesh_faces_cover_surface_and_only_exposed_cliffs_with_outward_normals():
    classes = load_native_classes()
    cfg = classes.config()
    terrain = classes.terrain(cfg.terrain, 4096, cfg.pie.terrain, seed=4)
    vertices, triangles, heights = terrain.vertices, terrain.triangles, terrain.height_field_raw
    normals = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                       vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    assert (np.linalg.norm(normals, axis=-1) > 0).all()
    quads = vertices.reshape(-1, 4, 3)
    assert len(triangles) == 2 * len(quads)
    bottom_units = int(heights.min()) - int(np.ceil(.2 / cfg.terrain.vertical_scale))
    below = bottom_units * cfg.terrain.vertical_scale
    bottom_count, wall_count = 0, 0
    def cell(coordinate):
        return int(np.floor((coordinate + cfg.terrain.border_size) / cfg.terrain.horizontal_scale))
    for quad, normal in zip(quads, normals[::2]):
        midpoint = quad.mean(axis=0)
        if np.ptp(quad[:, 2]) == 0:
            if normal[2] < 0:
                bottom_count += 1
                np.testing.assert_allclose(quad[:, 2], below, atol=1e-7, rtol=0)
            else:
                assert normal[2] > 0
                expected = heights[cell(midpoint[0]), cell(midpoint[1])] * cfg.terrain.vertical_scale
                np.testing.assert_allclose(quad[:, 2], expected, atol=1e-7, rtol=0)
            continue
        wall_count += 1
        axis = 0 if np.ptp(quad[:, 0]) == 0 else 1
        assert np.ptp(quad[:, axis]) == 0
        boundary = int(round((midpoint[axis] + cfg.terrain.border_size) / cfg.terrain.horizontal_scale))
        side = cell(midpoint[1-axis])
        if axis == 0:
            before = heights[boundary-1, side] if boundary else bottom_units
            after = heights[boundary, side] if boundary < heights.shape[0] else bottom_units
        else:
            before = heights[side, boundary-1] if boundary else bottom_units
            after = heights[side, boundary] if boundary < heights.shape[1] else bottom_units
        assert before != after
        expected = np.array([min(before, after), max(before, after)]) * cfg.terrain.vertical_scale
        actual = np.array([quad[:, 2].min(), quad[:, 2].max()])
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
        assert np.sign(normal[axis]) == np.sign(int(before) - int(after))
    assert bottom_count == 1 and wall_count > 0


def test_distance_curriculum_updates_origins_and_skips_initial_reset():
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
    task._update_terrain_curriculum(torch.arange(4))
    assert task.terrain_levels[:3].tolist() == [3, 1, 0]
    assert 0 <= task.terrain_levels[3] < 10
    torch.testing.assert_close(task.env_origins,
                               task.terrain_origins[task.terrain_levels, task.terrain_types])


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


def test_asset_complete_joint_order_and_stand_pose():
    cfg = EnvConfig()
    path = cfg.resolve_urdf()
    root = ET.parse(str(path)).getroot()
    movable = [j.attrib["name"] for j in root.findall("joint") if j.attrib["type"] != "fixed"]
    assert movable == cfg.robot.joint_names
    for mesh in root.findall(".//mesh"):
        assert (path.parent / mesh.attrib["filename"]).resolve().is_file()
    fk = UrdfKinematics(path, cfg.robot.joint_names, cfg.robot.base_name, "cpu")
    states = fk.forward(torch.tensor([[0.0, 0.0, cfg.robot.base_height]]),
                        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                        torch.tensor([cfg.robot.stand_angles]))
    feet = torch.cat([states[name][0] for name in cfg.robot.foot_names])
    assert torch.all(feet[:, 2] > 0.0)
    assert torch.all(feet[:, 2] < 0.1)
    assert torch.all(feet[:2, 0] > feet[2:, 0])
