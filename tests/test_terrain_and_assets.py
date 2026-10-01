import xml.etree.ElementTree as ET

import numpy as np
import torch

from legged_gym.pie.config import EnvConfig
from legged_gym.pie.kinematics import UrdfKinematics
from legged_gym.pie.terrain import TerrainSampler, build_atlas


def test_exact_plateau_geometry_and_labels():
    cfg = EnvConfig()
    atlas = build_atlas(cfg.terrain, seed=4)
    assert atlas.triangles.max() < len(atlas.vertices)
    sampler = TerrainSampler(atlas, "cpu")
    # Check heights at the centres of every slab. Mesh top vertices must contain
    # every labelled elevation, including randomly lowered gap bottoms.
    for level in (0, cfg.terrain.levels - 1):
        for column, kind in enumerate(atlas.kinds):
            x = (torch.arange(atlas.heights.shape[-1]) + 0.5) * cfg.terrain.resolution
            points = torch.zeros((1, len(x), 3))
            points[0, :, 0] = x + level * (cfg.terrain.length + cfg.terrain.spacing)
            points[0, :, 1] = column * (cfg.terrain.width + cfg.terrain.spacing) + cfg.terrain.width / 2
            actual = sampler.sample(points, torch.tensor([level]), torch.tensor([column]))
            np.testing.assert_array_equal(actual.numpy()[0], atlas.heights[level, column])
            mesh_z = np.unique(atlas.vertices[:, 2])
            assert all(value in mesh_z for value in np.unique(atlas.heights[level, column]))
    outside = torch.tensor([[[-1.0, -1.0, 0.0]]])
    assert sampler.sample(outside, torch.tensor([0]), torch.tensor([0])).item() == cfg.terrain.floor_height


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
