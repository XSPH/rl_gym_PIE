"""URDF FK for reset labels, before PhysX updates rigid-body state tensors."""
import math
import xml.etree.ElementTree as ET

import torch

from isaacgym.torch_utils import quat_mul, quat_apply as quat_rotate
from .math import axis_angle


def _vector(node, key, default):
    return [float(v) for v in node.attrib.get(key, default).split()]


def _rpy_quat(rpy):
    roll, pitch, yaw = [v / 2 for v in rpy]
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


class UrdfKinematics:
    def __init__(self, path, joint_names, base_name, device):
        root = ET.parse(str(path)).getroot()
        links = {node.attrib["name"] for node in root.findall("link")}
        remaining = []
        children = set()
        for node in root.findall("joint"):
            parent, child = node.find("parent").attrib["link"], node.find("child").attrib["link"]
            children.add(child)
            origin = node.find("origin")
            xyz = _vector(origin, "xyz", "0 0 0") if origin is not None else [0, 0, 0]
            rpy = _vector(origin, "rpy", "0 0 0") if origin is not None else [0, 0, 0]
            axis = node.find("axis")
            axis_value = _vector(axis, "xyz", "1 0 0") if axis is not None else [1, 0, 0]
            kind = node.attrib["type"]
            if kind not in ("fixed", "revolute", "continuous"):
                raise ValueError("Only fixed and revolute joints are supported for reset FK.")
            name = node.attrib["name"]
            index = joint_names.index(name) if kind != "fixed" else None
            remaining.append((parent, child, torch.tensor(xyz, dtype=torch.float32, device=device),
                              torch.tensor(_rpy_quat(rpy), dtype=torch.float32, device=device),
                              torch.tensor(axis_value, dtype=torch.float32, device=device), index))
        if links - children != {base_name}:
            raise ValueError("Configured base_name must be the URDF root link.")
        self.base_name, self.joints = base_name, []
        known = {base_name}
        while remaining:
            ready = [joint for joint in remaining if joint[0] in known]
            if not ready:
                raise ValueError("URDF joint graph is disconnected or cyclic.")
            for joint in ready:
                self.joints.append(joint)
                known.add(joint[1])
                remaining.remove(joint)

    def forward(self, base_pos, base_quat, joints):
        states = {self.base_name: (base_pos, base_quat)}
        n = len(base_pos)
        for parent, child, xyz, origin_quat, axis, index in self.joints:
            p, q = states[parent]
            child_pos = p + quat_rotate(q, xyz.expand(n, -1))
            child_quat = quat_mul(q, origin_quat.expand(n, -1))
            if index is not None:
                child_quat = quat_mul(child_quat, axis_angle(axis.expand(n, -1), joints[:, index]))
            states[child] = (child_pos, child_quat)
        return states
