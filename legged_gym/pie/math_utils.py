"""Quaternion operations use xyzw throughout, matching Isaac Gym and Warp."""
import torch


def quat_mul(a, b):
    av, bv = a[..., :3], b[..., :3]
    xyz = a[..., 3:] * bv + b[..., 3:] * av + torch.cross(av, bv, dim=-1)
    w = a[..., 3:] * b[..., 3:] - (av * bv).sum(-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def quat_rotate(q, v):
    t = 2 * torch.cross(q[..., :3], v, dim=-1)
    return v + q[..., 3:] * t + torch.cross(q[..., :3], t, dim=-1)


def quat_rotate_inverse(q, v):
    return quat_rotate(torch.cat((-q[..., :3], q[..., 3:]), dim=-1), v)


def quat_yaw(q):
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def axis_angle(axis, angle):
    return torch.cat((axis * torch.sin(angle[..., None] * 0.5),
                      torch.cos(angle[..., None] * 0.5)), dim=-1)
