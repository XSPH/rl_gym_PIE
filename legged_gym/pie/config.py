"""Serializable configuration; unspecified paper parameters are explicit choices."""
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class RobotConfig:
    profile: str = "lite3"
    urdf: str = ""
    # Policy order is independent of the URDF's internal DOF order.
    joint_names: List[str] = field(default_factory=lambda: [
        leg + "_" + joint + "_joint"
        for leg in ("FL", "FR", "HL", "HR")
        for joint in ("HipX", "HipY", "Knee")
    ])
    foot_names: List[str] = field(default_factory=lambda: [
        leg + "_FOOT" for leg in ("FL", "FR", "HL", "HR")
    ])
    base_name: str = "TORSO"
    stand_angles: List[float] = field(default_factory=lambda: [0.0, -0.8, 1.6] * 4)
    base_height: float = 0.30
    foot_radius: float = 0.022
    kp: float = 30.0
    kd: float = 0.8
    action_scale: float = 0.25
    action_clip: float = 100.0
    torque_limit: float = 30.5


@dataclass
class CameraConfig:
    height: int = 60
    width: int = 80
    history: int = 2
    update_every: int = 5
    latency_frames: int = 1
    near: float = 0.1
    far: float = 3.0
    hfov_degrees: float = 87.0
    position: List[float] = field(default_factory=lambda: [0.25, 0.0, 0.06])
    pitch_degrees: float = 30.0
    noise_std: float = 0.0
    salt_pepper_probability: float = 0.0
    # Network input: (optical-axis depth-near)/(far-near)-0.5.
    normalize: bool = True


@dataclass
class TerrainConfig:
    levels: int = 10
    variants: int = 2
    kinds: List[str] = field(default_factory=lambda: ["flat", "gap", "step", "hurdle", "stairs"])
    length: float = 8.0
    width: float = 3.0
    spacing: float = 2.0
    resolution: float = 0.05
    floor_height: float = -2.0
    initial_max_level: int = 5
    curriculum: bool = True
    max_gap: float = 1.0
    max_step: float = 0.75
    max_hurdle: float = 0.75
    max_stair: float = 0.25
    scan_x: List[float] = field(default_factory=lambda: [round(-0.8 + 0.1 * i, 3) for i in range(17)])
    scan_y: List[float] = field(default_factory=lambda: [round(-0.5 + 0.1 * i, 3) for i in range(11)])


@dataclass
class RandomizationConfig:
    enabled: bool = True
    friction: List[float] = field(default_factory=lambda: [0.2, 1.2])
    payload: List[float] = field(default_factory=lambda: [-1.0, 2.0])
    com_shift: float = 0.05
    gain_factor: List[float] = field(default_factory=lambda: [0.9, 1.1])
    motor_factor: List[float] = field(default_factory=lambda: [0.9, 1.1])
    joint_position_factor: List[float] = field(default_factory=lambda: [0.5, 1.5])
    max_delay_seconds: float = 0.015
    camera_position: float = 0.01
    camera_pitch_degrees: float = 1.0
    camera_hfov_degrees: List[float] = field(default_factory=lambda: [86.0, 88.0])


@dataclass
class EnvConfig:
    num_envs: int = 4096
    device: str = "cuda:0"
    seed: int = 1
    headless: bool = True
    physics_dt: float = 0.005
    decimation: int = 4
    episode_seconds: float = 20.0
    proprio_history: int = 10
    command_seconds: float = 10.0
    forward_velocity: List[float] = field(default_factory=lambda: [0.0, 1.5])
    yaw_velocity: List[float] = field(default_factory=lambda: [-1.2, 1.2])
    command_mode: str = "velocity"
    angular_velocity_scale: float = 0.25
    joint_velocity_scale: float = 0.05
    heightmap_offset: float = 0.5
    observation_noise: bool = True
    reward_scale_dt: bool = True
    robot: RobotConfig = field(default_factory=RobotConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)

    @property
    def policy_dt(self):
        return self.physics_dt * self.decimation

    @property
    def map_size(self):
        return len(self.terrain.scan_x) * len(self.terrain.scan_y)

    def validate(self):
        if self.num_envs < 1 or not self.device.startswith("cuda:"):
            raise ValueError("Isaac Gym GPU pipeline requires num_envs >= 1 and cuda:<index>.")
        if len(self.robot.joint_names) != 12 or len(set(self.robot.joint_names)) != 12:
            raise ValueError("Exactly 12 distinct policy joint names are required.")
        if len(self.robot.foot_names) != 4 or len(self.robot.stand_angles) != 12:
            raise ValueError("Expected 4 feet and 12 stand angles.")
        if self.decimation < 1 or self.physics_dt <= 0 or self.proprio_history < 1:
            raise ValueError("Invalid simulation timing or proprio history.")
        if self.randomization.max_delay_seconds < 0:
            raise ValueError("Action delay must be nonnegative.")
        for name in ("friction", "payload", "gain_factor", "motor_factor",
                     "joint_position_factor", "camera_hfov_degrees"):
            bounds = getattr(self.randomization, name)
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError("Invalid randomization range: " + name)
        if self.camera.history != 2 or self.camera.update_every < 1 or self.camera.latency_frames < 0:
            raise ValueError("PIE requires depth history 2 and positive update_every.")
        if not (0 < self.camera.near < self.camera.far and self.camera.height > 1 and self.camera.width > 1):
            raise ValueError("Invalid depth clipping or image resolution.")
        if self.terrain.levels < 1 or self.terrain.variants < 1:
            raise ValueError("Terrain levels and variants must be positive.")
        if self.terrain.length < 7 or self.terrain.width < 1:
            raise ValueError("Course length >= 7 m and width >= 1 m are required.")
        if not self.terrain.kinds or set(self.terrain.kinds) - {"flat", "gap", "step", "hurdle", "stairs"}:
            raise ValueError("Unsupported terrain kinds.")
        if not self.terrain.scan_x or not self.terrain.scan_y:
            raise ValueError("Height scan cannot be empty.")
        if self.command_mode not in ("velocity", "goal"):
            raise ValueError("command_mode must be velocity or goal.")
        return self

    def resolve_urdf(self):
        builtin = Path(__file__).resolve().parents[2] / "resources" / "robots" / "lite3" / "urdf" / "Lite3.urdf"
        candidate = self.robot.urdf or os.environ.get("PIE_ROBOT_URDF", "")
        if not candidate and self.robot.profile == "lite3" and builtin.is_file():
            candidate = str(builtin)
        if not candidate:
            raise FileNotFoundError("Set robot.urdf, --urdf, or PIE_ROBOT_URDF to a complete robot asset.")
        path = Path(candidate).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("Robot URDF does not exist: {}".format(path))
        return path

    def to_dict(self):
        return asdict(self)


def config_from_dict(values):
    cfg = EnvConfig()
    nested = {"robot": RobotConfig, "camera": CameraConfig, "terrain": TerrainConfig,
              "randomization": RandomizationConfig}
    for key, value in values.items():
        if key not in cfg.__dataclass_fields__:
            raise ValueError("Unknown environment configuration key: " + key)
        setattr(cfg, key, nested[key](**value) if key in nested else value)
    return cfg.validate()


def load_config(path=None):
    if path is not None:
        with open(path) as handle:
            return config_from_dict(json.load(handle))
    return EnvConfig().validate()
