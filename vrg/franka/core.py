"""Backend-agnostic types and robot interface for the Franka track (sim: Isaac Lab, real: franky + RealSense).

Everything above this layer (perception, VLM grounding, grasp planning, controllers, the episode runner) only sees
these types, so moving from sim to the real arm means swapping the backend, nothing else.

Frames: all poses are 4x4 homogeneous matrices in the robot BASE frame (panda_link0). Camera extrinsics follow the
OpenCV convention (x right, y down, z forward); `T_base_cam` maps camera-frame points to the base frame.
The tool frame is the Franka Hand TCP: 0.1034 m beyond the flange along its z axis, between the fingertips.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CameraFrame:
    name: str
    rgb: np.ndarray            # H x W x 3 uint8, upright
    depth: np.ndarray          # H x W float32 metres (0 or nan = invalid)
    K: np.ndarray              # 3 x 3 intrinsics
    T_base_cam: np.ndarray     # 4 x 4, OpenCV camera -> robot base

    def backproject(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Points (N x 3, base frame) for valid depth pixels (optionally within mask); same order as np.nonzero."""
        valid = np.isfinite(self.depth) & (self.depth > 0)
        if mask is not None: valid &= mask
        v, u = np.nonzero(valid); z = self.depth[v, u]
        P = np.stack([(u - self.K[0, 2]) * z / self.K[0, 0], (v - self.K[1, 2]) * z / self.K[1, 1], z], 1)
        return P @ self.T_base_cam[:3, :3].T + self.T_base_cam[:3, 3]

    def pixel_to_base(self, u: float, v: float, patch: int = 2) -> Optional[np.ndarray]:
        """Median depth in a small window around (u, v) -> base-frame point."""
        H, W = self.depth.shape; r, c = int(round(v)), int(round(u))
        win = self.depth[max(r - patch, 0):min(r + patch + 1, H), max(c - patch, 0):min(c + patch + 1, W)]
        win = win[np.isfinite(win) & (win > 0)]
        if win.size == 0: return None
        z = float(np.median(win))
        p = np.array([(u - self.K[0, 2]) * z / self.K[0, 0], (v - self.K[1, 2]) * z / self.K[1, 1], z])
        return self.T_base_cam[:3, :3] @ p + self.T_base_cam[:3, 3]

    def project(self, P: np.ndarray) -> np.ndarray:
        """Base-frame points (N x 3) -> pixel (u, v) N x 2."""
        T = np.linalg.inv(self.T_base_cam); Pc = P @ T[:3, :3].T + T[:3, 3]
        return np.stack([self.K[0, 0] * Pc[:, 0] / Pc[:, 2] + self.K[0, 2], self.K[1, 1] * Pc[:, 1] / Pc[:, 2] + self.K[1, 2]], 1)


@dataclass
class RobotState:
    q: np.ndarray              # 7 joint positions (rad)
    T_tcp: np.ndarray          # 4 x 4 TCP pose in base frame
    gripper_width: float       # metres between the fingers (0 .. 0.08)


@dataclass
class MotionResult:
    ok: bool
    reason: str = ""
    final_err_pos: float = np.nan      # m, TCP position error at the end
    final_err_rot: float = np.nan      # rad
    info: dict = field(default_factory=dict)


class Robot:
    """What every backend implements. Motions are blocking."""
    def state(self) -> RobotState: raise NotImplementedError
    def frames(self) -> dict: raise NotImplementedError                              # name -> CameraFrame
    def follow_joint_trajectory(self, Q: np.ndarray, dt: float) -> MotionResult: raise NotImplementedError
    def move_tcp_native(self, T: np.ndarray) -> MotionResult: raise NotImplementedError   # backend's own Cartesian control
    def gripper_move(self, width: float) -> None: raise NotImplementedError
    def gripper_grasp(self, width: float = 0.0, force: float = 40.0) -> bool: raise NotImplementedError   # True if an object is held
    def home(self) -> None: raise NotImplementedError


class SimOracle:
    """Privileged ground truth, available only in simulation; used for evaluation and oracle ablations, never by the method."""
    def objects(self) -> dict: raise NotImplementedError            # name -> dict(pose 4x4, aabb (2x3), category)
    def instance_mask(self, cam: str, name: str) -> np.ndarray: raise NotImplementedError
    def lifted(self, name: str, dz: float = 0.05) -> bool: raise NotImplementedError
