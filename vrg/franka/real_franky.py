"""Real-robot backend: Franka (FCI via franky) + Franka Hand + RealSense cameras, implementing vrg.franka.core.Robot.

UNTESTED ON HARDWARE (written alongside the Isaac backend so the episode code is shared). Before the first run:
  1. pip install franky-control pyrealsense2   (on the robot PC; real-time kernel as libfranka requires)
  2. In Desk, set the end effector to "Franka Hand" so O_T_EE is the TCP used here (0.1034 m, -45 deg about z).
  3. Calibrate the external camera (T_base_cam) and the wrist camera (T_tcp_cam, hand-eye) and write calib.json:
       {"ext": {"serial": "...", "T_base_cam": 4x4}, "wrist": {"serial": "...", "T_tcp_cam": 4x4}}
  4. Start with --dry_run (perception + grounding + planning only, no motion) and low speed (dynamics_factor 0.05).

Motions reuse vrg.franka.kinematics: joint trajectories are sent as franky joint waypoints (subsampled, franky's Ruckig
re-times them within the robot's limits); move_tcp_native uses franky's CartesianMotion (linear TCP motion).
"""
import json
import numpy as np
from vrg.franka.core import CameraFrame, RobotState, MotionResult, Robot
from vrg.franka.kinematics import fk, pose_err, Q_HOME


class RealSense:
    def __init__(self, serial, w=640, h=480, fps=30):
        import pyrealsense2 as rs
        self.rs = rs; self.pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps); cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        prof = self.pipe.start(cfg); self.align = rs.align(rs.stream.color)
        self.depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1.0]])
        for _ in range(15): self.pipe.wait_for_frames()             # auto-exposure settle

    def read(self):
        fr = self.align.process(self.pipe.wait_for_frames())
        rgb = np.asanyarray(fr.get_color_frame().get_data()).copy()
        depth = np.asanyarray(fr.get_depth_frame().get_data()).astype(np.float32) * self.depth_scale
        return rgb, depth


class FrankyRobot(Robot):
    def __init__(self, ip="172.16.0.2", calib="calib.json", dynamics_factor=0.05, dry_run=False):
        import franky
        self.f = franky; self.dry_run = dry_run
        self.robot = franky.Robot(ip); self.robot.relative_dynamics_factor = dynamics_factor
        self.robot.recover_from_errors()
        self.gripper = franky.Gripper(ip)
        C = json.load(open(calib))
        self.cams = {"ext": RealSense(C["ext"]["serial"]), "wrist": RealSense(C["wrist"]["serial"])}
        self.T_base_ext = np.array(C["ext"]["T_base_cam"]); self.T_tcp_wrist = np.array(C["wrist"]["T_tcp_cam"])

    def state(self):
        q = np.array(self.robot.current_joint_state.position)
        return RobotState(q=q, T_tcp=np.array(self.robot.current_cartesian_state.pose.end_effector_pose.matrix), gripper_width=float(self.gripper.width))

    def frames(self):
        T_tcp = self.state().T_tcp; out = {}
        for name, cam in self.cams.items():
            rgb, depth = cam.read()
            T = self.T_base_ext if name == "ext" else T_tcp @ self.T_tcp_wrist
            out[name] = CameraFrame(name=name, rgb=rgb, depth=depth, K=cam.K, T_base_cam=T)
        return out

    def follow_joint_trajectory(self, Q, dt=0.01, every=10):
        if self.dry_run: return MotionResult(ok=True, reason="dry_run")
        wp = [self.f.JointWaypoint(list(map(float, q))) for q in Q[every::every]] + [self.f.JointWaypoint(list(map(float, Q[-1])))]
        try: self.robot.move(self.f.JointWaypointMotion(wp))
        except Exception as e:                                   # reflex / limit violation: recover and report
            self.robot.recover_from_errors(); return MotionResult(ok=False, reason=f"franky: {e}")
        e = pose_err(fk(Q[-1]), self.state().T_tcp)
        return MotionResult(ok=True, final_err_pos=float(np.linalg.norm(e[:3])), final_err_rot=float(np.linalg.norm(e[3:])))

    def move_tcp_native(self, T):
        if self.dry_run: return MotionResult(ok=True, reason="dry_run")
        from scipy.spatial.transform import Rotation as Rot
        try: self.robot.move(self.f.CartesianMotion(self.f.Affine(list(T[:3, 3]), list(Rot.from_matrix(T[:3, :3]).as_quat()))))
        except Exception as e:
            self.robot.recover_from_errors(); return MotionResult(ok=False, reason=f"franky: {e}")
        e = pose_err(T, self.state().T_tcp)
        return MotionResult(ok=bool(np.linalg.norm(e[:3]) < 0.01), final_err_pos=float(np.linalg.norm(e[:3])), final_err_rot=float(np.linalg.norm(e[3:])))

    def gripper_move(self, width):
        if not self.dry_run: self.gripper.move(width, 0.05)

    def gripper_grasp(self, width=0.0, force=40.0):
        if self.dry_run: return False
        return bool(self.gripper.grasp(width, 0.05, force, epsilon_outer=0.08))

    def home(self):
        if not self.dry_run: self.robot.move(self.f.JointWaypointMotion([self.f.JointWaypoint(list(Q_HOME))]))
