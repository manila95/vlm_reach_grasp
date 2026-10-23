"""Isaac Lab backend: Franka Panda + Franka Hand on a table, YCB objects, an external RGB-D camera and a wrist RGB-D
camera (RealSense-like intrinsics), implementing vrg.franka.core.Robot (+ SimOracle for evaluation).

Usage (Isaac env; needs DISPLAY=:147 for Vulkan on this machine, see CLAUDE.md):
    from vrg.franka.isaac_backend import launch; app = launch()          # must come before any isaaclab import
    from vrg.franka.isaac_backend import IsaacFranka; env = IsaacFranka(); env.reset(seed=0)

World frame == robot base frame (panda_link0 at the origin); the table top is the plane z = 0.
Control: joint-position targets from vrg.franka.kinematics (the same trajectories the real arm executes), or the
"native" Isaac Lab differential-IK controller (move_tcp_native) as the baseline controller.
"""
import json, math
import numpy as np

ASSETS = "/scratch/kaustubh/vlm_reach_grasp/isaac/assets/ycb"
# natural-language names used in instructions; graspable = narrowest horizontal side fits the 8 cm Franka Hand
OBJECTS = {
    "003_cracker_box": ("cracker box", True), "004_sugar_box": ("sugar box", True),
    "005_tomato_soup_can": ("tomato soup can", True), "006_mustard_bottle": ("mustard bottle", True),
    "008_pudding_box": ("pudding box", True), "009_gelatin_box": ("gelatin box", True),
    "010_potted_meat_can": ("potted meat can", True), "011_banana": ("banana", True),
    "021_bleach_cleanser": ("bleach bottle", True), "037_scissors": ("scissors", True),
    "040_large_marker": ("marker", True), "051_large_clamp": ("clamp", True), "061_foam_brick": ("foam brick", True),
    "002_master_chef_can": ("coffee can", False), "007_tuna_fish_can": ("tuna can", False), "024_bowl": ("bowl", False),
    "025_mug": ("mug", False), "036_wood_block": ("wooden block", False),
}
# RealSense D435 colour stream at 640x480 (fx = fy ~ 615 px). Change to your calibrated intrinsics for real-sim parity.
K_D435 = np.array([[615.0, 0, 320.0], [0, 615.0, 240.0], [0, 0, 1]])
EXT_CAM_POS = (1.25, 0.35, 0.85)          # tripod opposite the robot, looking at the table centre
EXT_CAM_TARGET = (0.45, 0.00, 0.12)
WRIST_POS, WRIST_PITCH = (0.065, 0.0, 0.04), 0.0       # wrist camera in the panda_hand frame (hand-eye calibration on the real arm)
SIM_DT, CTRL_EVERY = 0.005, 2             # physics 200 Hz (CPU PhysX: 0.5 ms/step vs 18 ms on GPU for this scene), joint targets 100 Hz


def launch(headless=True):
    from isaaclab.app import AppLauncher
    return AppLauncher(headless=headless, enable_cameras=True).app


def look_at_quat_ros(pos, target):
    """(w, x, y, z) of a ROS-convention camera (z forward, y down) at pos looking at target, world z up."""
    z = np.asarray(target, float) - np.asarray(pos, float); z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(z, x)
    R = np.stack([x, y, z], 1)
    return mat_to_quat(R)


def mat_to_quat(R):
    w = math.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return (w, x, y, z)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class IsaacFranka:
    def __init__(self, device="cpu", n_objects=6, img_hw=(480, 640), record_video=False, extra_cams=None, video_cam="ext"):
        """extra_cams: {name: (pos, target)} or {name: dict(pos, target | pitch, mount="world"|"hand", K, hw)}: additional
        cameras, e.g. DROID-like wide-angle side and wrist views for VLA policies (the pipeline keeps its own cameras)."""
        self.extra_cams = extra_cams or {}; self.video_cam = video_cam
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
        from isaaclab.sensors.camera import Camera, CameraCfg
        from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
        from vrg.franka.core import CameraFrame, RobotState, MotionResult
        self.torch, self.CameraFrame, self.RobotState, self.MotionResult = torch, CameraFrame, RobotState, MotionResult
        self.device, self.n_objects, self.record_video, self.video = device, n_objects, record_video, []
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device=device, render_interval=CTRL_EVERY))
        self.meta = json.load(open(f"{ASSETS}/ycb_meta.json"))

        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg(), translation=(0, 0, -0.80))
        sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
        table = sim_utils.CuboidCfg(size=(1.4, 1.4, 0.05), collision_props=sim_utils.CollisionPropertiesCfg(),
                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
                                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7))
        table.func("/World/table", table, translation=(0.45, 0.0, -0.025))

        rcfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/robot")
        self.robot = Articulation(rcfg)
        self.objs = {}
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9)
        for i, name in enumerate(OBJECTS):
            cfg = RigidObjectCfg(prim_path=f"/World/obj_{name}",
                                 spawn=sim_utils.UsdFileCfg(usd_path=self.meta[name]["usd"],
                                                            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                                                            semantic_tags=[("class", name)]),
                                 init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.0 - 0.3 * i, 2.0, 0.1)))
            self.objs[name] = RigidObject(cfg)
            mat.func(f"/World/obj_{name}/physics_material", mat)
        H, W = img_hw
        spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(intrinsic_matrix=K_D435.flatten().tolist(), width=W, height=H,
                                                                 clipping_range=(0.05, 5.0))
        dtypes = ["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"]
        self.cams = {
            "ext": Camera(CameraCfg(prim_path="/World/ext_cam", height=H, width=W, data_types=dtypes, spawn=spawn, update_period=0, colorize_instance_id_segmentation=False,
                                    offset=CameraCfg.OffsetCfg(pos=EXT_CAM_POS, rot=look_at_quat_ros(EXT_CAM_POS, EXT_CAM_TARGET), convention="ros"))),
            # wrist camera on the hand, 6.5 cm off-axis and 4 cm up from the hand frame, tilted so it sees the finger tips
            "wrist": Camera(CameraCfg(prim_path="/World/robot/panda_hand/wrist_cam", height=H, width=W, data_types=dtypes, spawn=spawn,
                                      update_period=0, colorize_instance_id_segmentation=False, offset=CameraCfg.OffsetCfg(pos=WRIST_POS,
                                      rot=mat_to_quat(_rot_y(WRIST_PITCH)), convention="ros"))),
        }
        self.extra_cams = {n: (dict(pos=c[0], target=c[1]) if isinstance(c, tuple) else c) for n, c in self.extra_cams.items()}
        for name, c in self.extra_cams.items():
            Kc, (h, w) = np.asarray(c.get("K", K_D435)), c.get("hw", (H, W))
            sp = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(intrinsic_matrix=Kc.flatten().tolist(), width=w, height=h, clipping_range=(0.02, 5.0))
            if c.get("mount", "world") == "hand":
                prim, rot = f"/World/robot/panda_hand/{name}_cam", mat_to_quat(_rot_y(c.get("pitch", 0.0)))
            else:
                prim, rot = f"/World/{name}_cam", look_at_quat_ros(c["pos"], c["target"])
            self.cams[name] = Camera(CameraCfg(prim_path=prim, height=h, width=w, data_types=dtypes, spawn=sp, update_period=0,
                                               colorize_instance_id_segmentation=False, offset=CameraCfg.OffsetCfg(pos=c["pos"], rot=rot, convention="ros")))
        self.T_HAND_WRISTCAM = np.eye(4); self.T_HAND_WRISTCAM[:3, :3] = _rot_y(WRIST_PITCH); self.T_HAND_WRISTCAM[:3, 3] = WRIST_POS
        self.sim.reset()
        self.arm_ids, _ = self.robot.find_joints("panda_joint[1-7]")
        self.finger_ids, _ = self.robot.find_joints("panda_finger_joint.*")
        self.hand_idx = self.robot.find_bodies("panda_hand")[0][0]
        self.scene_objs, self.target = [], None
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ stepping
    def _step(self, n=1, render=False):
        for i in range(n):
            self.robot.write_data_to_sim()
            self.sim.step(render=render and i == n - 1)
            self.robot.update(SIM_DT)
            for o in self.objs.values(): o.update(SIM_DT)

    def _q(self):
        return self.robot.data.joint_pos[0, self.arm_ids].cpu().numpy().astype(float)

    def _set_targets(self, q=None, finger=None):
        t = self.torch
        if q is not None: self.robot.set_joint_position_target(t.tensor(q, dtype=t.float32, device=self.device)[None], joint_ids=self.arm_ids)
        if finger is not None: self.robot.set_joint_position_target(t.full((1, 2), float(finger), device=self.device), joint_ids=self.finger_ids)

    # ------------------------------------------------------------------ Robot interface
    def state(self):
        from vrg.franka.kinematics import fk
        q = self._q(); w = float(self.robot.data.joint_pos[0, self.finger_ids].sum())
        return self.RobotState(q=q, T_tcp=fk(q), gripper_width=w)

    def T_tcp_sim(self):
        """TCP pose from the simulator (panda_hand body + 0.1034 m), to check the numpy kinematics."""
        p = self.robot.data.body_pos_w[0, self.hand_idx].cpu().numpy(); q = self.robot.data.body_quat_w[0, self.hand_idx].cpu().numpy()
        T = np.eye(4); T[:3, :3] = quat_to_mat(q); T[:3, 3] = p + T[:3, :3] @ [0, 0, 0.1034]
        return T

    def T_base_cam(self, name):
        """Camera pose (OpenCV/ROS convention) as a calibration would give it: fixed for the external camera,
        hand pose x mount offset for the wrist camera. (Isaac's own camera pose buffers are empty with CPU physics.)"""
        if name in self.extra_cams and self.extra_cams[name].get("mount", "world") == "hand":
            c = self.extra_cams[name]; Tm = np.eye(4); Tm[:3, :3] = _rot_y(c.get("pitch", 0.0)); Tm[:3, 3] = c["pos"]
            p = self.robot.data.body_pos_w[0, self.hand_idx].cpu().numpy(); q = self.robot.data.body_quat_w[0, self.hand_idx].cpu().numpy()
            Th = np.eye(4); Th[:3, :3] = quat_to_mat(q); Th[:3, 3] = p; return Th @ Tm
        if name == "ext" or name in self.extra_cams:
            pos, tgt = (EXT_CAM_POS, EXT_CAM_TARGET) if name == "ext" else (self.extra_cams[name]["pos"], self.extra_cams[name]["target"])
            T = np.eye(4); T[:3, :3] = quat_to_mat(look_at_quat_ros(pos, tgt)); T[:3, 3] = pos; return T
        p = self.robot.data.body_pos_w[0, self.hand_idx].cpu().numpy(); q = self.robot.data.body_quat_w[0, self.hand_idx].cpu().numpy()
        Th = np.eye(4); Th[:3, :3] = quat_to_mat(q); Th[:3, 3] = p
        return Th @ self.T_HAND_WRISTCAM

    def frames(self):
        # render twice: the instance-segmentation annotator can lag rgb/depth by one render, which put ground-truth masks
        # where a still-settling object was a moment earlier (12/257 eval scenes, 2026-10-16)
        self.sim.render(); self.sim.render()
        out = {}
        for name, cam in self.cams.items():
            cam.update(0.0, force_recompute=True); d = cam.data
            rgb = d.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            depth = d.output["distance_to_image_plane"][0, ..., 0].cpu().numpy().astype(np.float32)
            depth[~np.isfinite(depth) | (depth > 4.0)] = 0
            T = self.T_base_cam(name)
            f = self.CameraFrame(name=name, rgb=rgb, depth=depth, K=d.intrinsic_matrices[0].cpu().numpy(), T_base_cam=T)
            seg = d.output["instance_id_segmentation_fast"][0, ..., 0].cpu().numpy()
            f._seg, f._seg_info = seg, d.info[0]["instance_id_segmentation_fast"]["idToLabels"]
            out[name] = f
        self._last_frames = out
        return out

    settle_s, settle_tol = 0.3, 0.01                 # wait after each move: at most settle_s, until every joint is within settle_tol rad

    def follow_joint_trajectory(self, Q, dt=0.01, settle=None, tol=None):
        settle = self.settle_s if settle is None else settle; tol = self.settle_tol if tol is None else tol
        steps = max(1, int(round(dt / SIM_DT)))
        for k, q in enumerate(Q):
            self._set_targets(q=q); self._step(steps, render=self.record_video and k % 5 == 0)
            if self.record_video and k % 5 == 0: self._grab_video()
        waited = 0.0
        for _ in range(int(settle / (SIM_DT * steps))):
            if np.max(np.abs(self._q() - Q[-1])) < tol: break
            self._step(steps); waited += SIM_DT * steps
        from vrg.franka.kinematics import fk, pose_err
        e = pose_err(fk(Q[-1]), self.T_tcp_sim())
        return self.MotionResult(ok=True, final_err_pos=float(np.linalg.norm(e[:3])), final_err_rot=float(np.linalg.norm(e[3:])), info=dict(settle_s=waited))

    def servo(self, q, finger_width, dt):
        """Streaming control (VLA policies): set joint and gripper targets, then simulate dt seconds. Records video at the control rate."""
        from vrg.franka.kinematics import Q_MIN, Q_MAX
        self._set_targets(q=np.clip(q, Q_MIN, Q_MAX), finger=finger_width / 2); self._step(max(1, int(round(dt / SIM_DT))), render=self.record_video)
        if self.record_video: self._grab_video()

    def move_tcp_native(self, T_goal, max_time=4.0, tol=0.003):
        """Baseline controller: Isaac Lab DifferentialIKController (DLS) tracking the goal pose directly, no trajectory."""
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        t = self.torch
        if not hasattr(self, "_dik"):
            self._dik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=1, device=self.device)
            self._jac_idx = self.hand_idx - 1            # fixed-base: jacobian rows exclude the root body
        # goal for the panda_hand frame = TCP goal shifted back by 0.1034 m along its z
        Th = T_goal.copy(); Th[:3, 3] = T_goal[:3, 3] - T_goal[:3, :3] @ [0, 0, 0.1034]
        self._dik.set_command(t.tensor(np.r_[Th[:3, 3], mat_to_quat(Th[:3, :3])], dtype=t.float32, device=self.device)[None])
        for k in range(int(max_time / (SIM_DT * CTRL_EVERY))):
            J = self.robot.root_physx_view.get_jacobians()[:, self._jac_idx, :, self.arm_ids]
            q = self.robot.data.joint_pos[:, self.arm_ids]
            qd = self._dik.compute(self.robot.data.body_pos_w[:, self.hand_idx], self.robot.data.body_quat_w[:, self.hand_idx], J, q)
            self.robot.set_joint_position_target(qd, joint_ids=self.arm_ids); self._step(CTRL_EVERY, render=self.record_video and k % 5 == 0)
            if self.record_video and k % 5 == 0: self._grab_video()
            from vrg.franka.kinematics import pose_err
            e = pose_err(T_goal, self.T_tcp_sim())
            if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < 0.02: break
        e = pose_err(T_goal, self.T_tcp_sim())
        return self.MotionResult(ok=bool(np.linalg.norm(e[:3]) < 0.01), final_err_pos=float(np.linalg.norm(e[:3])), final_err_rot=float(np.linalg.norm(e[3:])))

    def gripper_move(self, width):
        self._set_targets(finger=width / 2); self._step(int(0.6 / SIM_DT))

    def gripper_grasp(self, width=0.0, force=40.0, min_width=0.002):
        self._set_targets(finger=width / 2); self._step(int(0.8 / SIM_DT))
        return self.state().gripper_width > min_width

    def home(self, q=None):
        from vrg.franka.kinematics import Q_HOME
        q = Q_HOME if q is None else np.asarray(q)
        self.robot.write_joint_state_to_sim(self.torch.tensor(np.r_[q, 0.04, 0.04], dtype=self.torch.float32, device=self.device)[None],
                                            self.torch.zeros(1, 9, device=self.device))
        self._set_targets(q=q, finger=0.04); self._step(20)

    # ------------------------------------------------------------------ scenes
    def reset(self, seed, n_objects=None, region=((0.35, 0.70), (-0.30, 0.30)), min_gap=0.11):
        """Random scene: n objects (at least half graspable) at random non-overlapping xy and yaw, dropped and settled."""
        rng = np.random.default_rng(seed); n = n_objects or self.n_objects; t = self.torch
        names = list(OBJECTS); grasp = [k for k in names if OBJECTS[k][1]]; other = [k for k in names if not OBJECTS[k][1]]
        k_g = max(n // 2 + 1, n - rng.integers(0, 3))
        chosen = list(rng.choice(grasp, min(k_g, len(grasp)), replace=False)) + list(rng.choice(other, max(n - k_g, 0), replace=False))
        xy = []
        while len(xy) < len(chosen):
            p = np.array([rng.uniform(*region[0]), rng.uniform(*region[1])])
            if all(np.linalg.norm(p - q) > min_gap for q in xy): xy.append(p)
        self.home()
        for i, name in enumerate(OBJECTS):
            o = self.objs[name]
            if name in chosen:
                j = chosen.index(name); yaw = rng.uniform(-np.pi, np.pi)
                # assets are Y-up: rotate +90 deg about x so their up axis is world z, then yaw about z
                R = _rot_z(yaw) @ _rot_x(np.pi / 2); lo = np.array(self.meta[name]["aabb_min"])
                pos = np.r_[xy[j], -lo[1] + 0.01]
            else:
                R, pos = np.eye(3), np.array([-2.0 - 0.3 * i, 2.0, 0.1])
            o.write_root_pose_to_sim(t.tensor(np.r_[pos, mat_to_quat(R)], dtype=t.float32, device=self.device)[None])
            o.write_root_velocity_to_sim(t.zeros(1, 6, device=self.device))
        self._step(int(1.0 / SIM_DT))                       # settle
        self.scene_objs = chosen
        self.init_pose = {n_: self.obj_pose(n_) for n_ in chosen}
        return [(n_, OBJECTS[n_][0], OBJECTS[n_][1]) for n_ in chosen]

    # ------------------------------------------------------------------ oracle (evaluation only)
    def obj_pose(self, name):
        d = self.objs[name].data; T = np.eye(4)
        T[:3, :3] = quat_to_mat(d.root_quat_w[0].cpu().numpy()); T[:3, 3] = d.root_pos_w[0].cpu().numpy(); return T

    def obj_points_mask(self, frame, name):
        ids = [int(k) for k, v in frame._seg_info.items() if f"/World/obj_{name}" in str(v)]
        return np.isin(frame._seg, ids)

    def lifted(self, name, dz=0.05):
        return self.obj_pose(name)[2, 3] - self.init_pose[name][2, 3] > dz

    def moved_objects(self, thresh=0.03):
        return [n for n in self.scene_objs if np.linalg.norm(self.obj_pose(n)[:3, 3] - self.init_pose[n][:3, 3]) > thresh]

    def _grab_video(self):
        self.sim.render(); c = self.cams[self.video_cam]; c.update(0.0, force_recompute=True)
        self.video.append(c.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))


def _rot_x(a):
    c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
