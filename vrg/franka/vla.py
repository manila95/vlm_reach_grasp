"""End-to-end VLA episode (e.g. pi0.5-DROID through an openpi policy server) on any backend with a `servo` command.

Reproduces openpi's DROID client (examples/droid/main.py) and DROID's control law (droid/franka/robot.py):
  * observation: external + wrist RGB, each resized with padding to 224x224; 7 joint positions; gripper position in [0, 1]
    (1 = closed, = 1 - width / max_width); the language prompt.
  * a 15-step action chunk is requested every `open_loop` (8) steps; control at 15 Hz.
  * action ("velocity", pi05_droid): 7 normalised joint velocities v in [-1, 1] -> joint target q + 0.2 rad * v (rescaled so
    max |v| <= 1). openpi notes that joint-velocity policies are hard to evaluate in simulation (the real arm's soft
    impedance controller lags each target; a stiff simulated PD does not).
    action ("position", pi05_droid_jointpos*): 7 absolute joint targets (the server converts the predicted deltas back).
    Gripper command binarised at 0.5 -> width = max_width * (1 - cmd).
The robot acts only on images and joint state: no proposals, no VLM grounding, no scripted grasp. With a sim oracle the
episode is scored like the modular pipeline (target lifted > 5 cm) and logs which object the hand went for.
"""
import time
import numpy as np

HZ, MAX_JOINT_DELTA = 15, 0.2
DROID_RESET = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])     # droid/robot_env.py reset_joints
DROID_CAMS = {  # DROID's external cameras sit beside the robot looking across the workspace (a varied, hand-placed mount)
    "droid_left": ((0.60, 0.80, 0.65), (0.30, 0.0, 0.10)), "droid_right": ((0.60, -0.80, 0.65), (0.30, 0.0, 0.10))}   # chosen by eye: base at the side, hand and objects in view
# DROID records with wide-angle ZED cameras at 16:9 (ZED 2 exterior, HD720 fx ~ 528 px -> ~100 deg horizontal; ZED mini wrist
# fx ~ 700 px -> ~85 deg). Rendered here at 640x360 with the same fields of view.
K_ZED2 = np.array([[264.0, 0, 320], [0, 264.0, 180], [0, 0, 1]]); K_ZEDM = np.array([[350.0, 0, 320], [0, 350.0, 180], [0, 0, 1]])
DROID_WIDE = {"droid_left_wide": dict(pos=(0.60, 0.80, 0.65), target=(0.30, 0.0, 0.10), K=K_ZED2, hw=(360, 640)),
              "wrist_wide": dict(pos=(0.065, 0.0, 0.04), pitch=0.0, mount="hand", K=K_ZEDM, hw=(360, 640))}


class ToolRetarget:
    """Make a Franka Hand behave like the policy's training gripper (DROID: Robotiq 2F-85, longer, mounted without the hand's
    -45 deg twist). The policy commands a virtual arm carrying the virtual gripper; each step, IK moves the real arm so the
    Franka Hand TCP lands on the virtual gripper's TCP (same position and finger axes). The policy observes the virtual joints."""
    def __init__(self, dL, yaw=np.pi / 4):
        from vrg.franka import kinematics as K
        self.K = K; T = np.eye(4); T[2, 3] = K.D_TCP + dL
        self.T_rob = T @ K._rz(-np.pi / 4 + yaw)                 # virtual gripper TCP in the flange frame
        self.T_fl_tcp_inv = np.linalg.inv(K.T_FLANGE_TCP)

    def real_target(self, q_virtual, q_seed):
        T = self.K.fk(q_virtual) @ self.T_fl_tcp_inv @ self.T_rob      # where the virtual gripper's TCP is
        q, ok, *_ = self.K.ik(T, q_seed, q_rest=q_seed, restarts=0, iters=60); return q

    def virtual_state(self, q_real, q_seed):
        T = self.K.fk(q_real) @ np.linalg.inv(self.T_rob) @ self.K.T_FLANGE_TCP   # the virtual arm whose gripper TCP = our TCP
        q, ok, *_ = self.K.ik(T, q_seed, q_rest=q_seed, restarts=0, iters=60); return q


def obj_top(env, name):
    """Height of an object's highest point (from its pose and mesh box), for comparing with where the gripper closed."""
    T = env.obj_pose(name); lo, hi = np.array(env.meta[name]["aabb_min"]), np.array(env.meta[name]["aabb_max"])
    C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ T[:3, :3].T + T[:3, 3]
    return C[:, 2].max()


def resize_with_pad(img, h=224, w=224):
    from PIL import Image
    H, W = img.shape[:2]; s = min(h / H, w / W); nh, nw = int(round(H * s)), int(round(W * s))
    out = np.zeros((h, w, 3), np.uint8); r = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    y0, x0 = (h - nh) // 2, (w - nw) // 2; out[y0:y0 + nh, x0:x0 + nw] = r; return out


def run(robot, prompt, client, max_steps=450, open_loop=8, max_width=0.08, gt=None, hold_steps=10, video=None, ext_cam="ext", q0=None, retarget=None, action="velocity", wrist_cam="wrist", from_current=False):
    """from_current: start from wherever the arm is (after a scripted handoff) instead of resetting to q0 with an open gripper."""
    rec = dict(target=prompt, prompt=prompt, fail=None, t={}, ext_cam=ext_cam, action_space=action, retarget_dL=None if retarget is None else retarget.T_rob[2, 3] - 0.1034)
    if not from_current: robot.home(q0); robot.gripper_move(max_width)
    q_v = None if retarget is None else retarget.virtual_state(robot.state().q, robot.state().q)
    chunk, k, n_inf, t_inf, held, first_close = None, 0, 0, 0.0, 0, None
    tgt = gt.target if gt is not None else None
    min_d = np.inf; closes = 0; prev_cmd = 0; close_log = []; path = []
    for step in range(max_steps):
        st = robot.state(); q_obs = st.q if retarget is None else q_v
        if chunk is None or k >= open_loop:
            F = robot.frames()
            obs = {"observation/exterior_image_1_left": resize_with_pad(F[ext_cam].rgb), "observation/wrist_image_left": resize_with_pad(F[wrist_cam].rgb),
                   "observation/joint_position": q_obs.astype(np.float64), "observation/gripper_position": np.array([1 - st.gripper_width / max_width]),
                   "prompt": prompt}
            t0 = time.time(); chunk = np.asarray(client.infer(obs)["actions"]); t_inf += time.time() - t0; n_inf += 1; k = 0
            if step == 0: rec["chunk_shape"] = list(chunk.shape)
            if video is not None: video.append(F[ext_cam].rgb)
        a = chunk[k]; k += 1
        cmd = 1.0 if a[-1] > 0.5 else 0.0
        if action == "velocity":
            v = np.clip(a[:7], -1, 1); m = np.abs(v).max(); v = v / m if m > 1 else v
            q_cmd = q_obs + MAX_JOINT_DELTA * v
        else:
            q_cmd = np.asarray(a[:7], float)
        if retarget is not None: q_cmd_real = retarget.real_target(q_cmd, st.q)
        robot.servo(q_cmd if retarget is None else q_cmd_real, max_width * (1 - cmd), 1.0 / HZ)
        if retarget is not None: q_v = retarget.virtual_state(robot.state().q, q_cmd)
        if gt is not None:
            p = robot.state().T_tcp[:3, 3]; d = float(np.linalg.norm(p - gt.env.obj_pose(tgt)[:3, 3])); min_d = min(min_d, d)
            if cmd > prev_cmd:
                closes += 1
                dist = {n: float(np.linalg.norm(p[:2] - gt.env.obj_pose(n)[:2, 3])) for n in gt.env.scene_objs}; near = min(dist, key=dist.get)
                close_log.append(dict(step=step, tcp_z=float(p[2]), near=near, xy_dist=dist[near], near_top=float(obj_top(gt.env, near))))
                if first_close is None:      # which object is the hand at when it first closes? (the VLA's implicit "grounding")
                    dist = {n: float(np.linalg.norm(p[:2] - gt.env.obj_pose(n)[:2, 3])) for n in gt.env.scene_objs}
                    first_close = dict(step=step, obj=min(dist, key=dist.get), dist_to_obj=min(dist.values()), dist_to_target=dist[tgt])
            up = gt.lifted_name()
            held = held + 1 if up is not None else 0
            if held >= hold_steps: break
        prev_cmd = cmd
        if step % 3 == 0:
            p_ = robot.state().T_tcp[:3, 3]; path.append([round(float(x), 4) for x in p_] + [cmd] + ([round(float(gt.env.obj_pose(tgt)[2, 3]), 4)] if gt is not None else []))
    rec["path"] = path
    rec.update(steps=step + 1, n_inference=n_inf, inference_s=t_inf, robot_s=(step + 1) / HZ)
    if gt is not None:
        rec["lifted"] = gt.lifted_name() if held >= hold_steps else None
        rec["success"] = rec["lifted"] == tgt
        rec["min_tcp_to_target"] = min_d; rec["n_closes"] = closes; rec["first_close"] = first_close; rec["closes"] = close_log
        rec["moved"] = gt.moved(exclude_target=False)
        rec["chosen_obj"] = first_close["obj"] if first_close else None
        rec["grounding_correct"] = rec["chosen_obj"] == tgt
        if not rec["success"]:
            if rec["lifted"] is not None: rec["fail"] = "wrong_object"
            elif first_close is None: rec["fail"] = "never_closed"
            elif rec["chosen_obj"] != tgt and first_close["dist_to_obj"] < 0.05: rec["fail"] = "went_for_other_object"
            elif tgt in rec["moved"]: rec["fail"] = "touched_not_lifted"
            else: rec["fail"] = "closed_on_nothing"
    return rec
