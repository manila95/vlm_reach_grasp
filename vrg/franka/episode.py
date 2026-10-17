"""One pick episode on any Robot backend: observe -> propose -> ground -> plan grasp -> execute -> check.

Backend-agnostic: the same function runs in Isaac Lab and on the real arm. Success checking and failure attribution use
the simulator oracle when one is given; on the real arm, `held` (gripper width after closing and lifting) is the signal.

Controllers (how the arm gets to each pose):
  joint      our IK at the goal + min-jerk joint interpolation (curved TCP path) for every move
  cartesian  joint move to the pre-grasp (free space), then straight-line TCP approach and lift (IK per waypoint)
  native     the backend's own Cartesian controller for every move (Isaac: DifferentialIK DLS; real: franky CartesianMotion)
"""
import time
import numpy as np
from vrg.franka import kinematics as K
from vrg.franka.perception import proposals, top_down_grasp

PRE_Z, LIFT_Z = 0.12, 0.15
LOOK_Z = 0.22                   # TCP height above the object top for the wrist "look" view
CAM_IN_TCP = np.array([0.065, 0.0, 0.04 - 0.1034])      # wrist camera position in the TCP frame (hand-eye calibration)


def look_pose(prop):
    """TCP pose that puts the wrist camera straight above the object, LOOK_Z above its top, pointing down."""
    T = np.eye(4); T[:3, :3] = np.diag([1.0, -1.0, -1.0])             # TCP pointing down, hand x forward
    T[:2, 3] = prop["center"][:2] - (T[:3, :3] @ CAM_IN_TCP)[:2]; T[2, 3] = prop["top"] + LOOK_Z
    return T


def match_wrist(F, prop, max_d=0.06):
    """Wrist-view proposals and the one matching `prop` (nearest in xy within max_d), or None."""
    props_w, _ = proposals({"wrist": F["wrist"]})
    if not props_w: return props_w, None, np.inf
    d = [np.linalg.norm(p["center"][:2] - prop["center"][:2]) for p in props_w]; k = int(np.argmin(d))
    return props_w, (props_w[k] if d[k] <= max_d else None), float(d[k])


def look_refine(robot, prop, table, controller, grasp_mode):
    """Move the wrist camera straight above the chosen object, re-segment it from the top view, and re-plan the grasp.
    A side-on external view sees only front/top faces, which biases the footprint (centre and angle) of tall objects."""
    r, info = move(robot, look_pose(prop), controller if controller == "native" else "joint", straight=False)
    if r is None: return None, dict(reason="ik_look")
    F = robot.frames(); props_w, q, d = match_wrist(F, prop)
    if not props_w: return None, dict(reason="no_wrist_props")
    if q is None: return None, dict(reason="wrist_no_match", dist=d)
    g = top_down_grasp(q, table, yaw_mode=grasp_mode)
    return g, dict(reason="", shift=float(np.linalg.norm(q["center"][:2] - prop["center"][:2])), wrist_props=len(props_w),
                   _wrist=dict(frame=F["wrist"], props=props_w, match=q["id"]))           # for debugging dumps (popped before json)


def move(robot, T, controller, straight):
    """Move the TCP to T. Returns (MotionResult or None, info)."""
    if controller == "native":
        r = robot.move_tcp_native(T); return r, dict(reason="" if r.ok else "track")
    q0 = robot.state().q
    Q, info = (K.plan_cartesian(q0, T) if (straight and controller == "cartesian") else K.plan_joint(q0, T))
    if Q is None: return None, info
    return robot.follow_joint_trajectory(Q, dt=0.01), info


def run(robot, target_desc, ground, grasp_mode="pca", controller="cartesian", cams=("ext",), gt=None, propose=None, refine=False):
    """ground(frames, props, target_desc) -> (proposal id or None, grounding record).
    propose: optional replacement for perception.proposals (e.g. ground-truth segmentation in sim).
    gt: optional evaluator with .label(frames, props) -> {pid: object name}, .target, .lifted(), .moved(), .held_name()
    Returns a record with the outcome and the first failing stage."""
    rec = dict(target=target_desc, grasp_mode=grasp_mode, controller=controller, fail=None, t={})
    t0 = time.time(); robot.home(); robot.gripper_move(0.08)
    F = robot.frames(); props, table = (propose or proposals)({k: F[k] for k in cams})
    rec["_frames"], rec["_props"] = F, props          # for saving scenes / offline grounding (stripped before json)
    rec["n_props"] = len(props); rec["t"]["perceive"] = time.time() - t0
    if gt is not None:
        lab = gt.label(F, props); rec["prop_labels"] = lab
        rec["target_segmented"] = any(v == gt.target for v in lab.values())
    t0 = time.time(); pid, grec = ground(F, props, target_desc); rec["grounding"] = grec; rec["t"]["ground"] = time.time() - t0
    if gt is not None:
        rec["chosen_obj"] = rec["prop_labels"].get(pid) if pid is not None else None
        rec["grounding_correct"] = rec["chosen_obj"] == gt.target
    if pid is None: rec["fail"] = "segmentation" if (gt is not None and not rec["target_segmented"]) else "grounding"; return rec
    if gt is not None and not rec["target_segmented"]: rec["fail"] = "segmentation"
    elif gt is not None and not rec["grounding_correct"]: rec["fail"] = "grounding"
    prop = next(p for p in props if p["id"] == pid)
    g = top_down_grasp(prop, table, yaw_mode=grasp_mode)
    if refine:
        g2, rinfo = look_refine(robot, prop, table, controller, grasp_mode); rec["_look"] = rinfo.pop("_wrist", None); rec["refine"] = rinfo
        if g2 is not None: g = g2
    return execute(robot, g, rec, controller, gt)


def execute(robot, g, rec, controller, gt=None):
    """Pre-grasp -> straight approach -> close -> lift, then success and failure attribution (rec["fail"] keeps an earlier cause)."""
    rec["grasp"] = dict(width=g["width"], feasible=g["feasible"], T=g["T"].tolist())
    if not g["feasible"]: rec["fail"] = rec["fail"] or "grasp_plan"; return rec
    T_pre = g["T"].copy(); T_pre[2, 3] += PRE_Z
    t0 = time.time()
    r, info = move(robot, T_pre, controller if controller == "native" else "joint", straight=False)
    if r is None: rec["fail"] = rec["fail"] or "ik"; rec["ik_info"] = info; return rec
    r, info = move(robot, g["T"], controller, straight=True)
    if r is None: rec["fail"] = rec["fail"] or "ik"; rec["ik_info"] = info; return rec
    rec["reach_err"] = r.final_err_pos; rec["reach_err_rot"] = r.final_err_rot
    rec["T_goal"] = g["T"].tolist(); rec["T_reached"] = robot.state().T_tcp.tolist()
    if gt is not None: rec["disturbed_before_close"] = gt.moved(exclude_target=True)
    held = robot.gripper_grasp(width=0.0); rec["held_after_close"] = held; rec["width_after_close"] = robot.state().gripper_width
    T_up = robot.state().T_tcp.copy(); T_up[2, 3] += LIFT_Z
    r, info = move(robot, T_up, controller, straight=True)
    if r is None: rec["fail"] = rec["fail"] or "ik_lift"
    rec["t"]["execute"] = time.time() - t0
    rec["held_after_lift"] = robot.state().gripper_width > 0.002
    if gt is not None:
        rec["lifted"] = gt.lifted_name(); rec["success"] = rec["lifted"] == gt.target
        if not rec["success"] and rec["fail"] is None:
            if rec["reach_err"] > 0.01: rec["fail"] = "approach_collision" if rec.get("disturbed_before_close") or gt.target in gt.moved(exclude_target=False) else "reach"
            elif rec["lifted"] is not None: rec["fail"] = "wrong_object"
            elif not held: rec["fail"] = "grasp_miss"
            else: rec["fail"] = "grasp_slip"
    else:
        rec["success"] = rec["held_after_lift"]
    return rec
