"""Franka Emika Panda kinematics and IK in plain numpy (no simulator, no ROS): the same code plans motions in Isaac Lab
and on the real arm.

DH parameters: modified (Craig) convention from the Franka documentation (FCI "Robot and interface specifications").
TCP = flange + 0.1034 m along z, rotated -45 deg about z (the Franka Hand).

Controllers (all return a joint trajectory Q (N x 7) sampled at dt, which a backend executes):
  plan_joint      IK at the goal only, then a min-jerk joint-space move (fast, but the TCP path is not a straight line)
  plan_cartesian  straight-line TCP path (min-jerk timing), IK at every waypoint seeded by the previous one
                  -> what you want for approach / lift motions near objects
IK: damped least squares with joint-limit clamping, a null-space pull towards a comfortable posture, and random restarts.
"""
import numpy as np

# a_{i-1}, d_i, alpha_{i-1} (modified DH), joints 1..7, then the flange
DH = np.array([[0.0, 0.333, 0.0], [0.0, 0.0, -np.pi / 2], [0.0, 0.316, np.pi / 2], [0.0825, 0.0, np.pi / 2],
               [-0.0825, 0.384, -np.pi / 2], [0.0, 0.0, np.pi / 2], [0.088, 0.0, np.pi / 2]])
D_FLANGE, D_TCP = 0.107, 0.1034
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
QD_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])      # Franka "ready" posture, TCP pointing down


def _tf(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])


def _rz(t):
    c, s = np.cos(t), np.sin(t); return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


T_FLANGE_TCP = np.eye(4); T_FLANGE_TCP[2, 3] = D_TCP; T_FLANGE_TCP = T_FLANGE_TCP @ _rz(-np.pi / 4)


def fk_all(q):
    """Frames of joints 1..7 (z axis = joint axis) and the TCP, in the base frame."""
    T = np.eye(4); out = []
    for i in range(7):
        T = T @ _tf(*DH[i], q[i]); out.append(T)
    Tf = T @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, D_FLANGE], [0, 0, 0, 1.0]])
    return out, Tf @ T_FLANGE_TCP


def fk(q):
    return fk_all(q)[1]


def jacobian(q):
    """6 x 7 geometric Jacobian of the TCP (linear; angular) in the base frame."""
    Ts, Tt = fk_all(q); p = Tt[:3, 3]; J = np.zeros((6, 7))
    for i, T in enumerate(Ts):
        z, o = T[:3, 2], T[:3, 3]; J[:3, i] = np.cross(z, p - o); J[3:, i] = z
    return J


def rot_err(R_goal, R):
    """Axis-angle vector rotating R onto R_goal (base frame)."""
    E = R_goal @ R.T; c = np.clip((np.trace(E) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-9: return np.zeros(3)
    v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    if np.pi - th < 1e-4:                                            # near pi: use the symmetric part
        ax = np.sqrt(np.clip((np.diag(E) + 1) / 2, 0, 1)); ax *= np.sign(v + 1e-12); return th * ax / np.linalg.norm(ax)
    return th * v / (2 * np.sin(th))


def pose_err(T_goal, T):
    return np.r_[T_goal[:3, 3] - T[:3, 3], rot_err(T_goal[:3, :3], T[:3, :3])]


def ik(T_goal, q_seed, q_rest=Q_HOME, iters=200, tol_pos=1e-3, tol_rot=1e-2, damping=0.05, null_gain=0.1,
       restarts=8, rng=None, w_rot=0.5):
    """Damped-least-squares IK with joint limits and a null-space posture term. Returns (q, ok, pos_err, rot_err)."""
    rng = np.random.default_rng(0) if rng is None else rng
    best = None
    for attempt in range(restarts + 1):
        q = np.array(q_seed, float) if attempt == 0 else np.clip(q_seed + rng.normal(0, 0.5, 7), Q_MIN, Q_MAX)
        for _ in range(iters):
            T = fk(q); e = pose_err(T_goal, T); ep, er = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
            if ep < tol_pos and er < tol_rot: break
            J = jacobian(q); W = np.diag([1, 1, 1, w_rot, w_rot, w_rot])
            Jw = W @ J; dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + damping ** 2 * np.eye(6), W @ e)
            N = np.eye(7) - np.linalg.pinv(J) @ J
            dq += N @ (null_gain * (q_rest - q))
            step = np.max(np.abs(dq)); dq *= min(1.0, 0.3 / max(step, 1e-9))
            q = np.clip(q + dq, Q_MIN + 1e-3, Q_MAX - 1e-3)
        T = fk(q); e = pose_err(T_goal, T); ep, er = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
        cand = (ep + 0.05 * er, q, ep < tol_pos and er < tol_rot, ep, er)
        if best is None or cand[0] < best[0]: best = cand
        if cand[2]: break
    return best[1], best[2], best[3], best[4]


def min_jerk(n):
    s = np.linspace(0, 1, n); return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def _duration(dq, vel_scale):
    # min-jerk peak velocity = 1.875 x mean velocity
    return max(float(np.max(np.abs(dq) / (QD_MAX * vel_scale))) * 1.875, 0.2)


VEL_SCALE, LIN_SPEED = 0.3, 0.15          # defaults: fraction of the Panda's joint velocity limits; straight-line TCP speed (m/s)


def plan_joint(q0, T_goal, dt=0.01, vel_scale=None, **ik_kw):
    vel_scale = VEL_SCALE if vel_scale is None else vel_scale
    q1, ok, ep, er = ik(T_goal, q0, **ik_kw)
    if not ok: return None, dict(reason="ik", pos_err=ep, rot_err=er)
    n = int(np.ceil(_duration(q1 - q0, vel_scale) / dt)) + 1
    return q0 + min_jerk(n)[:, None] * (q1 - q0), dict(reason="", pos_err=ep, rot_err=er)


def slerp_R(R0, R1, s):
    w = rot_err(R1, R0); th = np.linalg.norm(w)
    if th < 1e-9: return R0
    k = w / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]); a = s * th
    return (np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx) @ R0


def plan_cartesian(q0, T_goal, dt=0.01, lin_speed=None, max_joint_jump=0.15, **ik_kw):
    lin_speed = LIN_SPEED if lin_speed is None else lin_speed
    """Straight TCP line with min-jerk timing; IK per waypoint (seeded), fails on large joint jumps (singularities)."""
    T0 = fk(q0); dist = np.linalg.norm(T_goal[:3, 3] - T0[:3, 3]); ang = np.linalg.norm(rot_err(T_goal[:3, :3], T0[:3, :3]))
    dur = max(1.875 * max(dist / lin_speed, ang / 1.0), 0.3); n = int(np.ceil(dur / dt)) + 1
    Q = [q0]; q = q0.copy(); worst = 0.0
    for s in min_jerk(n)[1:]:
        T = np.eye(4); T[:3, 3] = T0[:3, 3] + s * (T_goal[:3, 3] - T0[:3, 3]); T[:3, :3] = slerp_R(T0[:3, :3], T_goal[:3, :3], s)
        qn, ok, ep, er = ik(T, q, iters=50, restarts=0, null_gain=0.02, **ik_kw)
        if not ok:
            return None, dict(reason="ik_waypoint", pos_err=ep, rot_err=er, at=float(s))
        if np.max(np.abs(qn - q)) > max_joint_jump: return None, dict(reason="joint_jump", at=float(s))
        worst = max(worst, ep); Q.append(qn); q = qn
    return np.array(Q), dict(reason="", pos_err=worst, rot_err=0.0)
