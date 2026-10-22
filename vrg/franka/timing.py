"""Where does an episode's time go? Instruments a backend and the method's modules without changing them (run_isaac --timing).

Two clocks, kept separate:
  robot    simulated seconds of commanded motion: trajectories (and the settle wait after each), gripper actions. On the
           real arm the same commands take about the same time, since the trajectories are timed by our planner.
  compute  wall seconds of computation on this machine: perception (proposals), IK planning, VLM calls (server time).
Camera capture is logged as compute too, but in sim it is rendering time; on the real arm it is ~1 frame (33 ms).
The sim teleports the arm home at episode start, so the return-home move is planned (not executed) and its duration added.
"""
import time
from contextlib import contextmanager
import numpy as np


class Recorder:
    def __init__(self): self.reset()

    def reset(self): self.items = []; self.order = []

    def add(self, cat, kind, sec, **kw): self.items.append(dict(cat=cat, kind=kind, sec=float(sec), **kw))

    def summary(self):
        out = {}
        for it in self.items: k = f"{it['kind']}:{it['cat']}"; out[k] = out.get(k, 0.0) + it["sec"]
        out["robot_total"] = sum(v for k, v in out.items() if k.startswith("robot:")); out["compute_total"] = sum(v for k, v in out.items() if k.startswith("compute:"))
        out["n_moves"] = sum(1 for it in self.items if it["cat"] == "trajectory"); return out


def instrument(env, vlm=None):
    """Wrap env motions/captures, kinematics planners, perception and the VLM client with timers. Returns the Recorder."""
    from vrg.franka import kinematics as K, episode, reactive, perception
    from vrg.franka.isaac_backend import SIM_DT
    rec = Recorder()

    def wall(cat, fn):
        def w(*a, **k):
            t0 = time.time(); r = fn(*a, **k); rec.add(cat, "compute", time.time() - t0); return r
        return w
    K.plan_joint = wall("ik_planning", K.plan_joint); K.plan_cartesian = wall("ik_planning", K.plan_cartesian)
    for m in (episode, reactive): m.proposals = wall("perception", perception.proposals)
    f0 = env.follow_joint_trajectory

    def follow(Q, dt=0.01, **k):
        r = f0(Q, dt=dt, **k)
        rec.add("trajectory", "robot", len(Q) * dt, n=len(Q)); rec.add("settle_after_move", "robot", r.info.get("settle_s", env.settle_s)); return r
    env.follow_joint_trajectory = follow
    g0, g1 = env.gripper_move, env.gripper_grasp
    env.gripper_move = lambda w: (g0(w), rec.add("gripper", "robot", 0.6))[0]
    env.gripper_grasp = lambda *a, **k: (lambda r: (rec.add("gripper", "robot", 0.8), r)[1])(g1(*a, **k))
    env.frames = wall("camera_capture(sim render)", env.frames)
    if vlm is not None:
        for name in ("ground", "verify"):
            fn = getattr(vlm, name)
            def w(*a, _fn=fn, **k):
                r = _fn(*a, **k); rec.add("vlm", "compute", r.get("sec", 0.0)); return r
            setattr(vlm, name, w)
    return rec


def return_home_time(q_end, vel_scale=None, dt=0.01, settle=0.3):
    """Duration of the min-jerk joint move back home (the sim teleports instead)."""
    from vrg.franka import kinematics as K
    return K._duration(np.asarray(K.Q_HOME) - np.asarray(q_end), K.VEL_SCALE if vel_scale is None else vel_scale) + settle
