"""Hierarchical VLM -> reach -> learned grasp -> transport controller for LIBERO-Object."""
import collections, json
import numpy as np, torch
from robosuite.utils import camera_utils as CU
from robosuite.utils.transform_utils import quat2axisangle, quat2mat, axisangle2quat, mat2quat, quat_multiply, quat_inverse
from vrg.common import *

META = json.load(open(os.path.join(ROOT, "data/libero_object_meta.json")))
_M = lambda k: np.array([m[k] for m in META])
HANDOFF_OFF = (_M("ee_ts") - _M("obj0")).mean(0)        # EEF - object at handoff, from demos
HANDOFF_ORI = _M("ori_ts").mean(0)                       # axis-angle EEF orientation at handoff
# demos release anywhere along the basket (y std 13cm), so use its centre; height from demos + margin
RELEASE_OFF = np.array([0.0, 0.0, (_M("ee_tr") - _M("basket_tr")).mean(0)[2] + 0.03])
SETTLE = 10
VLM_RES = 1024
CROP = (0.33, 0.82, 0.08, 0.98)
DUMMY = np.array([0, 0, 0, 0, 0, 0, -1.0])


def proprio(obs):
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]).astype(np.float32)


def shuffle_layout(env, seed, language):
    """OOD layout: randomly permute the floor (x, y) positions of all grocery items (basket stays).
    The target always moves to a slot previously occupied by a different object."""
    rng = np.random.default_rng(seed)
    names = [n for n in env.env.objects_dict if n != "basket_1"]
    tgt = [n for n in names if n.startswith(target_name(language))][0]
    while True:
        perm = rng.permutation(len(names))
        if names[perm[names.index(tgt)]] != tgt: break
    qadr = {n: env.sim.model.get_joint_qpos_addr(env.env.objects_dict[n].joints[0])[0] for n in names}
    xy = {n: env.sim.data.qpos[qadr[n]:qadr[n] + 2].copy() for n in names}
    for i, n in enumerate(names):
        env.sim.data.qpos[qadr[n]:qadr[n] + 2] = xy[names[perm[i]]]
    env.sim.forward()


def reset_episode(env, tid, ep, shuffle=False, perturb=None):
    env.reset()
    obs = env.set_init_state(get_suite().get_task_init_states(tid)[ep])
    if shuffle:
        shuffle_layout(env, 7919 * tid + ep, get_suite().get_task(tid).language)
    if perturb:                                  # appearance-only dev perturbations (vrg/perturb.py), seeded per episode
        from vrg.perturb import perturb_scene
        perturb_scene(env, perturb, 104729 * tid + ep + 1)
    for _ in range(SETTLE):
        obs, *_ = env.step(DUMMY)
    return obs


# ---------------------------------------------------------------- perception
def render_rgbd(env, cam="agentview", res=VLM_RES):
    rgb, depth = env.sim.render(width=res, height=res, camera_name=cam, depth=True)
    depth = CU.get_real_depth_map(env.sim, depth)
    return rgb[::-1].copy(), depth[::-1].copy(), CU.get_camera_intrinsic_matrix(env.sim, cam, res, res), CU.get_camera_extrinsic_matrix(env.sim, cam)


def unproject(depth, K, T, u, v, r=2):
    d = float(np.median(depth[int(v) - r:int(v) + r + 1, int(u) - r:int(u) + r + 1]))
    return (T @ np.array([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d, 1]))[:3]


def vlm_ground(grounder, img, name):
    """Crop to the workspace, ask for a bbox, return bbox centre in full-image pixels (or None)."""
    import re
    H, W = img.shape[:2]; y0, y1, x0, x1 = int(CROP[0] * H), int(CROP[1] * H), int(CROP[2] * W), int(CROP[3] * W)
    c = img[y0:y1, x0:x1]
    q = ('Locate the %s in the image and output its bbox coordinates in JSON format like '
         '[{"bbox_2d": [x1, y1, x2, y2], "label": "%s"}].' % (name, name))
    r = grounder.ask(c, q)
    m = re.search(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]", r)
    if not m: return None, None, r
    b = [float(x) for x in m.groups()]; h, w = c.shape[:2]
    box = [x0 + b[0] / 1000 * w, y0 + b[1] / 1000 * h, x0 + b[2] / 1000 * w, y0 + b[3] / 1000 * h]
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), box, r


# ---------------------------------------------------------------- low-level control
def ori_error(obs, target_aa):
    """Axis-angle rotation taking current EEF orientation to target (world frame)."""
    q_cur = obs["robot0_eef_quat"]; q_tgt = axisangle2quat(target_aa)
    q_err = quat_multiply(q_tgt, quat_inverse(q_cur))
    if q_err[3] < 0: q_err = -q_err
    return quat2axisangle(q_err)


def move_to(env, obs, pos, ori_aa=None, grip=-1.0, tol=0.01, max_steps=80, gain=1.0, frames=None, cb=None):
    for k in range(max_steps):
        e = pos - obs["robot0_eef_pos"]
        if np.linalg.norm(e) < tol: break
        a = np.zeros(7); a[:3] = np.clip(gain * e / 0.05, -1, 1)
        if ori_aa is not None: a[3:6] = np.clip(ori_error(obs, ori_aa) / 0.5, -0.5, 0.5)
        a[6] = grip
        obs, r, done, info = env.step(a)
        if cb: cb(obs)
        if done: return obs, True
    return obs, False


# ---------------------------------------------------------------- learned policy wrapper
class PolicyRunner:
    def __init__(self, ckpt, device="cuda:0"):
        from vrg.policy import DiffusionPolicy
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        stats = torch.load(os.path.join(os.path.dirname(ckpt), "stats.pt"))
        a = ck["args"]; self.cams = tuple(a["cams"].split(","))
        self.pol = DiffusionPolicy(stats, n_tasks={"full": 10, "saycan": 11}.get(a["mode"], 0), cams=self.cams,
                                   img_norm=a.get("img_norm", False)).to(device)
        self.mode = a["mode"]; self.path_uv = None
        self.pol.load_state_dict(ck["ema"]); self.pol.eval(); self.dev = device
        self.mask_xy = a.get("mask_xy", False)
        self.reset()

    def reset(self):
        self.hist = collections.deque(maxlen=self.pol.n_obs); self.queue = collections.deque()

    def __call__(self, obs, tid=0):
        pr = proprio(obs)
        if self.mask_xy: pr[:2] = 0
        ag = obs["agentview_image"]
        if self.mode == "hamster":
            from vrg.paths import draw_path
            ag = draw_path(ag, self.path_uv)
        f = {"agent": ag, "wrist": obs["robot0_eye_in_hand_image"], "proprio": pr}
        self.hist.append(f)
        while len(self.hist) < self.pol.n_obs: self.hist.appendleft(f)
        if not self.queue:
            b = {k: torch.from_numpy(np.stack([h[k] for h in self.hist]))[None].to(self.dev) for k in ("agent", "wrist", "proprio")}
            b["tid"] = torch.tensor([tid], device=self.dev)
            self.queue.extend(self.pol.act(b)[0].float().cpu().numpy())
        return self.queue.popleft()


TRANSPORT_Z = 0.30
RETRACT_X = -0.05  # pull back toward the base before travelling: avoids the stretched-arm singularity (joint 4 at limit)


def transport_and_release(env, obs, p_bk, cb=None, retract=True, release_steps=20, hold_ori=True):
    """Scripted lift -> (retract) -> over basket -> descend -> open. Returns (obs, done)."""
    tgt = p_bk + RELEASE_OFF
    ORI = HANDOFF_ORI if hold_ori else None
    up = obs["robot0_eef_pos"].copy(); up[2] = max(up[2], TRANSPORT_Z)
    obs, d = move_to(env, obs, up, ORI, grip=1.0, max_steps=30, tol=0.02, cb=cb); done = d
    if retract == "auto":  # posture-aware: only when the elbow is close to its straight-arm limit
        retract = obs["robot0_joint_pos"][3] > -0.4
    if retract and up[0] > RETRACT_X:
        wp = up.copy(); wp[0] = RETRACT_X; wp[1] = 0.5 * (up[1] + tgt[1])
        obs, d = move_to(env, obs, wp, ORI, grip=1.0, max_steps=60, tol=0.03, cb=cb); done |= d
    over = tgt.copy(); over[2] = TRANSPORT_Z
    obs, d = move_to(env, obs, over, ORI, grip=1.0, max_steps=60, tol=0.015, cb=cb); done |= d
    obs, d = move_to(env, obs, tgt, ORI, grip=1.0, max_steps=40, tol=0.015, cb=cb); done |= d
    for _ in range(release_steps):
        if done: break
        obs, r, done, _ = env.step(DUMMY)
        if cb: cb(obs)
    return obs, done


def handoff_for(pre):
    """Handoff pose for a skill trained from `pre` steps before the grasp: mean (EEF - object) offset and orientation."""
    Z = np.load(os.path.join(ROOT, "data/libero_object.npz"))
    s = lambda m: m["start"] + max(m["tg"] - pre, 0)
    off = np.mean([Z["proprio"][s(m), :3] - np.array(m["obj0"]) for m in META], 0)
    ori = np.mean([Z["proprio"][s(m), 3:6] for m in META], 0)
    return off, ori
