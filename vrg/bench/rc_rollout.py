"""Closed-loop evaluation on RoboCasa (robocasa env). Grounding/text embeddings are precomputed (vlmrg env).

methods: ours | e2e | saycan | hamster
python -m vrg.bench.rc_rollout --method ours --split target --eps 0-49 --out runs/rc_eval/ours_target
"""
import robocasa.utils.object_utils as OU
import argparse, collections, json, os, pickle, time
import numpy as np, torch
from PIL import Image
from robosuite.utils.transform_utils import quat2axisangle, quat_multiply, quat_inverse, quat2mat, mat2quat
from vrg.bench.robocasa_env import TASKS, make, episode_seed, raw, raw_obs, base_pose
from vrg.bench.rc_train import ego_proprio, skill_texts, place_clause, ROOT
from vrg.policy import DiffusionPolicy

LIFT_DONE, GRASP_MAX, PLACE_MAX = 0.06, 150, 120


def img128(o, cam):
    return np.array(Image.fromarray(o[f"{cam}_image"][::-1]).resize((128, 128), Image.BILINEAR))


def state16(o):
    return np.concatenate([o["robot0_base_pos"], o["robot0_base_quat"], o["robot0_base_to_eef_pos"],
                           o["robot0_base_to_eef_quat"], o["robot0_gripper_qpos"]]).astype(np.float32)


class Runner:
    def __init__(self, ckpt, device):
        ck = torch.load(ckpt, map_location=device, weights_only=False); st = torch.load(os.path.join(os.path.dirname(ckpt), "stats.pt"))
        self.cams, self.mode, self.text = ck["cams"], ck["args"]["mode"], ck["text"]
        self.pol = DiffusionPolicy(st, adim=ck["adim"], pdim=ck["pdim"], cams=self.cams, text_dim=512 if self.text else 0).to(device)
        self.pol.load_state_dict(ck["ema"]); self.pol.eval(); self.dev = device; self.path_uv = None; self.emb = None; self.reset()

    def reset(self): self.hist = collections.deque(maxlen=2); self.q = collections.deque()

    def __call__(self, o):
        from vrg.paths import draw_path
        s16 = state16(o)[None]
        f = {"proprio": ego_proprio(s16)[0] if self.mode.startswith("local") else s16[0]}
        names = {"left": "robot0_agentview_left", "right": "robot0_agentview_right", "wrist": "robot0_eye_in_hand"}
        for c in self.cams:
            im = img128(o, names[c])
            if c == "left" and self.path_uv is not None: im = draw_path(im[::-1], self.path_uv)[::-1]
            f[c] = im
        self.hist.append(f)
        while len(self.hist) < 2: self.hist.appendleft(f)
        if not self.q:
            b = {k: torch.from_numpy(np.stack([h[k] for h in self.hist]))[None].to(self.dev) for k in f}
            if self.text: b["text"] = torch.from_numpy(self.emb)[None].to(self.dev)
            self.q.extend(self.pol.act(b)[0].float().cpu().numpy())
        return self.q.popleft()


def arm_action(o, target_w, quat_b=None, grip=-1.0, gain=1.0):
    bp, R = base_pose(o); e_b = R.T @ (target_w - o["robot0_eef_pos"])
    a = np.zeros(12); a[:3] = np.clip(gain * e_b / 0.05, -1, 1); a[6] = grip; a[11] = -1
    if quat_b is not None:
        qe = quat_multiply(quat_b, quat_inverse(o["robot0_base_to_eef_quat"]))
        if qe[3] < 0: qe = -qe
        a[3:6] = np.clip(quat2axisangle(qe) / 0.5, -0.5, 0.5)
    return a, float(np.linalg.norm(e_b))


def full_action(a7):
    a = np.zeros(12); a[:7] = a7; a[6] = 1.0 if a7[6] > 0 else -1.0; a[11] = -1; return a


def place_target(g, o, S):
    """VLM place point; for fixtures (cabinet/counter), fall back to the demo-derived robot-relative prior when the
    VLM point is missing or at an implausible height (e.g. the upper cabinet is outside the base cameras' view)."""
    bp, R = base_pose(o); prior = bp + R @ S["release_base"]
    p = g.get("place")
    if p is None: return prior, "prior"
    if not g.get("place_is_object", False) and abs(p[2] - prior[2]) > 0.12: return prior, "prior"
    return p, "vlm"


def run_episode(A, task, split, ep, env, genv, runners, G, TXT, STATS, rec):
    genv.reset(seed=episode_seed(split, ep)); o = raw_obs(env)
    from robocasa.utils.dataset_registry import get_ds_meta
    H = get_ds_meta(task, split)["horizon"]; n = [0]; done = [False]; PH = [""]
    pos0 = {k: env.sim.data.body_xpos[env.obj_body_id[k]].copy() for k in env.objects}

    def step(a):
        nonlocal o
        o, r, d, i = env.step(a); n[0] += 1; rec(env, o, PH[0])
        if env._check_success(): done[0] = True
        return o

    def move_to(target, quat_b=None, grip=-1.0, tol=0.012, max_steps=120, gain=1.0):
        for _ in range(max_steps):
            if done[0] or n[0] >= H: return
            a, err = arm_action(o, target, quat_b, grip, gain)
            if err < tol: return
            step(a)

    def align_base(target, max_steps=60, reach=0.6):
        """Slide the base sideways along the counter until the target is roughly straight ahead (base frame y ~ 0).
        Only when the target is beyond comfortable arm reach; the base never drives forward (the counter is there)."""
        for _ in range(max_steps):
            if done[0] or n[0] >= H: return
            bp, Rb = base_pose(o); rel = Rb.T @ (target - bp)
            if np.linalg.norm(rel[:2]) < reach and abs(rel[1]) < 0.35: return
            if abs(rel[1]) < 0.04: return
            a = np.zeros(12); a[8] = np.clip(rel[1] / 0.15, -1, 1); a[11] = 1.0; a[6] = grip_state[0]; step(a)
        for _ in range(8):  # let the arm settle after base motion
            a = np.zeros(12); a[11] = -1; a[6] = grip_state[0]; step(a)

    def reach_via(target, quat_b, grip=-1.0, clear=0.15):
        """Collision-avoiding scripted reach: rise to a clearance height, move over the target, descend."""
        up = o["robot0_eef_pos"].copy(); up[2] = max(up[2], target[2] + clear)
        move_to(up, None, grip=grip, max_steps=40, tol=0.03)
        over = target.copy(); over[2] = up[2]
        move_to(over, quat_b, grip=grip, max_steps=100, tol=0.02)
        move_to(target, quat_b, grip=grip, max_steps=60)

    grip_state = [-1.0]
    fail = None
    if A.method in ("e2e", "hamster", "saycan"):
        r = runners["main"]; r.reset(); lang = env.get_ep_meta()["lang"]
        if A.method == "e2e": r.emb = TXT[lang]
        if A.method == "hamster":
            from vrg.paths import project
            g = G[(split, task, ep)]; bp, R = base_pose(o); S = STATS[task]
            kp = np.stack([o["robot0_eef_pos"], g["pick"] + R @ S["pick_off"], g["pick"] + R @ S["grasp_off"],
                           g["pick"] + R @ S["lift_off"], place_target(g, o, S)[0] + R @ S["release_off"]])
            f = 128 / 2 / np.tan(np.radians(30)); K = np.array([[f, 0, 64], [0, f, 64], [0, 0, 1]])
            from robosuite.utils import camera_utils as CU
            r.path_uv = project(kp, K, CU.get_camera_extrinsic_matrix(env.sim, "robot0_agentview_left"))
        skill = "pick"
        if A.method == "saycan":
            pl = G[(split, task, ep)]["saycan"]; r.emb = TXT[pl["pick_text"]]; PH[0] = f'skill: "{pl["pick_text"]}"'
        z_close = None
        while n[0] < H and not done[0]:
            a = r(o); step(a if len(a) == 12 else full_action(a))
            if A.method == "saycan" and skill == "pick":
                z = o["robot0_base_to_eef_pos"][2]
                if a[6] > 0 and z_close is None: z_close = z
                if a[6] <= 0: z_close = None
                if z_close is not None and z > z_close + LIFT_DONE:
                    skill = "place"; r.reset(); r.emb = TXT[pl["place_text"]]; PH[0] = f'skill: "{pl["place_text"]}"'
    else:  # ours
        g = G[(split, task, ep)]; S = STATS[task]; bp, R = base_pose(o)
        PO, PQ = (S["pick_off40"], S["pick_quat40"]) if A.pick_pre == 40 else (S["pick_off"], S["pick_quat"])   # match the skill's training start
        place_pt, place_src = place_target(g, o, S)
        if g["pick"] is None: fail = "grounding"
        else:
            PH[0] = "[1] VLM-guided base alignment + reach (scripted)"
            align_base(g["pick"]); bp, R = base_pose(o)
            reach_via(g["pick"] + R @ PO, PQ)
            bp, R = base_pose(o); reach_err = float(np.linalg.norm(o["robot0_eef_pos"] - (g["pick"] + R @ PO)))
            PH[0] = "[2] learned egocentric pick"
            rp = runners["pick"]; z_close = None
            for attempt in range(1 + A.retries):
                rp.reset(); z_close = None
                for _ in range(GRASP_MAX):
                    if done[0] or n[0] >= H: break
                    a7 = rp(o); step(full_action(a7)); z = o["robot0_base_to_eef_pos"][2]
                    if a7[6] > 0 and z_close is None: z_close = z
                    if a7[6] <= 0: z_close = None
                    if z_close is not None and z > z_close + LIFT_DONE: break
                width = o["robot0_gripper_qpos"][0] - o["robot0_gripper_qpos"][1]
                if z_close is not None and width > 0.006: break
                move_to(o["robot0_eef_pos"] + np.array([0, 0, 0.06]), PQ, grip=-1.0, max_steps=20)
                bp, R = base_pose(o); move_to(g["pick"] + R @ PO, PQ, max_steps=60)
            width = o["robot0_gripper_qpos"][0] - o["robot0_gripper_qpos"][1]
            if not (z_close is not None and width > 0.006): fail = "grasp"
            held_pick = OU.check_obj_grasped(env, "obj")   # diagnostics only (failure labels), never used for control
            PH[0] = "[3] VLM-guided transport (scripted)"
            grip_state[0] = 1.0
            up = o["robot0_eef_pos"].copy(); up[2] += 0.08; move_to(up, None, grip=1.0, max_steps=30, tol=0.03)
            align_base(place_pt); bp, R = base_pose(o)
            ph = place_pt + R @ S["place_off"]
            up = o["robot0_eef_pos"].copy(); up[2] = max(up[2], ph[2]) + 0.05
            move_to(up, None, grip=1.0, max_steps=40, tol=0.03)
            over = ph.copy(); over[2] = up[2]
            move_to(over, S["place_quat"], grip=1.0, max_steps=100, tol=0.02)
            move_to(ph, S["place_quat"], grip=1.0, max_steps=60)
            held_place = OU.check_obj_grasped(env, "obj")
            PH[0] = "[4] learned egocentric place"
            rq = runners["place"]; rq.reset(); opened = 0
            for _ in range(PLACE_MAX):
                if done[0] or n[0] >= H: break
                a7 = rq(o); step(full_action(a7))
                opened = opened + 1 if a7[6] <= 0 else 0
                if opened >= 6: break
            PH[0] = "[5] release + retreat (scripted)"
            # finish opening in place before moving: retreating while the fingers still hold the object drags it out
            w_prev = None
            for k in range(20):
                if done[0] or n[0] >= H: break
                step(np.r_[np.zeros(6), -1, np.zeros(4), -1]); w = o["robot0_gripper_qpos"][0] - o["robot0_gripper_qpos"][1]
                if w_prev is not None and k >= 4 and abs(w - w_prev) < 5e-4: break
                w_prev = w
            up = o["robot0_eef_pos"] + np.array([0, 0, 0.10]); move_to(up, None, grip=-1.0, max_steps=30, tol=0.03)
            back = o["robot0_eef_pos"] + R @ np.array([-0.25, 0, 0.05])
            move_to(back, None, grip=-1.0, max_steps=60, tol=0.03)
            for _ in range(10):
                if done[0] or n[0] >= H: break
                step(np.r_[np.zeros(6), -1, np.zeros(4), -1])
            if not done[0]:   # diagnostic label: first phase that went wrong
                fail = ("reach" if reach_err > 0.05 and not held_pick else "grasp" if not held_pick
                        else "transport" if not held_place else "place")
    disp = {k: float(np.linalg.norm(env.sim.data.body_xpos[env.obj_body_id[k]] - v)) for k, v in pos0.items()}
    moved = max(disp, key=disp.get)
    return dict(task=task, split=split, ep=ep, success=bool(done[0]), fail=None if done[0] else (fail or "timeout"),
                steps=n[0], moved=moved if disp[moved] > 0.05 else None, target="obj", lang=env.get_ep_meta()["lang"],
                place_src=locals().get("place_src"), reach_err=locals().get("reach_err"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["ours", "e2e", "saycan", "hamster"])
    ap.add_argument("--split", required=True); ap.add_argument("--eps", default="0-49"); ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--ckpt", default=None); ap.add_argument("--pick_ckpt", default=None); ap.add_argument("--place_ckpt", default=None)
    ap.add_argument("--grounding", default=f"{ROOT}/grounding.pkl"); ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--pick_pre", type=int, default=20, choices=[20, 40], help="handoff pose: 20 or 40 steps before grasp (local_pick --pre)")
    ap.add_argument("--video", type=int, default=1); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0")
    A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
    lo, hi = map(int, A.eps.split("-")); dev = A.device
    G = pickle.load(open(A.grounding, "rb")) if A.method in ("ours", "hamster", "saycan") else {}
    TXT = pickle.load(open(f"{ROOT}/text_emb.pkl", "rb")) if A.method in ("e2e", "saycan") else {}
    STATS = json.load(open(f"{ROOT}/handoff_stats.json")); STATS = {t: {k: np.array(v) for k, v in d.items()} for t, d in STATS.items()}
    runners = {"pick": Runner(A.pick_ckpt, dev), "place": Runner(A.place_ckpt, dev)} if A.method == "ours" else {"main": Runner(A.ckpt, dev)}
    import imageio
    res = []
    for task in A.tasks.split(","):
        genv = make(task, A.split, episode_seed(A.split, lo)); env = raw(genv)
        for ep in range(lo, hi + 1):
            frames = []

            def rec(env_, o_, ph):
                if A.video and len([r for r in res if r["task"] == task]) < A.video and len(frames) < 1500:
                    frames.append(np.concatenate([o_["robot0_agentview_left_image"][::-1], o_["robot0_eye_in_hand_image"][::-1]], 1))
            t0 = time.time(); r = run_episode(A, task, A.split, ep, env, genv, runners, G, TXT, STATS, rec); r["sec"] = time.time() - t0
            res.append(r); print(f"[{A.method}/{A.split}] {task} ep{ep}: success={r['success']} fail={r['fail']} moved={r['moved']} steps={r['steps']} ({r['sec']:.0f}s)", flush=True)
            if frames:
                os.makedirs(f"{A.out}/videos", exist_ok=True)
                imageio.mimsave(f"{A.out}/videos/{task}_e{ep}_{'ok' if r['success'] else 'fail'}.mp4", frames, fps=20)
            json.dump(res, open(f"{A.out}/results.json", "w"), indent=1)
        genv.close()
    print(f"== {A.method}/{A.split} success {np.mean([r['success'] for r in res]):.3f} ({len(res)} eps)",
          {t: round(float(np.mean([r['success'] for r in res if r['task'] == t])), 3) for t in A.tasks.split(",")})


if __name__ == "__main__":
    main()
