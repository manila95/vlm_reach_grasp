"""Keyframe diffusion policies on Colosseum training demos (vlmrg env; data from col_build_dataset).

Action = next keyframe(s): position (3), rotation 6-D (6), gripper_open (1), ignore_collisions (1)  -> 11-D, horizon 4.
Every frame t is a sample whose target is the next 4 keyframes after t (denser than keyframe-only training).

modes
  ego      ours: wrist RGB + [gripper_open, goal vector in gripper frame (3), gripper z]; targets in the GRIPPER frame;
           frames from k0 - W onward (the skill takes over at the VLM handoff, just before first contact)
  e2e      front + wrist RGB + gripper pose/open + instruction (CLIP text); targets in the WORLD frame; all frames
  hamster  e2e with the demo's keyframe path drawn on the front image; no text
"""
import argparse, copy, json, os, pickle, time
import numpy as np, torch
from scipy.spatial.transform import Rotation as Rot
from vrg.policy import DiffusionPolicy

ROOT = "/scratch/kaustubh/vlm_reach_grasp/colosseum"
TASKS = ["basketball_in_hoop", "close_box", "close_laptop_lid", "get_ice_from_fridge", "meat_on_grill", "move_hanger",
         "open_drawer", "reach_and_drag", "put_money_in_safe", "place_wine_at_rack_location", "turn_oven_on", "scoop_with_spatula"]
W_PRE, H = 10, 4


def rot6(q):  # quat xyzw (N,4) -> first two columns of R (N,6)
    R = Rot.from_quat(q).as_matrix(); return np.concatenate([R[:, :, 0], R[:, :, 1]], 1)


def from6(r6):  # (N,6) -> quat xyzw, Gram-Schmidt
    a, b = r6[:, :3], r6[:, 3:]
    x = a / np.linalg.norm(a, axis=1, keepdims=True); b = b - (x * b).sum(1, keepdims=True) * x
    y = b / np.linalg.norm(b, axis=1, keepdims=True); z = np.cross(x, y)
    return Rot.from_matrix(np.stack([x, y, z], 2)).as_quat()


def to_gripper_frame(cur, tgt):
    """cur (7,) pose; tgt (N,7) poses -> (N,9) target pos/rot expressed in the current gripper frame."""
    Rc = Rot.from_quat(cur[3:]); p = Rc.inv().apply(tgt[:, :3] - cur[:3]); R = (Rc.inv() * Rot.from_quat(tgt[:, 3:])).as_quat()
    return np.concatenate([p, rot6(R)], 1)


def project_rlbench(P, K, T):
    """World points (N,3) -> pixel (u, v) for an RLBench/PyRep camera (inverse of pointcloud_from_depth_and_camera_params)."""
    x = (K @ (T[:3, :3].T @ (np.asarray(P).T - T[:3, 3:4]))).T
    return x[:, :2] / x[:, 2:3]


def ego_proprio(pose, gopen, goal):
    g = Rot.from_quat(pose[3:]).inv().apply(goal - pose[:3])
    return np.array([gopen, *g, pose[2]], np.float32)


def build(tasks, mode, text_emb=None):
    from vrg.paths import project, draw_path
    S = {k: [] for k in ["front", "wrist", "proprio", "action", "text"]}
    for task in tasks:
        Zf = np.load(f"{ROOT}/train_npz/{task}.npz")  # read every array once: each Zf[k] re-reads the whole array
        Z = {k: Zf[k] for k in Zf.files if not (mode == "ego" and k == "front")}; meta = json.load(open(f"{ROOT}/train_npz/{task}_meta.json"))
        for m in meta:
            s, T, kps = m["start"], m["T"], m["kps"]; pose, go, ig = Z["pose"][s:s + T], Z["open"][s:s + T], Z["ignore"][s:s + T]
            goal = np.array(m["pose_goal"][:3]); t_lo = max(m["k0"] - W_PRE, 0) if mode == "ego" else 0
            if mode == "hamster":
                K, Tf = np.array(m["front_K"]), np.array(m["front_T"])
                uv = project_rlbench(pose[[0, m["k0"], m["kgoal"]], :3], K, Tf)   # same 3 points the VLM provides at test time
            for t in range(t_lo, T - 1):
                nxt = [k for k in kps if k > t and (mode != "ego" or k >= m["k0"])][:H]
                if not nxt: break
                nxt = nxt + [nxt[-1]] * (H - len(nxt)); tg = pose[nxt]
                if mode == "ego":
                    a = np.concatenate([to_gripper_frame(pose[t], tg), go[nxt, None], ig[nxt, None]], 1)
                    S["proprio"].append(ego_proprio(pose[t], go[t], goal))
                else:
                    a = np.concatenate([tg[:, :3], rot6(tg[:, 3:]), go[nxt, None], ig[nxt, None]], 1)
                    S["proprio"].append(np.concatenate([pose[t, :3], rot6(pose[t:t + 1, 3:])[0], [go[t]]]).astype(np.float32))
                S["action"].append(a.astype(np.float32)); S["wrist"].append(Z["wrist"][s + t])
                if mode != "ego":
                    f = Z["front"][s + t]
                    if mode == "hamster": f = draw_hamster(f, uv)
                    S["front"].append(f)
                if mode == "e2e": S["text"].append(text_emb[m["desc"]])
        print(task, "samples so far", len(S["action"]), flush=True)
    return {k: np.stack(v) for k, v in S.items() if v}


def draw_hamster(img, uv):
    """Draw a keyframe path (start, keyframes...) on an upright image: gradient polyline, green dots at keyframes."""
    from PIL import Image, ImageDraw
    sc = 4; H_ = img.shape[0]; im = Image.fromarray(img).resize((H_ * sc, H_ * sc)); d = ImageDraw.Draw(im); p = [tuple(x * sc) for x in uv]
    for i in range(len(p) - 1):
        t = i / max(len(p) - 2, 1); d.line([p[i], p[i + 1]], fill=(int(40 + 215 * t), int(60 + 120 * (1 - t)), int(255 - 155 * t)), width=2 * sc)
    for x, y in p[1:]: d.ellipse([x - 2 * sc, y - 2 * sc, x + 2 * sc, y + 2 * sc], fill=(0, 230, 0))
    return np.array(im.resize((H_, H_)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True, choices=["ego", "e2e", "hamster"])
    ap.add_argument("--tasks", default=",".join(TASKS)); ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0")
    A = ap.parse_args(); os.makedirs(A.out, exist_ok=True); json.dump(vars(A), open(f"{A.out}/args.json", "w"))
    dev = torch.device(A.device); torch.manual_seed(0)
    TE = pickle.load(open(f"{ROOT}/text_emb.pkl", "rb")) if A.mode == "e2e" else None
    D = build(A.tasks.split(","), A.mode, TE)
    cams = ("wrist",) if A.mode == "ego" else ("front", "wrist")
    G = {k: torch.from_numpy(v).to(dev) for k, v in D.items()}; del D
    a = G["action"].reshape(-1, G["action"].shape[-1])
    stats = {"action_min": a.min(0).values.cpu(), "action_max": a.max(0).values.cpu(),
             "proprio_min": G["proprio"].min(0).values.cpu(), "proprio_max": G["proprio"].max(0).values.cpu()}
    torch.save(stats, f"{A.out}/stats.pt")
    adim, pdim = G["action"].shape[-1], G["proprio"].shape[-1]
    pol = DiffusionPolicy(stats, n_obs=1, horizon=H, n_act=1, adim=adim, pdim=pdim, cams=cams, text_dim=512 if A.mode == "e2e" else 0).to(dev)
    ema = copy.deepcopy(pol).eval(); [p.requires_grad_(False) for p in ema.parameters()]
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-4, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 500) * 0.5 * (1 + np.cos(np.pi * s / A.steps)))
    N = len(G["action"]); print(A.mode, "samples", N, flush=True); t0 = time.time(); log = open(f"{A.out}/log.txt", "a")
    start = 1; SP = f"{A.out}/state.pt"
    if os.path.exists(SP):  # resume an interrupted run
        st = torch.load(SP, map_location=dev); pol.load_state_dict(st["pol"]); ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["step"] + 1
        torch.set_rng_state(st["rng"]); print("resumed at step", start, flush=True)
    t0 = time.time()
    for step in range(start, A.steps + 1):
        j = torch.randint(0, N, (A.bs,), device=dev)
        obs = {c: G[c][j][:, None] for c in cams}; obs["proprio"] = G["proprio"][j][:, None]
        if A.mode == "e2e": obs["text"] = G["text"][j]
        with torch.autocast("cuda", dtype=torch.bfloat16): loss = pol.loss(obs, G["action"][j])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step(); sched.step()
        with torch.no_grad():
            d = min(0.9999, (1 + step) / (10 + step))
            for pe, p in zip(ema.parameters(), pol.parameters()): pe.lerp_(p, 1 - d)
        if step % 500 == 0:
            msg = f"step {step} loss {loss.item():.4f} {(time.time() - t0) / (step - start + 1) * 1000:.0f}ms/it"; print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if step % 5000 == 0:
            torch.save({"pol": pol.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "rng": torch.get_rng_state()}, SP + ".tmp"); os.replace(SP + ".tmp", SP)
        if step % 20000 == 0 or step == A.steps:
            torch.save({"ema": ema.state_dict(), "args": vars(A), "adim": adim, "pdim": pdim, "cams": cams, "text": A.mode == "e2e"}, f"{A.out}/ckpt.pt")
