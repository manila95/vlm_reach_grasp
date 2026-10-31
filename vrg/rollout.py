"""Stage 2: closed-loop evaluation.

methods:
  hier_vlm      VLM grounding -> scripted reach to handoff -> learned local grasp -> scripted transport/release
  --oracle      any hier_* method with ground-truth object/basket positions (isolates grounding errors)
  hier_scripted VLM grounding -> scripted reach -> scripted top-down grasp (no learning) -> transport
  full          end-to-end task-conditioned diffusion policy (baseline)
"""
import argparse, json, os, pickle, time
import multiprocessing as mp
import numpy as np

MAX_STEPS = 300  # LIBERO-Object standard budget is 280 after 10 settle steps
GRASP_MAX, LIFT_DONE = 120, 0.08
TRANSPORT_Z = 0.30
SCRIPTED_GRASP_DZ = float(os.environ.get('SCRIPTED_GRASP_DZ', 0.0))


def calib_bias(G):
    """Mean (VLM 3D point - GT body position) over held-out calibration scenes with correct identification."""
    o = [r["obj"] - r["gt_obj"] for r in G.values() if r["calib"] and r["obj_hit"]]
    b = [r["basket"] - r["gt_basket"] for r in G.values() if r["calib"] and r["basket"] is not None]
    return np.mean(o, 0), np.mean(b, 0)


def run_task(args):
    tid, eps, A = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch, imageio
    from vrg.common import make_env, target_name
    from vrg.pipeline import (reset_episode, move_to, PolicyRunner, HANDOFF_OFF, HANDOFF_ORI, RELEASE_OFF, DUMMY, transport_and_release, ori_error)
    dev = f"cuda:{tid % torch.cuda.device_count()}" if A.device == "auto" else A.device
    G = pickle.load(open(A.grounding, "rb")) if A.method not in ("full", "saycan") else None
    if A.method == "saycan":
        PLANS = pickle.load(open(f"data/saycan_plans{'_shuf' if A.shuffle else ''}.pkl", "rb"))["plans"]
    if A.method == "hamster":
        import json as _j
        from vrg.paths import keypoint_offsets, path_from_estimates
        KP_OFF = keypoint_offsets(_j.load(open("data/libero_object_meta.json")), np.load("data/libero_object.npz"))
        cm = np.load("data/agentview_cam.npy"); CAM = (cm[0][:3, :3], cm[1])
    if G and not A.oracle: ob_bias, bk_bias = calib_bias(pickle.load(open(A.calib, "rb")) if A.calib else G)
    runner = PolicyRunner(A.ckpt, dev) if A.method != "hier_scripted" else None
    if A.pre:
        from vrg.pipeline import handoff_for
        HO_PRE, HORI_PRE = handoff_for(A.pre)
    env, task = make_env(tid)
    out = []
    for ep in eps:
        frames, n = [], [0]
        obs = reset_episode(env, tid, ep, A.shuffle, getattr(A, "perturb", None))
        pos0 = {k[:-4]: obs[k].copy() for k in obs if k.endswith("_pos") and "_to_" not in k and not k.startswith("robot") and not k.startswith("basket")}

        PH = [""]; path3d = [None]; hi_frames, hi_phase = [], []

        def rec(o):
            n[0] += 1
            if A.video and len(out) < A.video:
                frames.append(np.concatenate([o["agentview_image"][::-1], o["robot0_eye_in_hand_image"][::-1]], 1))
            if A.hires:  # presentation video: high-res agentview + wrist inset (+ HAMSTER path), phase logged per frame
                H = A.hires; f = env.sim.render(width=H, height=H, camera_name="agentview")[::-1].copy()
                if path3d[0] is not None:
                    from vrg.paths import camera_mats, project, draw_path
                    uv = project(path3d[0], *camera_mats(env, res=H)); f = draw_path(f[::-1], uv, scale=1)[::-1].copy()
                w = env.sim.render(width=H // 3, height=H // 3, camera_name="robot0_eye_in_hand")[::-1]
                f[-H // 3 - 6:-6, -H // 3 - 6:-6] = w
                hi_frames.append(f); hi_phase.append(PH[0])

        def step(a):
            o, r, d, i = env.step(a); rec(o); return o, d

        done, phase_fail = False, None
        t0 = time.time()
        if A.method == "saycan":
            # step 1: planner's chosen skill (LM x affordance, cached); step 2: re-plan online with "holding" affordance
            pl = PLANS[(tid, ep)]; skill = pl["first"]; runner.reset(); z_close = None; sk_steps = 0; plan_log = [skill]
            from vrg.saycan_plan import SKILLS as _SK
            while n[0] < MAX_STEPS and not done and skill != 11:  # 11 = "done" (planner may terminate immediately)
                PH[0] = f'planner chose: "{_SK[skill]}"'
                a = runner(obs, skill); a[6] = 1.0 if a[6] > 0 else -1.0
                obs, done = step(a); sk_steps += 1
                if skill < 10:  # pick skill terminates when an object has been lifted (proprio) or it times out
                    z = obs["robot0_eef_pos"][2]
                    if a[6] > 0 and z_close is None: z_close = z
                    if a[6] < 0: z_close = None
                    lifted = z_close is not None and z > z_close + LIFT_DONE
                    if lifted or sk_steps > 200:
                        width = obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]
                        holding = lifted and width > 0.01
                        lm2 = pl["lm2"] if pl["lm2"] is not None else np.ones(12) / 12
                        aff2 = np.zeros(12); aff2[10] = 1.0 if holding else 0.0; aff2[11] = 0.1; aff2[skill] = 0.0 if holding else 1.0
                        skill = int(np.argmax(lm2 * aff2)); plan_log.append(skill); runner.reset(); sk_steps = 0
                        if skill == 11: break
            phase_fail = None if done else ("wrong_pick" if pl["first"] != tid else "execution")
        elif A.method == "hamster":
            g = G[(tid, ep)]
            if g["obj"] is None or g["basket"] is None:
                out.append(dict(tid=tid, ep=ep, success=False, fail="grounding_parse", steps=0)); continue
            p_obj, p_bk = (g["gt_obj"], g["gt_basket"]) if A.oracle else (g["obj"] - ob_bias, g["basket"] - bk_bias)
            runner.reset(); runner.path_uv = path_from_estimates(obs["robot0_eef_pos"], p_obj, p_bk, KP_OFF, *CAM)
            path3d[0] = np.stack([obs["robot0_eef_pos"], p_obj + KP_OFF["pre"], p_obj + KP_OFF["grasp"], p_obj + KP_OFF["lift"], p_bk + KP_OFF["release"]])
            PH[0] = "policy conditioned on VLM path (drawn)"
            while n[0] < MAX_STEPS and not done:
                a = runner(obs); a[6] = 1.0 if a[6] > 0 else -1.0
                obs, done = step(a)
        elif A.method == "full":
            runner.reset()
            while n[0] < MAX_STEPS and not done:
                a = runner(obs, tid); a[6] = 1.0 if a[6] > 0 else -1.0
                obs, done = step(a)
        else:
            g = G[(tid, ep)]
            if A.oracle: p_obj, p_bk = g["gt_obj"], g["gt_basket"]
            else:
                if g["obj"] is None or g["basket"] is None:
                    out.append(dict(tid=tid, ep=ep, success=False, fail="grounding_parse", steps=0)); continue
                p_obj, p_bk = g["obj"] - ob_bias, g["basket"] - bk_bias
            if A.noise > 0:  # simulated localisation error (deterministic per episode)
                p_obj = p_obj + np.r_[np.random.default_rng(1000 * tid + ep).normal(0, A.noise, 2), 0]
            PH[0] = "[1] VLM-guided reach to handoff (scripted)"
            # (1) scripted reach to the handoff pose the demos pass through ~1s before grasping
            ho = (HO_PRE if A.pre else HANDOFF_OFF) + np.array([A.handoff_dx, 0, A.handoff_dz])
            obs, done = move_to(env, obs, p_obj + ho, HORI_PRE if A.pre else HANDOFF_ORI, max_steps=100, cb=rec)
            # (2) grasp + lift
            z_close, holding = None, False
            PH[0] = "[2] scripted top-down grasp" if A.method == "hier_scripted" else "[2] learned egocentric grasp (diffusion, wrist cam)"
            if A.method == "hier_scripted":
                obs, _ = move_to(env, obs, p_obj + np.array([0, 0, SCRIPTED_GRASP_DZ]), HANDOFF_ORI, max_steps=60, tol=0.008, gain=0.6, cb=rec)
                for _ in range(15): obs, done = step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
                z_close = obs["robot0_eef_pos"][2]
                obs, _ = move_to(env, obs, obs["robot0_eef_pos"] + [0, 0, LIFT_DONE + 0.02], grip=1.0, max_steps=40, cb=rec)
            else:
                for attempt in range(1 + A.retries):
                    if attempt > 0:  # retry: open, back off to the handoff pose, run the skill again
                        obs, _ = move_to(env, obs, obs["robot0_eef_pos"] + [0, 0, 0.05], HANDOFF_ORI, grip=-1.0, max_steps=20, tol=0.02, cb=rec)
                        obs, done = move_to(env, obs, p_obj + HANDOFF_OFF, HANDOFF_ORI, max_steps=60, cb=rec)
                    runner.reset(); z_close = None
                    for _ in range(GRASP_MAX):
                        a = runner(obs); a[6] = 1.0 if a[6] > 0 else -1.0
                        if A.lock_ori:  # policy controls translation + gripper; orientation held at the handoff pose
                            hold = np.clip(ori_error(obs, HANDOFF_ORI) / 0.5, -0.5, 0.5)
                            if A.lock_ori == "tilt": a[3:5] = hold[:2]   # lock roll/pitch only, yaw stays with the policy
                            elif A.lock_ori == "auto":                   # posture-aware: lock tilt only near the elbow limit
                                if obs["robot0_joint_pos"][3] > A.j4_thresh: a[3:5] = hold[:2]
                            else: a[3:6] = hold
                        obs, done = step(a)
                        z = obs["robot0_eef_pos"][2]
                        if a[6] > 0 and z_close is None: z_close = z
                        if a[6] < 0: z_close = None  # re-opened: wait for next close
                        if z_close is not None and z > z_close + LIFT_DONE: break
                    width = obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]
                    if (z_close is not None and width > 0.01) or n[0] > MAX_STEPS - 120: break
            # grip width tells us if something is held (fully closed fingers -> missed)
            width = obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]
            holding = z_close is not None and width > 0.01
            if not holding: phase_fail = "grasp"
            PH[0] = "[3] VLM-guided transport + release (scripted)"
            # (3) scripted transport + release above the VLM-localised basket
            obs, d = transport_and_release(env, obs, p_bk, cb=rec, retract=("auto" if A.auto_retract else not A.no_retract)); done = done or d
            if not done and phase_fail is None: phase_fail = "place"
        disp = {k: float(np.linalg.norm(env.env.sim.data.body_xpos[env.env.obj_body_id[k]] - v)) for k, v in pos0.items()}
        moved = max(disp, key=disp.get)  # analysis only (privileged): which object did the robot actually move?
        out.append(dict(tid=tid, ep=ep, success=bool(done), fail=None if done else phase_fail, steps=n[0], sec=time.time() - t0,
                        moved=moved if disp[moved] > 0.05 else None, target=target_name(getattr(task, "name", task.language)) + "_1"))
        print(f"[{A.method}] task {tid} ep {ep}: success={done} fail={out[-1]['fail']} steps={n[0]}", flush=True)
        if A.hires and hi_frames:
            os.makedirs(f"{A.out}/hires", exist_ok=True)
            imageio.mimsave(f"{A.out}/hires/t{tid}_e{ep}.mp4", hi_frames, fps=30)
            json.dump(dict(out[-1], phases=hi_phase), open(f"{A.out}/hires/t{tid}_e{ep}.json", "w"))
        if frames:
            os.makedirs(f"{A.out}/videos", exist_ok=True)
            imageio.mimsave(f"{A.out}/videos/t{tid}_e{ep}_{'ok' if done else 'fail'}.mp4", frames, fps=30)
    env.close()
    return out


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["hier_vlm", "hier_scripted", "full", "saycan", "hamster"])
    ap.add_argument("--shuffle", action="store_true", help="OOD: permute object positions")
    ap.add_argument("--perturb", default=None, choices=["light", "texture", "both"], help="appearance dev perturbation (vrg/perturb.py)")
    ap.add_argument("--calib", default=None, help="grounding file whose calibration scenes define the bias (default: --grounding)")
    ap.add_argument("--hires", type=int, default=0, help="also save presentation videos at this resolution")
    ap.add_argument("--eps", default="", help="comma-separated episode indices (overrides the eval list)")
    ap.add_argument("--holdout", action="store_true", help="evaluate on the 25 held-out init states per task")
    ap.add_argument("--pre", type=int, default=None, help="skill trained with --pre: use the matching handoff pose")
    ap.add_argument("--handoff_dz", type=float, default=0.0, help="raise the handoff pose (m) above the demo-derived one")
    ap.add_argument("--handoff_dx", type=float, default=0.0, help="shift the handoff pose (m) in x")
    ap.add_argument("--retries", type=int, default=0, help="grasp retries when the gripper check says nothing is held")
    ap.add_argument("--lock_ori", nargs="?", const="all", default=None, choices=["all", "tilt", "auto"], help="hold handoff orientation during the learned grasp (all axes, or roll/pitch only)")
    ap.add_argument("--j4_thresh", type=float, default=-0.6, help="lock_ori auto: lock tilt when joint 4 exceeds this")
    ap.add_argument("--auto_retract", action="store_true", help="retract only when joint 4 is near its limit")
    ap.add_argument("--no_retract", action="store_true", help="v1 transport (no retract waypoint)")
    ap.add_argument("--noise", type=float, default=0.0, help="xy std (m) added to the object estimate")
    ap.add_argument("--oracle", action="store_true", help="use ground-truth positions instead of VLM grounding")
    ap.add_argument("--ckpt", default=None); ap.add_argument("--grounding", default="data/grounding.pkl")
    ap.add_argument("--n_eps", type=int, default=20); ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--video", type=int, default=2, help="videos saved per task"); ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=10)
    return ap


if __name__ == "__main__":
    A = build_parser().parse_args()
    from vrg.ground_episodes import EVAL_EPS, HOLDOUT_EPS
    if A.holdout: EVAL_EPS = HOLDOUT_EPS
    if A.eps: EVAL_EPS = [int(e) for e in A.eps.split(",")]
    lo, hi = map(int, A.tasks.split("-"))
    os.makedirs(A.out, exist_ok=True)
    jobs = [(t, EVAL_EPS[:A.n_eps], A) for t in range(lo, hi + 1)]
    with mp.get_context("spawn").Pool(min(A.workers, len(jobs))) as pool:
        res = [r for rs in pool.map(run_task, jobs) for r in rs]
    json.dump(res, open(f"{A.out}/results.json", "w"), indent=1)
    sr = np.mean([r["success"] for r in res])
    per = {t: np.mean([r["success"] for r in res if r["tid"] == t]) for t in range(lo, hi + 1)}
    fails = {}
    for r in res:
        if r["fail"]: fails[r["fail"]] = fails.get(r["fail"], 0) + 1
    print(f"== {A.method} success {sr:.3f} ({len(res)} eps) per-task {per} fails {fails} mean steps {np.mean([r['steps'] for r in res]):.0f}")
    json.dump(dict(method=A.method, success=sr, per_task=per, fails=fails), open(f"{A.out}/summary.json", "w"), indent=1)
