"""Colosseum closed-loop evaluation (colosseum env, python 3.8). Policies run in policy_server.py (vlmrg env, GPU).

methods: ours | e2e | hamster        (actions = keyframe poses executed by EndEffectorPoseViaPlanning, max 25 keyframes)
$VGLRUN python vrg/bench/col_rollout.py --method ours --port 6001 --tasks close_box --factors 0,2 --n 10 --out runs/col_eval/ours
"""
import argparse, json, os, sys, time
from multiprocessing.connection import Client
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, "/scratch/kaustubh/vlm_reach_grasp/colosseum")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from colosseum_helpers import make_env, compat_pickle_load
from rlbench.backend.exceptions import InvalidActionError, TaskEnvironmentError
from pyrep.errors import IKError, ConfigurationPathError
from vrg.bench.col_cache_scenes import ep_dir

ROOT = "/scratch/kaustubh/vlm_reach_grasp/colosseum"
MAX_KF = 25


def rot6(q):
    R = Rot.from_quat(q).as_matrix(); return np.concatenate([R[:, 0], R[:, 1]])


def from6(r):
    a, b = r[:3], r[3:]; x = a / np.linalg.norm(a); b = b - x.dot(b) * x; y = b / np.linalg.norm(b)
    return Rot.from_matrix(np.stack([x, y, np.cross(x, y)], 1)).as_quat()


def project_rlbench(P, K, T):
    x = (K @ (T[:3, :3].T @ (np.asarray(P).T - T[:3, 3:4]))).T; return x[:, :2] / x[:, 2:3]


def draw_hamster(img, uv):
    from PIL import Image, ImageDraw
    sc = 4; H_ = img.shape[0]; im = Image.fromarray(img).resize((H_ * sc, H_ * sc)); d = ImageDraw.Draw(im); p = [tuple(x * sc) for x in uv]
    for i in range(len(p) - 1):
        t = i / max(len(p) - 2, 1); d.line([p[i], p[i + 1]], fill=(int(40 + 215 * t), int(60 + 120 * (1 - t)), int(255 - 155 * t)), width=2 * sc)
    for x, y in p[1:]: d.ellipse([x - 2 * sc, y - 2 * sc, x + 2 * sc, y + 2 * sc], fill=(0, 230, 0))
    return np.array(im.resize((H_, H_)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True, choices=["ours", "e2e", "hamster"])
    ap.add_argument("--port", type=int, required=True); ap.add_argument("--tasks", required=True)
    ap.add_argument("--factors", default="0,2,6,8,9,10,11,12,14"); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", required=True); A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
    G = json.load(open(f"{ROOT}/grounding.json")); S = json.load(open(f"{ROOT}/col_stats.json"))
    TE = None
    if A.method == "e2e":
        import pickle; TE = pickle.load(open(f"{ROOT}/text_emb_py38.pkl", "rb"))
    conn = Client(("localhost", A.port), authkey=b"vrg")
    resf = f"{A.out}/results.json"; res = json.load(open(resf)) if os.path.exists(resf) else []
    done_keys = {(r["task"], r["factor"], r["ep"]) for r in res}
    for task in A.tasks.split(","):
        st = {k: np.array(v) for k, v in S[task].items() if isinstance(v, list)}
        for factor in [int(f) for f in A.factors.split(",")]:
            d = ep_dir("test", task, factor)
            if not os.path.isdir(d) or all((task, factor, e) in done_keys for e in range(A.n)): continue
            env, te = make_env(task, factor, image_size=128); te.set_variation(0); te.reset()
            for e in range(A.n):
                demo = compat_pickle_load(f"{d}/episode{e}/low_dim_obs.pkl"); vf = f"{d}/episode{e}/variation_number.pkl"
                te.set_variation(compat_pickle_load(vf) if os.path.exists(vf) else 0)
                for _ in range(3):          # object placement is randomized and occasionally fails (close_laptop_lid, MO_Size)
                    try: desc, obs = te.reset_to_demo(demo); break
                    except TaskEnvironmentError: desc = None
                if desc is None: print(f"[{A.method}] {task} f{factor} ep{e}: scene could not be placed, skipped", flush=True); continue
                if (task, factor, e) in done_keys: continue   # after the reset: the perturbation RNG advances once per reset
                g = G.get(f"test|{task}|{factor}|{e}"); t0 = time.time(); reward, n_kf, err = 0.0, 0, None
                path_start = obs.gripper_pose[:3].copy()
                reach = None if (g is None or g["reach"] is None) else np.array(g["reach"])
                goal = None
                if reach is not None:
                    goal = (np.array(g["dest"]) if g["dest"] is not None else reach) + st["off_goal"]

                def step(pos, quat, gopen, ignore):
                    return te.step(np.r_[pos, quat / np.linalg.norm(quat), float(gopen), float(ignore)])
                try:
                    if A.method == "ours":
                        if reach is None: raise RuntimeError("grounding")
                        hand = reach + st["off_k0"] + st["pre"]
                        obs, reward, term = step(hand, st["quat_pre"], 1.0, 0.0); n_kf += 1
                    while n_kf < MAX_KF and reward < 1.0:
                        gp = obs.gripper_pose
                        if A.method == "ours":
                            gv = Rot.from_quat(gp[3:]).inv().apply(goal - gp[:3])
                            o = {"wrist": obs.wrist_rgb, "proprio": np.array([obs.gripper_open, gv[0], gv[1], gv[2], gp[2]], np.float32)}
                        else:
                            front = obs.front_rgb
                            if A.method == "hamster":
                                if reach is None: raise RuntimeError("grounding")
                                K, T = obs.misc["front_camera_intrinsics"], obs.misc["front_camera_extrinsics"]
                                pts = np.stack([path_start, reach + st["off_k0"], goal])   # same 3 points as in training
                                front = draw_hamster(front, project_rlbench(pts, K, T))
                            o = {"front": front, "wrist": obs.wrist_rgb, "proprio": np.r_[gp[:3], rot6(gp[3:]), obs.gripper_open].astype(np.float32)}
                            if A.method == "e2e": o["text"] = TE[desc[0]]
                        conn.send({"obs": o}); a = conn.recv()[0]
                        if A.method == "ours":
                            Rc = Rot.from_quat(gp[3:]); pos = gp[:3] + Rc.apply(a[:3]); quat = (Rc * Rot.from_quat(from6(a[3:9]))).as_quat()
                        else:
                            pos, quat = a[:3], from6(a[3:9])
                        obs, reward, term = step(pos, quat, a[9] > 0.5, a[10] > 0.5); n_kf += 1
                        if term: break
                except (InvalidActionError, IKError, ConfigurationPathError) as ex:
                    err = type(ex).__name__
                except RuntimeError as ex:
                    err = str(ex)
                r = dict(task=task, factor=factor, ep=e, success=bool(reward >= 1.0), keyframes=n_kf, error=err, sec=time.time() - t0, desc=desc[0])
                res.append(r); json.dump(res, open(resf, "w"), indent=1)
                print(f"[{A.method}] {task} f{factor} ep{e}: success={r['success']} kf={n_kf} err={err} ({r['sec']:.1f}s)", flush=True)
            env.shutdown()
    conn.send("close")
    print(f"== {A.method} success {np.mean([r['success'] for r in res]):.3f} ({len(res)} eps)")


if __name__ == "__main__":
    main()
