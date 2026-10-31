"""Evaluate on LIBERO-Plus libero_object (liberoplus env). One trial per task from init state 0 (the benchmark's protocol),
the benchmark's instruction string, 280 steps after a 10-step settle. Sharded and resumable: <out>/results_<lo>_<hi>.json.

  --method ours    our v5 pipeline (VLM handoff + egocentric local grasp + scripted transport; vrg.rollout.run_task) with
                   the LIBERO-Plus grounding (vrg.lplus.ground) and the standard-scene calibration (data/grounding.pkl)
  --method openpi  an openpi policy server (pi0 / pi0-FAST / pi0.5 LIBERO checkpoints), client as openpi examples/libero/main.py
  --method lerobot a LeRobot policy server (vrg/lplus/lerobot_server.py: SmolVLA, MolmoAct2), raw obs at 360 px as lerobot-eval
  $LPY -m vrg.lplus.run --method openpi --port 8002 --tasks 0-629 --out $S/liberoplus/runs/pi05_libero
"""
import argparse, collections, glob, json, math, os, pickle, sys, time
import numpy as np

S = "/scratch/kaustubh/vlm_reach_grasp"
CLS = None


def category(tid):
    """Perturbation category and difficulty from LIBERO-Plus's task_classification.json, matched by task name."""
    global CLS
    if CLS is None:
        CLS = {d["name"]: d for d in json.load(open(f"{S}/src/LIBERO-plus/libero/libero/benchmark/task_classification.json"))["libero_object"]}
    from vrg.common import get_suite
    d = CLS.get(get_suite().get_task(tid).name, {}); return d.get("category"), d.get("difficulty_level")


def quat2axisangle(quat):
    quat = quat.copy(); quat[3] = min(max(quat[3], -1.0), 1.0); den = np.sqrt(1.0 - quat[3] * quat[3])
    return np.zeros(3) if math.isclose(den, 0.0) else (quat[:3] * 2.0 * math.acos(quat[3])) / den


def run_openpi(tid, client, video_dir=None, max_steps=280, wait=10, replan=5):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from vrg.common import get_suite
    suite = get_suite(); t = suite.get_task(tid)
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file), camera_heights=256, camera_widths=256)
    env.seed(7); env.reset(); obs = env.set_init_state(suite.get_task_init_states(tid)[0])
    plan, frames, done, step = collections.deque(), [], False, 0
    for step in range(max_steps + wait):
        if step < wait: obs, _, done, _ = env.step([0.0] * 6 + [-1.0]); continue
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
        wimg = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
        if video_dir is not None: frames.append(img)
        if not plan:
            el = {"observation/image": img, "observation/wrist_image": wimg, "prompt": str(t.language),
                  "observation/state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))}
            for attempt in range(4):              # the server's first call compiles (> websocket keepalive): reconnect and retry
                try: plan.extend(client.infer(el)["actions"][:replan]); break
                except Exception:
                    if attempt == 3: raise
                    time.sleep(10); client.reconnect()
        obs, _, done, _ = env.step(plan.popleft().tolist())
        if done: break
    env.close()
    if video_dir and frames:
        import imageio; os.makedirs(video_dir, exist_ok=True); imageio.mimsave(f"{video_dir}/t{tid}_{'ok' if done else 'fail'}.mp4", frames, fps=20)
    return dict(tid=tid, success=bool(done), steps=step + 1, language=t.language, name=t.name)


def run_lerobot(tid, conn, video_dir=None, max_steps=280, wait=10, size=360):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from vrg.common import get_suite
    suite = get_suite(); t = suite.get_task(tid)
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file), camera_heights=size, camera_widths=size)
    env.seed(7); env.reset(); obs = env.set_init_state(suite.get_task_init_states(tid)[0])
    for _ in range(wait): obs, _, done, _ = env.step([0.0] * 6 + [-1.0])
    conn.send({"cmd": "reset"}); conn.recv(); frames, done, step = [], False, 0
    keys = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
            "robot0_gripper_qvel", "robot0_joint_pos", "robot0_joint_vel")
    for step in range(max_steps):
        o = {k: obs[k] for k in keys}; o["eef_mat"] = env.env.robots[0].controller.ee_ori_mat.copy()
        if video_dir is not None: frames.append(obs["agentview_image"][::-1, ::-1])
        conn.send({"cmd": "act", "obs": o, "task": str(t.language)}); a = conn.recv()
        obs, _, done, _ = env.step(np.asarray(a, dtype=np.float64).tolist())
        if done: break
    env.close()
    if video_dir and frames:
        import imageio; os.makedirs(video_dir, exist_ok=True); imageio.mimsave(f"{video_dir}/t{tid}_{'ok' if done else 'fail'}.mp4", frames, fps=20)
    return dict(tid=tid, success=bool(done), steps=step + 1, language=t.language, name=t.name)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", choices=["ours", "openpi", "lerobot"], required=True); ap.add_argument("--tasks", required=True)
    ap.add_argument("--port", type=int, default=8002); ap.add_argument("--out", required=True); ap.add_argument("--video_every", type=int, default=100)
    ap.add_argument("--wait", type=int, default=10)
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all", help="data/liberoplus_split.json (fixed before any tuning)")
    ap.add_argument("--oracle", action="store_true", help="ours: ground-truth object/basket positions (attribution, E0)")
    ap.add_argument("--skill", choices=["learned", "scripted"], default="learned", help="ours: learned egocentric grasp or scripted grasp (E0)")
    ap.add_argument("--ckpt", default="runs/local_ego_50/ckpt.pt", help="ours: grasp-skill checkpoint")
    ap.add_argument("--grounding", default=f"{S}/liberoplus/grounding_all.pkl"); A = ap.parse_args()
    lo, hi = map(int, A.tasks.split("-")); os.makedirs(A.out, exist_ok=True); f = f"{A.out}/results_{lo}_{hi}.json"
    res = json.load(open(f)) if os.path.exists(f) else []; done = {r["tid"] for r in res}
    if A.method == "openpi":
        from openpi_client import websocket_client_policy

        class Client:
            def __init__(self): self.reconnect()
            def reconnect(self):                    # the handshake times out while the server compiles for other clients: retry
                for attempt in range(60):
                    try: self.c = websocket_client_policy.WebsocketClientPolicy("localhost", A.port); return
                    except Exception as e:
                        if attempt == 59: raise
                        print(f"connect retry {attempt}: {repr(e)[:80]}", flush=True); time.sleep(30)
            def infer(self, el): return self.c.infer(el)
        client = Client()
    elif A.method == "lerobot":
        from multiprocessing.connection import Client as MPClient
        conn = MPClient(("127.0.0.1", A.port), authkey=b"vrg-lplus")
    else:
        from vrg.rollout import run_task, build_parser
        RA = build_parser().parse_args(["--method", "hier_vlm" if A.skill == "learned" else "hier_scripted", "--out", A.out, "--auto_retract",
                                        "--lock_ori", "auto", "--retries", "2", "--j4_thresh=-1.0", "--ckpt", A.ckpt, "--grounding", A.grounding,
                                        "--calib", "data/grounding.pkl", "--device", "cuda:0"] + (["--oracle"] if A.oracle else []))
    keep = None if A.split == "all" else set(json.load(open(f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/data/liberoplus_split.json"))[A.split])
    for tid in range(lo, hi + 1):
        if tid in done or (keep is not None and tid not in keep): continue
        t0 = time.time(); vid = (tid - lo) % A.video_every == 0
        try:
            if A.method == "openpi":
                r = run_openpi(tid, client, video_dir=f"{A.out}/videos" if vid else None, wait=A.wait)
            elif A.method == "lerobot":
                r = run_lerobot(tid, conn, video_dir=f"{A.out}/videos" if vid else None, wait=A.wait)
            else:
                RA.video = 1 if vid else 0
                r = run_task((tid, [0], RA))[0]; r["tid"] = tid
        except Exception as e:                       # infrastructure error: do not record, so a resumed run retries this task
            print(f"task {tid} ERROR (not recorded): {repr(e)[:200]}", flush=True); continue
        r["category"], r["difficulty"] = category(tid); r["sec"] = time.time() - t0
        res.append(r); json.dump(res, open(f, "w"), default=float)
        print(f"task {tid} [{r['category']} L{r['difficulty']}] success={r['success']} {r.get('fail', '')} {r.get('error', '')} ({r['sec']:.0f}s)", flush=True)
    print(f"== {A.out} {lo}-{hi}: success {np.mean([r['success'] for r in res]):.3f} over {len(res)}", flush=True)


if __name__ == "__main__":
    main()
