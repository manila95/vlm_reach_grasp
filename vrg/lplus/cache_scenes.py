"""E8 (grounding study): cache LIBERO-Plus scenes once so every grounding model / strategy sees identical inputs offline.

Per task (init state 0, settled, as the benchmark): upright agentview RGB (VLM_RES), metric depth, intrinsics/extrinsics,
grounding-v6 proposals (robot self-filter, box centres), and ground truth (target, basket, every object). Ground truth is for
scoring only. Output: <out>/scene_<tid>.npz (+ meta.json); resumable. liberoplus env:
  $LPY -m vrg.lplus.cache_scenes --split dev --out $S/liberoplus/scene_cache
"""
import argparse, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="dev"); ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="0-2517"); A = ap.parse_args()
    from vrg.common import make_env, target_name
    from vrg.pipeline import reset_episode, render_rgbd, robot_mask
    from vrg.vlm import propose
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    keep = set(json.load(open(f"{root}/data/liberoplus_split.json"))[A.split]); lo, hi = map(int, A.tasks.split("-"))
    os.makedirs(A.out, exist_ok=True)
    for tid in sorted(t for t in keep if lo <= t <= hi):
        f = f"{A.out}/scene_{tid}.npz"
        if os.path.exists(f): continue
        try:
            env, t = make_env(tid); obs = reset_episode(env, tid, 0)
            img, depth, K, T = render_rgbd(env); mask = robot_mask(env)
            props = propose(depth, K, T, exclude=mask, center="box")
            tgt = target_name(t.name) + "_1"
            gt_all = {k[:-4]: obs[k].copy() for k in obs if k.endswith("_pos") and "_to_" not in k and not k.startswith("robot")}
            np.savez_compressed(f, img=img, depth=depth.astype(np.float32), K=K, T=T,
                                boxes=np.array([p["box"] for p in props]), xy=np.array([p["xy"] for p in props]),
                                top=np.array([p["top"] for p in props]), language=t.language, name=t.name, target=tgt,
                                gt=json.dumps({k: v.tolist() for k, v in gt_all.items()}))
            env.close(); print(tid, len(props), "proposals", flush=True)
        except Exception as e:
            print(tid, "ERROR", repr(e)[:200], flush=True)
    print("CACHE DONE", flush=True)


if __name__ == "__main__":
    main()
