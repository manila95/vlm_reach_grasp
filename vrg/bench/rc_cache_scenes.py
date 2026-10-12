"""Stage 1 (robocasa env): reset each evaluation episode and cache RGB-D (both base cams, 512 px), K/T, GT, instruction.

python -m vrg.bench.rc_cache_scenes --split pretrain --eps 0-9 [--tasks ...]
"""
import argparse, os, pickle, time
import numpy as np
from vrg.bench.robocasa_env import TASKS, CAMS, make, episode_seed, raw, render_rgbd, gt_objects, scene_info, raw_obs, base_pose

OUT = "/scratch/kaustubh/vlm_reach_grasp/robocasa/scenes"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True); ap.add_argument("--eps", default="0-9")
    ap.add_argument("--tasks", default=",".join(TASKS)); ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--out", default=OUT, help="scenes_dev for debugging episodes (>=50), kept apart from the eval scenes"); A = ap.parse_args()
    OUT = A.out
    lo, hi = map(int, A.eps.split("-")); os.makedirs(f"{OUT}/{A.split}", exist_ok=True)
    for task in A.tasks.split(","):
        g = None
        for ep in range(lo, hi + 1):
            f = f"{OUT}/{A.split}/{task}_{ep}.pkl"
            if os.path.exists(f): continue
            t0 = time.time(); seed = episode_seed(A.split, ep)
            if g is None: g = make(task, A.split, seed)
            g.reset(seed=seed); e = raw(g); o = raw_obs(e)
            views = {c: dict(zip(("rgb", "depth", "K", "T"), render_rgbd(e, c, A.res))) for c in CAMS}
            for v in views.values(): v["depth"] = v["depth"].astype(np.float16)
            bp, R = base_pose(o)
            pickle.dump(dict(task=task, split=A.split, ep=ep, seed=seed, views=views, gt=gt_objects(e), info=scene_info(e),
                             base_pos=bp, base_R=R, eef=o["robot0_eef_pos"].copy()), open(f, "wb"))
            print(task, ep, f"{time.time() - t0:.1f}s", scene_info(e)["lang"], flush=True)
        if g is not None: g.close()
