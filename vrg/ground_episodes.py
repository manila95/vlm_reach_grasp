"""Stage 1: VLM task parsing + grounding for every evaluation episode (cached; scene is deterministic after settle)."""
import argparse, pickle, os
import numpy as np
from PIL import Image, ImageDraw
from vrg.common import *
from vrg.pipeline import reset_episode, render_rgbd, robot_mask, unproject, vlm_ground
from vrg.vlm import Grounder, ground_propose_verify

CALIB_EPS = [0, 10, 20, 30, 40]
EVAL_EPS = [e for e in range(50) if e % 10 != 0][:20]
HOLDOUT_EPS = [e for e in range(50) if e % 10 != 0][20:]  # 25/task, untouched by any tuning

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="data/grounding.pkl"); ap.add_argument("--shuffle", action="store_true"); ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--v6", action="store_true", help="grounding v6: robot self-filter + box centres (docs/PLAN_perceptual_robustness.md)")
    ap.add_argument("--perturb", default=None, help="vrg/perturb.py kind for the grounded scenes"); A = ap.parse_args()
    g = Grounder(A.model); res = {}
    os.makedirs("logs/grounding", exist_ok=True)
    suite = get_suite()
    for tid in range(suite.n_tasks):
        env, t = make_env(tid)
        pick, place, raw = g.parse_target(t.language)          # VLM decomposes the instruction
        for ep in CALIB_EPS + (HOLDOUT_EPS if A.holdout else EVAL_EPS):
            obs = reset_episode(env, tid, ep, A.shuffle, A.perturb)
            img, depth, K, T = render_rgbd(env); V6 = dict(exclude=robot_mask(env), center="box") if A.v6 else {}
            r = dict(tid=tid, ep=ep, pick=pick, place=place, calib=ep in CALIB_EPS,
                     gt_obj=obs[target_name(t.language) + "_1_pos"].copy(), gt_basket=obs["basket_1_pos"].copy(),
                     gt_all={k[:-4]: obs[k].copy() for k in obs if k.endswith("_pos") and "_to_" not in k and not k.startswith("robot")})
            for key, name in (("obj", pick), ("basket", place)):
                best, props = ground_propose_verify(g, img, depth, K, T, name, **V6)
                r[key + "_box"] = [float(x) for x in best["box"]] if best else None
                r[key + "_scores"] = sorted([p["score"] for p in props], reverse=True)
                r[key] = np.array([*best["xy"], best["top"] / 2]) if best else None
            p = r["obj"]
            cands = {k: v for k, v in r["gt_all"].items() if k != "basket_1"}
            r["obj_hit"] = p is not None and min(cands, key=lambda k: np.linalg.norm((cands[k] - p)[:2])) == target_name(t.language) + "_1"
            res[(tid, ep)] = r
            if ep in (CALIB_EPS[0], EVAL_EPS[0]):
                im = Image.fromarray(img); d = ImageDraw.Draw(im)
                for key, col in (("obj_box", (255, 0, 0)), ("basket_box", (0, 120, 255))):
                    if r[key]: d.rectangle(r[key], outline=col, width=4)
                d.text((10, 10), f"pick: {pick} | place: {place} | hit={r['obj_hit']}", fill=(255, 255, 0))
                im.save(f"logs/grounding/{tid}_{ep}{'_shuf' if A.shuffle else ''}.png")
            print(tid, ep, pick, "| hit", r["obj_hit"], flush=True)
        env.close()
    pickle.dump(res, open(A.out, "wb"))
    ev = [r for r in res.values() if not r["calib"]]
    print("parse:", {r["tid"]: (r["pick"], r["place"]) for r in ev})
    print("eval grounding acc", np.mean([r["obj_hit"] for r in ev]))
