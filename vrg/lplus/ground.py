"""VLM grounding for LIBERO-Plus libero_object tasks (liberoplus env: source /scratch/.../liberoplus/env.sh).

Same procedure as vrg/ground_episodes.py, on LIBERO-Plus's perturbed tasks: the VLM parses the benchmark's instruction
string (pick, place), then grounds both in the settled agentview RGB-D of init state 0 (one trial per task, as the
benchmark's protocol). Camera matrices come from each perturbed scene's simulator, so viewpoint perturbations are
handled. Ground truth is stored for analysis only. Output: <out>/ground_<lo>_<hi>.pkl {(tid, 0): record}; resumable.
  $LPY -m vrg.lplus.ground --tasks 0-629 --device cuda:0 --out $S/liberoplus/grounding
"""
import argparse, os, pickle, time
import numpy as np
from vrg.common import get_suite, make_env, target_name
from vrg.pipeline import reset_episode, render_rgbd
from vrg.vlm import Grounder, ground_propose_verify


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True); A = ap.parse_args()
    lo, hi = map(int, A.tasks.split("-")); os.makedirs(A.out, exist_ok=True); f = f"{A.out}/ground_{lo}_{hi}.pkl"
    res = pickle.load(open(f, "rb")) if os.path.exists(f) else {}
    g = Grounder(A.model, device=A.device); suite = get_suite()
    for tid in range(lo, hi + 1):
        if (tid, 0) in res: continue
        t0 = time.time(); env, t = None, suite.get_task(tid)
        try:
            env, t = make_env(tid)
            pick, place, raw = g.parse_target(t.language)
            obs = reset_episode(env, tid, 0)
            img, depth, K, T = render_rgbd(env)
            tgt = target_name(t.name) + "_1"
            r = dict(tid=tid, ep=0, pick=pick, place=place, calib=False, language=t.language, name=t.name,
                     gt_obj=obs[tgt + "_pos"].copy() if tgt + "_pos" in obs else None, gt_basket=obs.get("basket_1_pos"),
                     gt_all={k[:-4]: obs[k].copy() for k in obs if k.endswith("_pos") and "_to_" not in k and not k.startswith("robot")})
            for key, name in (("obj", pick), ("basket", place)):
                best, props = ground_propose_verify(g, img, depth, K, T, name)
                r[key + "_box"] = [float(x) for x in best["box"]] if best else None
                r[key] = np.array([*best["xy"], best["top"] / 2]) if best else None
            cands = {k: v for k, v in r["gt_all"].items() if k != "basket_1"}
            r["obj_hit"] = r["obj"] is not None and bool(cands) and min(cands, key=lambda k: np.linalg.norm((cands[k] - r["obj"])[:2])) == tgt
        except Exception as e:                                         # a broken perturbed scene is recorded, not fatal
            r = dict(tid=tid, ep=0, error=repr(e)[:300], obj=None, basket=None, obj_hit=False, name=t.name, language=t.language)
        if env is not None: env.close()
        res[(tid, 0)] = r; pickle.dump(res, open(f, "wb"))
        print(tid, r.get("pick"), "| hit", r.get("obj_hit"), r.get("error", ""), f"({time.time() - t0:.0f}s)", flush=True)
    print("GROUND DONE", lo, hi, flush=True)


if __name__ == "__main__":
    main()
