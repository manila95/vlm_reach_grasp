"""E3 (docs/PLAN_perceptual_robustness.md): grounding robustness on our own dev perturbations, offline (no rollouts).

Scenes: the 10 standard LIBERO-Object tasks x 5 eval init states, clean and under vrg/perturb.py light / texture / both.
Depth proposals are unaffected by appearance, so every variant re-scores the same proposals; only verification changes.
  G0  current: VLM yes/no on the raw zoomed crop
  G1  auto-exposure: per-crop percentile (2-98) contrast stretch before the VLM
  G2  G0 + size prior for containers: z-scored VLM score + z-scored bird's-eye area (only for the place target)
  G3  pooled: mean of the G0 and G1 scores
A choice is correct when it is the proposal nearest the ground-truth position. Output: <out> (pickle of per-scene records).
  LIBERO_CONFIG_PATH=$S/libero_std_cfg $PY -m vrg.eval_grounding_dev --out runs/grounding_dev/e3.pkl
"""
import argparse, os, pickle
import numpy as np


def stretch(img):
    x = img.astype(np.float32); lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    return (np.clip((x - lo) / max(hi - lo, 1.0), 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--eps", default="1,3,5,7,9"); ap.add_argument("--out", required=True); A = ap.parse_args()
    from vrg.common import get_suite, make_env, target_name
    from vrg.pipeline import reset_episode, render_rgbd
    from vrg.vlm import Grounder, propose, zoom
    from vrg.ground_episodes import EVAL_EPS
    g = Grounder(A.model); suite = get_suite(); os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
    res = pickle.load(open(A.out, "rb")) if os.path.exists(A.out) else {}
    q = ('This is a close-up of one item in a robot workspace. Is this item the %s? '
         'Read any label on the packaging. Answer Yes or No.')
    eps = [EVAL_EPS[int(i)] for i in A.eps.split(",")]
    for tid in range(suite.n_tasks):
        env, t = make_env(tid); pick, place, _ = g.parse_target(t.language)
        for ep in eps:
            for kind in ("none", "light", "texture", "both"):
                if (tid, ep, kind) in res: continue
                obs = reset_episode(env, tid, ep, False, None if kind == "none" else kind)
                img, depth, K, T = render_rgbd(env); props = propose(depth, K, T)
                gt = {"obj": obs[target_name(t.language) + "_1_pos"], "basket": obs["basket_1_pos"]}
                r = dict(tid=tid, ep=ep, kind=kind, n=len(props))
                for key, name in (("obj", pick), ("basket", place)):
                    raw = np.array([g.yes_score(zoom(img, p["box"]), q % name) for p in props])
                    nrm = np.array([g.yes_score(stretch(zoom(img, p["box"])), q % name) for p in props])
                    area = np.array([float(np.prod(p["extent"])) for p in props])
                    z = lambda v: (v - v.mean()) / (v.std() + 1e-6)
                    true = int(np.argmin([np.linalg.norm(p["xy"] - gt[key][:2]) for p in props]))
                    sc = {"G0": raw, "G1": nrm, "G3": (raw + nrm) / 2, "G2": z(raw) + (z(area) if key == "basket" else 0)}
                    for m, s in sc.items():
                        c = int(np.argmax(s)); r[f"{key}_{m}_ok"] = c == true
                        r[f"{key}_{m}_err"] = float(np.linalg.norm(props[c]["xy"] - gt[key][:2]))
                    r[f"{key}_true_err"] = float(np.linalg.norm(props[true]["xy"] - gt[key][:2]))
                res[(tid, ep, kind)] = r; pickle.dump(res, open(A.out, "wb"))
                print(tid, ep, kind, {k: v for k, v in r.items() if k.endswith("_ok")}, flush=True)
        env.close()
    print("E3 DONE", flush=True)
    for kind in ("none", "light", "texture", "both"):
        rs = [r for r in res.values() if r["kind"] == kind]
        print(kind, {f"{k}_{m}": round(np.mean([r[f'{k}_{m}_ok'] for r in rs]), 3) for k in ("obj", "basket") for m in ("G0", "G1", "G2", "G3")})


if __name__ == "__main__":
    main()
