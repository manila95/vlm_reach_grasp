"""E3b / I3 (docs/PLAN_perceptual_robustness.md): a viewpoint-invariant basket centre from the depth proposal (no VLM).

For the proposal that is the basket (nearest the ground truth), compare centre estimators:
  med  median of the cluster's points (current; bias-calibrated on clean scenes)
  box  centre of the cluster's bird's-eye bounding box
  rim  centre of the bounding box of the rim band (points within 2 cm of the cluster's 98th-percentile height)
Each estimator gets its own bias from the clean CALIB_EPS scenes; residual xy error is reported on the clean EVAL_EPS scenes
and under our camera perturbation (vrg/perturb.py "camera"). Same for the target object (median vs box).
  LIBERO_CONFIG_PATH=$S/libero_std_cfg $PY -m vrg.eval_basket_center
"""
import numpy as np


def centres(P):
    xy, z = P[:, :2], P[:, 2]; top = np.percentile(z, 98); rim = xy[z > top - 0.02]
    return {"med": np.median(xy, 0), "box": (xy.min(0) + xy.max(0)) / 2, "rim": (rim.min(0) + rim.max(0)) / 2}


def main():
    from vrg.common import get_suite, make_env, target_name
    from vrg.pipeline import reset_episode, render_rgbd
    from vrg.vlm import backproject, propose
    from vrg.ground_episodes import CALIB_EPS, EVAL_EPS
    suite = get_suite(); rec = []
    for tid in range(suite.n_tasks):
        env, t = make_env(tid)
        for ep in CALIB_EPS + EVAL_EPS:
            for kind in ([None] if ep in CALIB_EPS else [None, "camera"]):
                obs = reset_episode(env, tid, ep, False, kind); img, depth, K, T = render_rgbd(env)
                P = backproject(depth, K, T); props = propose(depth, K, T)
                # recover each proposal's points by re-running the same mask on the box (proposals keep only summaries)
                for key, gt in (("basket", obs["basket_1_pos"]), ("obj", obs[target_name(t.language) + "_1_pos"])):
                    p = min(props, key=lambda p: np.linalg.norm(p["xy"] - gt[:2]))
                    x0, y0, x1, y1 = p["box"]; Q = P[y0:y1 + 1, x0:x1 + 1].reshape(-1, 3)
                    Q = Q[(Q[:, 2] > 0.004) & (Q[:, 2] < 0.25) & (np.linalg.norm(Q[:, :2] - p["xy"], axis=1) < 0.15)]
                    rec.append(dict(tid=tid, ep=ep, kind=kind or "none", key=key, calib=ep in CALIB_EPS, gt=gt[:2].copy(),
                                    **{k: v for k, v in centres(Q).items()}))
        env.close(); print(tid, flush=True)
    for key in ("basket", "obj"):
        for est in ("med", "box", "rim"):
            cal = [r for r in rec if r["key"] == key and r["calib"]]
            bias = np.mean([r[est] - r["gt"] for r in cal], 0)
            out = []
            for kind in ("none", "camera"):
                e = [np.linalg.norm(r[est] - bias - r["gt"]) for r in rec if r["key"] == key and not r["calib"] and r["kind"] == kind]
                out.append(f"{kind}: median {100*np.median(e):.1f} cm, p90 {100*np.percentile(e, 90):.1f} cm")
            print(f"{key:6s} {est}: bias {np.round(100*bias, 1)} cm | " + " | ".join(out), flush=True)
    print("CENTRE DONE")


if __name__ == "__main__":
    main()
