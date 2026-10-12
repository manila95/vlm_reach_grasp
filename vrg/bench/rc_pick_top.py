"""Add 'pick_top' (estimated top-surface height of the pick target) to a RoboCasa grounding file, from the cached RGB-D.

Points from both base cameras within R_XY of the VLM pick point (and within a plausible height band) are back-projected;
the 95th-percentile height is the top. Perception only: no simulator ground truth.
python -m vrg.bench.rc_pick_top grounding.pkl scenes
"""
import glob, os, pickle, shutil, sys
import numpy as np

R_XY, BAND = 0.05, (-0.10, 0.40)


def cloud(view):
    d = view["depth"].astype(np.float32); K, T = view["K"], view["T"]; H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]; x = (u - K[0, 2]) * d / K[0, 0]; y = (v - K[1, 2]) * d / K[1, 1]
    P = np.stack([x, y, d, np.ones_like(d)], -1).reshape(-1, 4) @ T.T; return P[:, :3]


def top(scene, p):
    P = np.concatenate([cloud(v) for v in scene["views"].values()])
    m = (np.linalg.norm(P[:, :2] - p[:2], axis=1) < R_XY) & (P[:, 2] > p[2] + BAND[0]) & (P[:, 2] < p[2] + BAND[1])
    return float(np.percentile(P[m, 2], 95)) if m.sum() > 20 else float(p[2])


if __name__ == "__main__":
    out, scenes = sys.argv[1], sys.argv[2]
    shutil.copy(out, out + ".bak"); G = pickle.load(open(out, "rb"))
    for f in sorted(glob.glob(f"{scenes}/*/*.pkl")):
        s = pickle.load(open(f, "rb")); k = (s["split"], s["task"], s["ep"])
        if k in G and G[k]["pick"] is not None: G[k]["pick_top"] = top(s, np.asarray(G[k]["pick"]))
    pickle.dump(G, open(out, "wb"))
    h = [100 * (r["pick_top"] - r["pick"][2]) for r in G.values() if "pick_top" in r]
    print(out, len(h), "scenes; top - pick point (cm): median %.1f, p90 %.1f, max %.1f" % (np.median(h), np.percentile(h, 90), np.max(h)))
