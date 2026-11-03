"""Refine the VLM pick point of a RoboCasa grounding file with the cached RGB-D point cloud (perception only, no GT).

xy: centre of the bounding box of the points within 8 cm of the VLM point that stand more than 1.2 cm above the local
support plane (20th percentile height of a 10-22 cm ring around the point); z: the VLM point's height (its error is small).
Falls back to the VLM point when fewer than 30 object points are found. Writes a new grounding file with pick replaced.
  $PY -m vrg.bench.rc_pick_refine <grounding.pkl> <scenes dir> <out.pkl>
"""
import pickle, sys
import numpy as np
from vrg.bench.rc_pick_top import cloud

G = pickle.load(open(sys.argv[1], "rb")); out = {}; n_ref = 0
for key, g in G.items():
    g = dict(g); sp, task, ep = key
    if g.get("pick") is not None:
        S = pickle.load(open(f"{sys.argv[2]}/{sp}/{task}_{ep}.pkl", "rb")); p = g["pick"]
        P = np.concatenate([cloud(v) for v in S["views"].values()]); d = np.linalg.norm(P[:, :2] - p[:2], axis=1)
        ring = P[(d > 0.10) & (d < 0.22) & (np.abs(P[:, 2] - p[2]) < 0.15)]; z0 = np.percentile(ring[:, 2], 20) if len(ring) > 50 else p[2] - 0.03
        near = P[(d < 0.08) & (P[:, 2] > z0 + 0.012) & (P[:, 2] < p[2] + 0.15)]
        if len(near) > 30:
            g["pick_vlm"] = p.copy(); g["pick"] = np.array([*((near[:, :2].min(0) + near[:, :2].max(0)) / 2), p[2]]); n_ref += 1
    out[key] = g
pickle.dump(out, open(sys.argv[3], "wb")); print("refined", n_ref, "of", len(G))
