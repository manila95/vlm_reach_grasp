"""Geometry-only perception on calibrated RGB-D frames (base frame): table plane, object proposals, top-down grasps.

Works on any CameraFrame, real or simulated. No learned components here; the VLM decides WHICH proposal (vlm.py).
"""
from dataclasses import dataclass
import numpy as np
from vrg.franka.core import CameraFrame


@dataclass
class Workspace:
    """Region (base frame) where objects can be. Set it to the real table area in front of the robot."""
    x: tuple = (0.25, 0.80)
    y: tuple = (-0.40, 0.40)
    z_max_above_table: float = 0.30


def fit_table(P, iters=200, thresh=0.006, rng=None):
    """RANSAC plane among the lowest points; returns (normal n with n_z > 0, offset d) with n.p + d = 0."""
    rng = np.random.default_rng(0) if rng is None else rng
    low = P[P[:, 2] < np.percentile(P[:, 2], 40)]
    best, bn, bd = -1, np.array([0, 0, 1.0]), -np.median(low[:, 2])
    for _ in range(iters):
        a, b, c = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn; n = n if n[2] > 0 else -n
        if n[2] < 0.9: continue                                   # table must be roughly horizontal in the base frame
        d = -n @ a; k = np.sum(np.abs(low @ n + d) < thresh)
        if k > best: best, bn, bd = k, n, d
    return bn, bd


def proposals(frames, ws=Workspace(), cell=0.005, min_pts=60, z_min=0.008):
    """Class-agnostic objects: points above the table, clustered in a bird's-eye occupancy grid, fused over cameras.

    Returns (props, table) with props = list of dict(id, points (N x 3), center, top, extent, pix: {cam: (box, mask)}).
    """
    from scipy import ndimage
    allP, src = [], []
    for name, f in frames.items():
        valid = np.isfinite(f.depth) & (f.depth > 0); v, u = np.nonzero(valid)
        P = f.backproject(); allP.append(P); src.append((name, v, u))
    P = np.concatenate(allP)
    n, d = fit_table(P[(P[:, 0] > ws.x[0]) & (P[:, 0] < ws.x[1]) & (np.abs(P[:, 1]) < ws.y[1])])
    h = P @ n + d
    m = (h > z_min) & (h < ws.z_max_above_table) & (P[:, 0] > ws.x[0]) & (P[:, 0] < ws.x[1]) & (P[:, 1] > ws.y[0]) & (P[:, 1] < ws.y[1])
    ij = np.floor((P[:, :2] - [ws.x[0], ws.y[0]]) / cell).astype(int)
    shp = (int((ws.x[1] - ws.x[0]) / cell) + 1, int((ws.y[1] - ws.y[0]) / cell) + 1)
    occ = np.zeros(shp, bool); occ[ij[m, 0], ij[m, 1]] = True
    lab, k = ndimage.label(ndimage.binary_closing(occ, iterations=1))
    plab = np.zeros(len(P), int); plab[m] = lab[ij[m, 0], ij[m, 1]]
    props = []
    for c in range(1, k + 1):
        sel = plab == c
        if sel.sum() < min_pts: continue
        Q = P[sel]; hh = h[sel]
        pix, off = {}, 0
        for (name, v, u), Pc in zip(src, allP):
            s = sel[off:off + len(Pc)]; off += len(Pc)
            if s.sum() < 20: continue
            mask = np.zeros(frames[name].depth.shape, bool); mask[v[s], u[s]] = True
            pix[name] = ([int(u[s].min()), int(v[s].min()), int(u[s].max()), int(v[s].max())], mask)
        props.append(dict(id=len(props), points=Q, center=np.median(Q, 0), top=float(np.percentile(hh, 98)),
                          extent=np.ptp(Q[:, :2], 0), pix=pix))
    return props, (n, d)


def top_down_grasp(prop, table, max_width=0.075, finger_depth=0.03, min_clear=0.012, yaw_mode="pca"):
    """Scripted top-down grasp for one proposal.

    yaw_mode: "pca"   close the fingers across the object's narrow horizontal axis (a simple antipodal heuristic)
              "fixed" fingers along the base y axis (what a naive scripted grasp does)
    Returns dict(T (4x4 TCP pose), width (opening needed), feasible, why).
    """
    n, d = table; Q = prop["points"]; xy = Q[:, :2] - Q[:, :2].mean(0)
    if yaw_mode == "pca":
        w, V = np.linalg.eigh(np.cov(xy.T)); minor = V[:, 0]             # smallest variance direction
        close_dir = minor
    else:
        close_dir = np.array([0.0, 1.0])
    proj = xy @ close_dir; width = float(np.percentile(proj, 98) - np.percentile(proj, 2)) + 0.01
    top_z = float(np.percentile(Q[:, 2], 98)); table_z = -d / n[2]
    z = max(top_z - finger_depth, table_z + min_clear)
    c = np.r_[np.median(Q[:, :2], 0), z]
    # TCP frame: z down, y = finger closing direction (Franka Hand fingers move along the hand's y axis)
    y = np.r_[close_dir, 0.0]; y /= np.linalg.norm(y); zax = np.array([0, 0, -1.0]); x = np.cross(y, zax)
    if x[0] < 0: x, y = -x, -y                                     # keep the hand's x roughly forward (avoids wrist flips)
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, zax, c
    ok = width <= max_width
    return dict(T=T, width=width, feasible=ok, why="" if ok else f"too wide ({width*100:.1f} cm)", top_z=top_z)
