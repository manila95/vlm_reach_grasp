"""HAMSTER-style 2D path prompts: project 3D keypoints into the (128px, opengl-flipped) agentview image and draw them.

Path = start -> pre-grasp -> grasp (green dot = close) -> lift -> release (red dot = open), drawn as a polyline whose
colour goes from dark to bright with time (as in HAMSTER's gradient trajectories).
"""
import numpy as np
from PIL import Image, ImageDraw

RES = 128


def camera_mats(env, cam="agentview", res=RES):
    from robosuite.utils import camera_utils as CU
    return CU.get_camera_intrinsic_matrix(env.sim, cam, res, res), CU.get_camera_extrinsic_matrix(env.sim, cam)


def project(pts, K, T):
    """world pts (N,3) -> upright pixel coords (N,2)."""
    P = np.c_[pts, np.ones(len(pts))] @ np.linalg.inv(T).T
    return np.stack([K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2], K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]], 1)


def _grad(i, n):
    t = i / max(n - 1, 1)
    return (int(40 + 215 * t), int(60 + 120 * (1 - t)), int(255 - 155 * t))


def draw_path(img_flipped, uv, grasp_idx=2, release_idx=4, scale=4):
    """img_flipped: raw (opengl) HxWx3 obs. uv: upright pixel keypoints. Returns raw-orientation image with path."""
    H = img_flipped.shape[0]
    im = Image.fromarray(img_flipped[::-1]).resize((H * scale, H * scale), Image.BILINEAR)
    d = ImageDraw.Draw(im); p = [tuple(x * scale) for x in uv]
    for i in range(len(p) - 1):
        d.line([p[i], p[i + 1]], fill=_grad(i, len(p) - 1), width=2 * scale)
    r = 3 * scale
    for idx, col in ((grasp_idx, (0, 230, 0)), (release_idx, (255, 30, 30))):
        x, y = p[idx]; d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    return np.array(im.resize((H, H), Image.BILINEAR))[::-1].copy()


# demo-derived keypoint offsets (relative to object / basket), so test-time paths come from VLM points
def keypoint_offsets(meta, Z):
    ee = Z["proprio"][:, :3]
    f = lambda key, ref: np.mean([ee[m["start"] + m[key]] - np.array(m[ref]) for m in meta], 0)
    return {"pre": f("ts", "obj0"), "grasp": f("tg", "obj0"), "lift": f("te", "obj0"), "release": f("tr_last", "basket0")}


def path_from_estimates(ee0, p_obj, p_bk, off, K, T):
    kp = np.stack([ee0, p_obj + off["pre"], p_obj + off["grasp"], p_obj + off["lift"], p_bk + off["release"]])
    return project(kp, K, T)
