"""VLM grounding: instruction + agentview image -> 2D point -> 3D world point."""
import json, re
import numpy as np
import torch
from PIL import Image
from robosuite.utils import camera_utils as CU

MODEL = "Qwen/Qwen3-VL-8B-Instruct"


class Grounder:
    def __init__(self, model=MODEL, device="cuda:0"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model, dtype=torch.bfloat16, device_map=device).eval()
        self.proc = AutoProcessor.from_pretrained(model)

    @torch.no_grad()
    def ask(self, img, text, max_new_tokens=128):
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(img)},
                                             {"type": "text", "text": text}]}]
        inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            return_dict=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def parse_target(self, instruction):
        """Ask the VLM which object must be picked and where it must go (task decomposition)."""
        q = ("A robot receives the instruction: \"%s\". Answer in JSON with keys "
             "\"pick\" (the object to grasp) and \"place\" (the receptacle), using short noun phrases." % instruction)
        r = self.ask(np.zeros((64, 64, 3), np.uint8), q)
        d = json.loads(re.search(r"\{.*\}", r, re.S).group(0))
        return d["pick"], d["place"], r

    def point(self, img, name):
        """Return (u, v) pixel in the given (upright) image, plus raw VLM text."""
        H, W = img.shape[:2]
        q = ('Point to the %s in the image. Output its point coordinates in JSON format like '
             '[{"point_2d": [x, y], "label": "%s"}].' % (name, name))
        r = self.ask(img, q)
        m = re.search(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", r)
        if m is None:
            return None, r
        x, y = float(m.group(1)), float(m.group(2))
        return (x / 1000.0 * W, y / 1000.0 * H), r


def lift(sim, cam, depth_raw, uv, H, W, patch=2):
    """Upright-image pixel (u, v) + robosuite (opengl, upside-down) depth -> world xyz."""
    depth = CU.get_real_depth_map(sim, depth_raw)[..., 0]
    depth = depth[::-1]  # make upright, matching the image given to the VLM
    u, v = uv
    r0, c0 = int(round(v)), int(round(u))
    win = depth[max(r0 - patch, 0):r0 + patch + 1, max(c0 - patch, 0):c0 + patch + 1]
    d = float(np.median(win))
    K = CU.get_camera_intrinsic_matrix(sim, cam, H, W)
    T = CU.get_camera_extrinsic_matrix(sim, cam)  # cam (cv convention) -> world
    p = np.array([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d, 1.0])
    return (T @ p)[:3]


# ---------------------------------------------------------------- propose (geometry) + verify (VLM) grounding
def backproject(depth, K, T):
    H, W = depth.shape; vv, uu = np.mgrid[0:H, 0:W]
    P = np.stack([(uu - K[0, 2]) * depth / K[0, 0], (vv - K[1, 2]) * depth / K[1, 1], depth, np.ones_like(depth)], -1)
    return (P @ T.T)[..., :3]


def propose(depth, K, T, z_floor=0.004, z_max=0.25, cell=0.006, min_px=150, exclude=None, center="median"):
    """Class-agnostic object proposals: points above the floor plane, clustered in a bird's-eye occupancy grid.
    Grounding v6 (docs/PLAN_perceptual_robustness.md): exclude = robot self-filter mask; center="box" = bird's-eye box
    centre, which is unbiased and viewpoint-invariant (the median is pulled 7 cm toward the camera-facing side)."""
    from scipy import ndimage
    P = backproject(depth, K, T)
    m = (P[..., 2] > z_floor) & (P[..., 2] < z_max) & (np.abs(P[..., 0]) < 0.45) & (np.abs(P[..., 1]) < 0.45)
    if exclude is not None: m &= ~exclude
    ij = np.floor((P[..., :2] + 0.45) / cell).astype(int); n = int(0.9 / cell) + 1
    occ = np.zeros((n, n), bool); occ[ij[m][:, 0], ij[m][:, 1]] = True
    lab, k = ndimage.label(ndimage.binary_closing(occ, iterations=1))
    pix_lab = np.zeros(depth.shape, int); pix_lab[m] = lab[ij[m][:, 0], ij[m][:, 1]]
    props = []
    for c in range(1, k + 1):
        vs, us = np.nonzero(pix_lab == c)
        if len(vs) < min_px: continue
        pts = P[vs, us]
        xy = (pts[:, :2].min(0) + pts[:, :2].max(0)) / 2 if center == "box" else np.median(pts[:, :2], 0)
        props.append(dict(box=[us.min(), vs.min(), us.max(), vs.max()], xy=xy,
                          top=np.percentile(pts[:, 2], 98), npx=len(vs), extent=np.ptp(pts[:, :2], 0)))
    return props


def zoom(img, box, margin=0.5, out=448, min_side=80):
    x0, y0, x1, y1 = box; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0, min_side) * (1 + margin)
    H, W = img.shape[:2]
    a, b = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0)); c, d = int(min(cx + s / 2, W)), int(min(cy + s / 2, H))
    return np.array(Image.fromarray(img[b:d, a:c]).resize((out, int(out * (d - b) / max(c - a, 1))), Image.BICUBIC))


@torch.no_grad()
def yes_score(self, img, text):
    msgs = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(img)}, {"type": "text", "text": text}]}]
    inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.model.device)
    logits = self.model(**inp).logits[0, -1].float()
    tok = self.proc.tokenizer
    y = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")]
    n = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")]
    return (torch.logsumexp(logits[y], 0) - torch.logsumexp(logits[n], 0)).item()


Grounder.yes_score = yes_score


def ground_propose_verify(g, img, depth, K, T, name, exclude=None, center="median"):
    props = propose(depth, K, T, exclude=exclude, center=center)
    q = ('This is a close-up of one item in a robot workspace. Is this item the %s? '
         'Read any label on the packaging. Answer Yes or No.' % name)
    for p in props:
        p["score"] = g.yes_score(zoom(img, p["box"]), q)
    best = max(props, key=lambda p: p["score"]) if props else None
    return best, props
