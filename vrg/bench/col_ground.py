"""Stage 2 (vlmrg env): VLM grounding for cached Colosseum scenes -> colosseum/grounding.pkl.

For each scene: the VLM reads the instruction and names (a) the object/part the robot must reach first and (b) the
destination (or null); both are grounded with VLM boxes + zoomed yes/no verification on the 512 px front view and
lifted to 3D with the simulator point cloud (median over the box centre).
"""
import glob, json, os, pickle, re, sys
import numpy as np
from vrg.vlm import Grounder, zoom
from vrg.bench.rc_ground import parse_boxes, iou

ROOT = "/scratch/kaustubh/vlm_reach_grasp/colosseum"
P_PARSE = ('A robot arm on a table receives the instruction: "{instr}". Answer in JSON with keys "reach" (the object or '
           'object part the gripper must touch or grasp first, a short noun phrase, e.g. "the handle of the bottom drawer", '
           '"the ball", "the lid of the box") and "destination" (where that object must end up, a short noun phrase, or null '
           'if there is no separate destination).')
P_TGT = ('Locate {n} in the image. Output JSON like [{{"bbox_2d": [x1, y1, x2, y2], "label": "{n}"}}] '
         '(several boxes if unsure), or [] if absent.')
P_VER = 'This is a close-up crop of a robot workspace. Is the thing in the centre of this crop {n}? Answer Yes or No.'


def parse(g, instr):
    r = g.ask(np.zeros((64, 64, 3), np.uint8), P_PARSE.format(instr=instr), max_new_tokens=96)
    d = json.loads(re.search(r"\{.*\}", r, re.S).group(0))
    return d.get("reach"), d.get("destination")


def lift(pc, box, frac=0.3):
    x0, y0, x1, y1 = box; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; w, h = max((x1 - x0) * frac / 2, 1), max((y1 - y0) * frac / 2, 1)
    P = pc[int(cy - h):int(cy + h) + 1, int(cx - w):int(cx + w) + 1].reshape(-1, 3)
    return np.median(P, 0)


def ground(g, front, name):
    img = front["rgb"]; H, W = img.shape[:2]
    r = g.ask(img, P_TGT.format(n=name), max_new_tokens=256); boxes = parse_boxes(r, W, H)
    cands = []
    for b in boxes:
        if any(iou(b, c["box"]) > 0.7 for c in cands): continue
        s = g.yes_score(zoom(img, b, margin=0.6, out=448, min_side=48), P_VER.format(n=name))
        cands.append(dict(box=b, score=s, point=lift(front["pc"], b)))
    return (max(cands, key=lambda c: c["score"]) if cands else None), cands


if __name__ == "__main__":
    out = f"{ROOT}/grounding.pkl"; res = pickle.load(open(out, "rb")) if os.path.exists(out) else {}
    g = Grounder(); cache = {}
    for f in sorted(glob.glob(f"{ROOT}/scenes/*/*.pkl")):
        s = pickle.load(open(f, "rb")); key = (s["mode"], s["task"], s["factor"], s["ep"])
        if key in res: continue
        if s["desc"] not in cache: cache[s["desc"]] = parse(g, s["desc"])
        reach, dest = cache[s["desc"]]
        br, _ = ground(g, s["front"], reach); bd, _ = ground(g, s["front"], dest) if dest else (None, [])
        res[key] = dict(desc=s["desc"], reach_name=reach, dest_name=dest,
                        reach=None if br is None else br["point"], reach_score=None if br is None else br["score"],
                        dest=None if bd is None else bd["point"], gt_k0=np.array(s["gt_k0"]), gt_goal=np.array(s["gt_goal"]))
        e = None if br is None else float(np.linalg.norm(br["point"] - s["gt_k0"][:3]))
        print(key, f'"{s["desc"]}" -> reach="{reach}" dest="{dest}" | dist(VLM reach point, demo k0) = {e if e is None else round(100*e,1)} cm', flush=True)
        pickle.dump(res, open(out, "wb"))
