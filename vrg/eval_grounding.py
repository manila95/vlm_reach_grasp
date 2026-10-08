"""Offline grounding accuracy: does the lifted 3D point land nearest the target object?"""
import numpy as np, pickle, re, sys, json, argparse
from PIL import Image
from vrg.vlm import Grounder
from vrg.common import target_name
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--res", type=int, default=1024); ap.add_argument("--variants", default="crop_bbox,detect_choose")
ap.add_argument("--verbose", action="store_true"); A = ap.parse_args()
S = pickle.load(open(f"data/scenes_{A.res}.pkl", "rb"))
g = Grounder(A.model)
CROP = (0.33, 0.82, 0.08, 0.98)  # floor region (rows, cols) as image fractions

def unproject(s, u, v):
    d = float(np.median(s["depth"][int(v)-2:int(v)+3, int(u)-2:int(u)+3])); K = s["K"]
    return (s["T"] @ np.array([(u-K[0,2])*d/K[0,0], (v-K[1,2])*d/K[1,1], d, 1]))[:3]

def crop(img):
    H, W = img.shape[:2]; y0, y1, x0, x1 = int(CROP[0]*H), int(CROP[1]*H), int(CROP[2]*W), int(CROP[3]*W)
    return img[y0:y1, x0:x1], (x0, y0)

def boxes(r):
    return [[float(v) for v in m.split(",")] for m in re.findall(r"\[\s*(-?[\d.]+\s*,\s*-?[\d.]+\s*,\s*-?[\d.]+\s*,\s*-?[\d.]+)\s*\]", r)]

def to_px(b, c, off):
    h, w = c.shape[:2]; return off[0] + (b[0]+b[2])/2/1000*w, off[1] + (b[1]+b[3])/2/1000*h

P_BB = 'Locate the {n} in the image and output its bbox coordinates in JSON format like [{{"bbox_2d": [x1, y1, x2, y2], "label": "{n}"}}].'
P_ALL = ('Detect every grocery item on the floor (ignore the basket and the robot). Read the packaging labels carefully. '
         'Output JSON like [{"bbox_2d": [x1, y1, x2, y2], "label": "<product name and container type>"}].')

def ground(s, name, variant):
    c, off = crop(s["img"])
    if variant == "crop_bbox":
        r = g.ask(c, P_BB.format(n=name)); b = boxes(r)
        return (to_px(b[0], c, off) if b else None), r
    if variant == "detect_choose":
        r = g.ask(c, P_ALL, max_new_tokens=512)
        try: dets = json.loads(re.search(r"\[.*\]", r, re.S).group(0))
        except Exception: return None, r
        dets = [d for d in dets if "bbox_2d" in d and len(d["bbox_2d"]) == 4]
        if not dets: return None, r
        listing = "\n".join(f"{i}: {d.get('label','?')}" for i, d in enumerate(dets))
        q = (f"Detected items:\n{listing}\nWhich item index best matches \"{name}\"? Answer with only the index number.")
        a = g.ask(np.zeros((64, 64, 3), np.uint8), q, max_new_tokens=8)
        m = re.search(r"\d+", a); i = int(m.group(0)) if m and int(m.group(0)) < len(dets) else 0
        return to_px(dets[i]["bbox_2d"], c, off), listing + f"\n-> {a}"

for vn in A.variants.split(","):
    ok = 0; errs = []; per = {}
    for s in S:
        name = target_name(s["lang"]); tgt = name + "_1"
        uv, r = ground(s, name.replace("_", " "), vn); hit = False
        if uv is not None:
            p = unproject(s, *uv)
            cands = {k: v for k, v in s["objs"].items() if k != "basket_1"}
            hit = min(cands, key=lambda k: np.linalg.norm((cands[k]-p)[:2])) == tgt
            errs.append(np.linalg.norm((s["objs"][tgt]-p)[:2]))
        ok += hit; per.setdefault(name, []).append(hit)
        if A.verbose and not hit: print("   miss", s["tid"], s["ep"], name, r.replace("\n", " | ")[:300])
    print(f"{A.model.split('/')[-1]} res={A.res} {vn:14s} acc={ok}/{len(S)} med_xy_err={np.median(errs):.3f}  per-task:",
          {k: sum(v) for k, v in per.items()}, flush=True)
