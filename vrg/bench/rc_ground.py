"""VLM grounding for RoboCasa scenes (run in the `vlmrg` env; reads scene caches written by rc_cache_scenes).

Candidates come from the VLM itself (open-vocabulary boxes for the target name + generic movable objects, on both
base cameras), every candidate is re-checked on a zoomed crop with a yes/no logit, and the best box is lifted to 3D
with the median depth of its central region.
"""
import json, re
import numpy as np
from vrg.vlm import zoom

P_TGT = ('Locate every "{n}" in the image (there may be one, several, or none). '
         'Output JSON like [{{"bbox_2d": [x1, y1, x2, y2], "label": "{n}"}}], or [] if there is none.')
P_ALL = ('Detect all small movable objects in this kitchen image (food, fruit, vegetables, utensils, tools, bottles, '
         'cups, bowls, plates, pans, boxes). Ignore furniture, appliances, the robot. '
         'Output JSON like [{"bbox_2d": [x1, y1, x2, y2], "label": "name"}].')
P_VER = ('This is a close-up crop of a kitchen scene. Is the main object in the centre of this crop a {n}? '
         'Answer Yes or No.')


def parse_boxes(r, W, H):
    out = []
    for m in re.finditer(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]", r):
        b = [float(x) for x in m.groups()]
        b = [b[0] / 1000 * W, b[1] / 1000 * H, b[2] / 1000 * W, b[3] / 1000 * H]
        if b[2] - b[0] > 3 and b[3] - b[1] > 3: out.append(b)
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(u, 1e-6)


def lift(view, box, frac=0.4):
    x0, y0, x1, y1 = box; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * frac / 2, (y1 - y0) * frac / 2
    d = view["depth"].astype(np.float32)[int(cy - h):int(cy + h) + 1, int(cx - w):int(cx + w) + 1]
    z = float(np.median(d)); K, T = view["K"], view["T"]
    return (T @ np.array([(cx - K[0, 2]) * z / K[0, 0], (cy - K[1, 2]) * z / K[1, 1], z, 1]))[:3]


def candidates(g, view, name, generic=True):
    img = view["rgb"]; H, W = img.shape[:2]
    boxes = parse_boxes(g.ask(img, P_TGT.format(n=name), max_new_tokens=256), W, H)
    if generic:
        for b in parse_boxes(g.ask(img, P_ALL, max_new_tokens=768), W, H):
            if all(iou(b, c) < 0.5 for c in boxes): boxes.append(b)
    return boxes


def ground(g, scene, name, generic=True, min_score=0.0):
    """Targeted boxes on both cameras, each verified on a zoomed crop; generic object proposals only as a fallback
    when no targeted candidate is confidently verified. Returns best candidate dict(cam, box, score, point) and all."""
    cands = []
    for use_generic in ((False, True) if generic else (False,)):
        for cam, view in scene["views"].items():
            img = view["rgb"]; H, W = img.shape[:2]
            boxes = parse_boxes(g.ask(img, P_ALL if use_generic else P_TGT.format(n=name), max_new_tokens=512 if use_generic else 256), W, H)
            for b in boxes:
                if any(c["cam"] == cam and iou(b, c["box"]) > 0.5 for c in cands): continue
                s = g.yes_score(zoom(view["rgb"], b, margin=0.6, out=448, min_side=48), P_VER.format(n=name))
                cands.append(dict(cam=cam, box=b, score=s, point=lift(view, b)))
        if cands and max(c["score"] for c in cands) > min_score: break
    best = max(cands, key=lambda c: c["score"]) if cands else None
    return best, cands


def nearest(gt, p, exclude=()):
    d = {k: float(np.linalg.norm(v - p)) for k, v in gt.items() if k not in exclude}
    k = min(d, key=d.get); return k, d[k]


# ---------------------------------------------------------------- instruction parsing + placement grounding
P_PARSE = ('A kitchen robot receives the instruction: "{instr}". Answer in JSON with keys "pick" (the object to grasp, '
           'a short noun phrase; if the instruction is vague like "the toasted item", name the likely object), '
           '"place" (the receptacle or surface to put it in/on, short noun phrase) and "place_is_object" '
           '(true if the receptacle is a movable item like a pan, plate, bowl or container; false for furniture/appliances '
           'like a cabinet, counter, drawer, sink).')
P_POINT = ('Point to an empty spot {prep} the {place} where a {obj} could be placed. '
           'Output JSON like [{{"point_2d": [x, y], "label": "spot"}}].')


def parse(g, instr):
    r = g.ask(np.zeros((64, 64, 3), np.uint8), P_PARSE.format(instr=instr), max_new_tokens=96)
    d = json.loads(re.search(r"\{.*\}", r, re.S).group(0))
    return d["pick"], d["place"], bool(d.get("place_is_object", False))


def point_place(g, scene, place, obj, base_pos, reach=0.95):
    """VLM points at a free spot on a fixture, in each base camera; keep the reachable one closest to the base."""
    prep = "inside" if any(w in place for w in ("cabinet", "drawer", "sink", "microwave", "oven")) else "on"
    out = []
    for cam, view in scene["views"].items():
        img = view["rgb"]; H, W = img.shape[:2]
        r = g.ask(img, P_POINT.format(prep=prep, place=place, obj=obj), max_new_tokens=64)
        m = re.search(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]", r)
        if not m: continue
        u, v = float(m.group(1)) / 1000 * W, float(m.group(2)) / 1000 * H
        p = lift(view, [u - 6, v - 6, u + 6, v + 6], frac=1.0)
        out.append(dict(cam=cam, uv=(u, v), point=p, dist=float(np.linalg.norm((p - base_pos)[:2]))))
    ok = [o for o in out if o["dist"] < reach] or out
    return (min(ok, key=lambda o: o["dist"]) if ok else None), out


def ground_place(g, scene, place, place_is_object, obj, base_pos):
    if place_is_object:
        best, cands = ground(g, scene, place)
        return best, cands
    return point_place(g, scene, place, obj, base_pos)
