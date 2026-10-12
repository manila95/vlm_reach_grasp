"""Offline grounding accuracy on cached RoboCasa scenes (vlmrg env)."""
import glob, pickle, sys, collections, numpy as np
from vrg.vlm import Grounder
from vrg.bench.rc_ground import ground, nearest
files = sorted(glob.glob("/scratch/kaustubh/vlm_reach_grasp/robocasa/scenes/*/*.pkl"))
generic = "--no_generic" not in sys.argv
g = Grounder(); res = collections.defaultdict(list)
for f in files:
    s = pickle.load(open(f, "rb")); name = s["info"]["target_lang"]
    best, cands = ground(g, s, name, generic)
    if best is None: hit, err, k = False, np.nan, None
    else: k, _ = nearest(s["gt"], best["point"]); err = float(np.linalg.norm(best["point"] - s["gt"]["obj"])); hit = k == "obj"
    res[(s["split"], s["task"])].append(hit)
    print(f"{s['split']:8s} {s['task']:28s} ep{s['ep']} {name:16s} hit={hit} err={100*err:5.1f}cm nearest={k} cands={len(cands)} best={best and round(best['score'],1)} cam={best and best['cam'][13:]}", flush=True)
for (sp, t), v in sorted(res.items()): print(f"{sp:8s} {t:28s} {sum(v)}/{len(v)}")
for sp in ("pretrain", "target"): v = sum((x for (s_, t), x in res.items() if s_ == sp), []); print(sp, "overall", f"{sum(v)}/{len(v)}")
