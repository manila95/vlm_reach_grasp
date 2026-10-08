"""Offline eval of propose(depth)+verify(VLM) grounding on cached 1024 scenes."""
import pickle, sys, numpy as np
from vrg.vlm import Grounder, ground_propose_verify, propose
from vrg.common import target_name
S = pickle.load(open("data/scenes_1024.pkl", "rb"))
g = Grounder(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-8B-Instruct")
ok, per, errs, nprop = 0, {}, [], []
for s in S:
    name = target_name(s["lang"]); best, props = ground_propose_verify(g, s["img"], s["depth"], s["K"], s["T"], name.replace("_", " "))
    nprop.append(len(props))
    cands = {k: v for k, v in s["objs"].items()}
    near = min(cands, key=lambda k: np.linalg.norm(cands[k][:2] - best["xy"]))
    hit = near == name + "_1"; ok += hit; per.setdefault(name, []).append(hit)
    errs.append(best["xy"] - s["objs"][name + "_1"][:2])
    gt_ok = sum(min(props, key=lambda p: np.linalg.norm(p["xy"] - v[:2])) is not None for v in cands.values())
    if not hit: print("  miss", s["tid"], s["ep"], name, "->", near, "scores", sorted([round(p["score"], 1) for p in props], reverse=True))
print(f"propose+verify acc={ok}/{len(S)} per-task", {k: sum(v) for k, v in per.items()}, "n_props", np.mean(nprop),
      "xy err mean", np.mean(errs, 0).round(3), "abs", np.median(np.linalg.norm(errs, axis=1)).round(3))
