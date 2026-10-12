"""Calibration offsets for the Colosseum pipeline (vlmrg env) -> colosseum/col_stats.json + grounding.json (py3.8-safe).

off_k0   = mean over CALIBRATION (training) scenes of (demo first-contact gripper position - VLM reach point)
off_goal = mean over calibration scenes of (demo goal gripper position - VLM destination point, or reach point if none)
pre      = mean over training demos of (gripper position W_PRE frames before k0 - position at k0)   [world]
quat_pre = mean gripper orientation W_PRE frames before k0 (sign-aligned)
"""
import json, pickle, numpy as np
from vrg.bench.col_train import ROOT, TASKS, W_PRE
G = pickle.load(open(f"{ROOT}/grounding.pkl", "rb")); out = {}
for t in TASKS:
    cal = [r for k, r in G.items() if k[0] == "calib" and k[1] == t and r["reach"] is not None]
    Z = np.load(f"{ROOT}/train_npz/{t}.npz"); meta = json.load(open(f"{ROOT}/train_npz/{t}_meta.json"))
    pre = np.mean([Z["pose"][m["start"] + max(m["k0"] - W_PRE, 0), :3] - Z["pose"][m["start"] + m["k0"], :3] for m in meta], 0)
    qs = np.array([Z["pose"][m["start"] + max(m["k0"] - W_PRE, 0), 3:] for m in meta]); qs *= np.sign(qs @ qs[0])[:, None]; q = qs.mean(0); q /= np.linalg.norm(q)
    off_k0 = np.mean([r["gt_k0"][:3] - r["reach"] for r in cal], 0) if cal else np.zeros(3)
    off_goal = np.mean([r["gt_goal"][:3] - (r["dest"] if r["dest"] is not None else r["reach"]) for r in cal], 0) if cal else np.zeros(3)
    out[t] = dict(off_k0=off_k0.tolist(), off_goal=off_goal.tolist(), pre=pre.tolist(), quat_pre=q.tolist(), n_calib=len(cal),
                  k0_resid_cm=float(100 * np.mean([np.linalg.norm(r["gt_k0"][:3] - r["reach"] - off_k0) for r in cal])) if cal else None)
    print(t, {k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in out[t].items()})
json.dump(out, open(f"{ROOT}/col_stats.json", "w"), indent=1)
J = {"|".join(map(str, k)): {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in r.items()} for k, r in G.items()}
json.dump(J, open(f"{ROOT}/grounding.json", "w"))
