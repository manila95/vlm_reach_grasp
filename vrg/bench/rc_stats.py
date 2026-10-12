"""Per-task handoff statistics from RoboCasa demos (robocasa env) -> robocasa/handoff_stats.json.

All offsets are in the robot BASE frame, relative to the target object's position at the given time:
  pick_off  = eef(ts) - obj(ts)      grasp_off = eef(tg) - obj(tg)     lift_off = eef(te) - obj(0)
  place_off = eef(tr-PRE_PLACE) - obj(tr)                             pick_off40 = eef(tg-40) - obj(tg-40) (for --pre 40)                             release_off = eef(tr) - obj(tr)
Quaternions (xyzw, base frame) are sign-aligned and averaged.
"""
import json, numpy as np
from robosuite.utils.transform_utils import quat2mat
from vrg.bench.robocasa_env import TASKS
from vrg.bench.rc_train import ROOT, PRE_PLACE


def qmean(qs):
    qs = np.array(qs); qs = qs * np.sign(qs @ qs[0])[:, None]; m = qs.mean(0); return m / np.linalg.norm(m)


out = {}
for task in TASKS:
    Z = np.load(f"{ROOT}/train/{task}_pretrain.npz"); meta = [m for m in json.load(open(f"{ROOT}/train/{task}_pretrain_meta.json")) if m["success"]]
    st, obj = Z["state"], Z["obj"]
    R = np.stack([quat2mat(q) for q in st[:, 3:7]]); bp = st[:, :3]; eef = bp + np.einsum("tij,tj->ti", R, st[:, 7:10])
    rel = lambda t, ref: R[t].T @ (eef[t] - ref)
    rows = {k: [] for k in ["pick_off", "grasp_off", "lift_off", "place_off", "release_off", "pick_quat", "place_quat", "release_base",
                            "pick_off40", "pick_quat40"]}
    for m in meta:
        s = m["start"]; ts, tg, te, tr = (s + m[k] for k in ("ts", "tg", "te", "tr")); tp = max(tr - PRE_PLACE, s)
        rows["pick_off"].append(rel(ts, obj[ts])); rows["grasp_off"].append(rel(tg, obj[tg])); rows["lift_off"].append(rel(te, obj[s]))
        rows["place_off"].append(rel(tp, obj[tr])); rows["release_off"].append(rel(tr, obj[tr]))
        rows["pick_quat"].append(st[ts, 10:14]); rows["place_quat"].append(st[tp, 10:14])
        t40 = max(tg - 40, s); rows["pick_off40"].append(rel(t40, obj[t40])); rows["pick_quat40"].append(st[t40, 10:14])  # local_pick --pre 40
        rows["release_base"].append(R[tr].T @ (obj[tr] - bp[tr]))   # where the object ends up, relative to the base
    d = {k: (qmean(v) if "quat" in k else np.mean(v, 0)).tolist() for k, v in rows.items()}
    out[task] = d
    sd = {k: np.round(100 * np.std(v, 0), 1).tolist() for k, v in rows.items() if "quat" not in k}
    print(task, f"({len(meta)} successful demos)")
    for k in ["pick_off", "pick_off40", "place_off", "release_base"]: print(f"   {k:12s} mean(cm) {np.round(100*np.array(d[k]),1).tolist()}  std(cm) {sd[k]}")
json.dump(out, open(f"{ROOT}/handoff_stats.json", "w"), indent=1)
