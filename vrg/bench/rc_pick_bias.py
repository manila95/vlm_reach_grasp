"""Pick-point bias of a RoboCasa grounding file, in the robot base frame, from the DEV episodes only (pretrain eps 50-59).

The VLM's 3D pick point is the median depth of the box centre: a point on the camera-facing surface, offset from the
object's centre. A scripted grasp needs that offset removed (the learned skill absorbed it). Per task: mean over the
dev episodes whose pick was correct (within 6 cm) of R_base^T (pick - object centre). Output: json {task: [bx, by, bz]}.
  $PY -m vrg.bench.rc_pick_bias <grounding_dev.pkl> <scenes_dev dir> <out.json>
"""
import glob, json, pickle, sys, collections
import numpy as np

G = pickle.load(open(sys.argv[1], "rb")); B = collections.defaultdict(list)
for (split, task, ep), g in G.items():
    if g.get("pick") is None: continue
    S = pickle.load(open(f"{sys.argv[2]}/{split}/{task}_{ep}.pkl", "rb"))
    d = S["base_R"].T @ (g["pick"] - S["gt"]["obj"]); 
    if np.linalg.norm(g["pick"] - S["gt"]["obj"]) < 0.06: B[task].append(d)
out = {t: np.mean(v, 0).tolist() for t, v in B.items()}; out["_all"] = np.mean([x for v in B.values() for x in v], 0).tolist()
out["_n"] = {t: len(v) for t, v in B.items()}
json.dump(out, open(sys.argv[3], "w"), indent=1); print({k: (np.round(v, 3).tolist() if k != "_n" else v) for k, v in out.items()})
