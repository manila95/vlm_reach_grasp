"""Draw each demo's own keypoint path (start, pre-grasp, grasp, lift, release) on all of its agentview frames."""
import json, numpy as np
from vrg.common import *
from vrg.paths import camera_mats, project, draw_path
Z = np.load("data/libero_object.npz"); meta = json.load(open("data/libero_object_meta.json"))
cams = {}
for tid in range(10):
    env, _ = make_env(tid); env.reset(); cams[tid] = camera_mats(env); env.close()
K0, T0 = cams[0]
print("camera identical across tasks:", all(np.allclose(cams[t][1], T0, atol=1e-6) and np.allclose(cams[t][0], K0) for t in cams))
agent = Z["agent"]; out = np.empty_like(agent); ee = Z["proprio"][:, :3]
for i, m in enumerate(meta):
    s = m["start"]; g = Z["action"][s:s + m["T"], 6]
    opens = np.where((g[:-1] > 0) & (g[1:] < 0))[0]
    m["tr_last"] = int(opens[-1] + 1) if len(opens) else m["T"] - 1   # final release (demos sometimes re-grasp)
    kp = np.stack([ee[s], ee[s + m["ts"]], ee[s + m["tg"]], ee[s + m["te"]], ee[s + m["tr_last"]]])
    uv = project(kp, *cams[m["tid"]])
    for t in range(m["T"]): out[s + t] = draw_path(agent[s + t], uv)
    if i % 100 == 0: print(i, flush=True)
np.save("data/agent_path.npy", out)
json.dump(meta, open("data/libero_object_meta.json", "w"))
print("demos with early re-open:", sum(m["tr"] != m["tr_last"] for m in meta))
np.save("data/agentview_cam.npy", np.stack([np.pad(K0, ((0, 1), (0, 1))), T0]))
from PIL import Image
Image.fromarray(np.concatenate([out[meta[k]["start"]][::-1] for k in (0, 120, 260, 400)], 1)).resize((1024, 256)).save("logs/hamster_paths.png")
