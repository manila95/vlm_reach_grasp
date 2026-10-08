import h5py, numpy as np, sys
from vrg.common import *
suite = get_suite()
rows = []
for tid in range(suite.n_tasks):
    env, t = make_env(tid)
    obj = target_name(t.language) + "_1"
    f = h5py.File(demo_file(t.language))
    for k in f["data"]:
        e = f["data"][k]
        st = e["states"][:]
        env.reset(); obs = env.set_init_state(st[0])
        op = obs[obj + "_pos"]
        ee = e["obs/ee_pos"][:]
        g = e["actions"][:, 6]
        tg = int(np.argmax(g > 0))
        # object pose at grasp time
        obs_g = env.set_init_state(st[tg])
        rows.append(dict(tid=tid, T=len(g), tg=tg, ee0_err=np.linalg.norm(obs["robot0_eef_pos"]-ee[0]),
                         off_g=ee[tg]-obs_g[obj+"_pos"], off_m20=ee[max(tg-20,0)]-op, obj_moved=np.linalg.norm(obs_g[obj+"_pos"]-op)))
    env.close()
    print(tid, t.language, flush=True)
import pickle; pickle.dump(rows, open("data/demo_stats.pkl","wb"))
a = lambda k: np.array([r[k] for r in rows])
print("ee0 consistency err max", a("ee0_err").max())
print("T", a("T").mean(), "tg mean/min/max", a("tg").mean(), a("tg").min(), a("tg").max())
print("offset at grasp mean", a("off_g").mean(0), "std", a("off_g").std(0))
print("offset 20 steps before grasp mean", a("off_m20").mean(0), "std", a("off_m20").std(0))
print("obj moved before grasp max", a("obj_moved").max())
