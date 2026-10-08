"""Convert LIBERO-Object hdf5 demos into one npz with phase labels.

Per demo: tg = first gripper-close step; ts = tg - PRE (handoff, EEF ~15cm above object);
te = first step after tg with object lifted LIFT m above its rest height; tr = first re-open (release).
"""
import h5py, numpy as np, json
from vrg.common import *
PRE, LIFT = 20, 0.10
suite = get_suite()
D = {k: [] for k in ["agent", "wrist", "proprio", "action", "tid", "demo"]}
meta = []
for tid in range(suite.n_tasks):
    env, t = make_env(tid)
    obj = target_name(t.language) + "_1"
    env.reset()
    jn = env.env.objects_dict[obj].joints[0]
    qa = env.sim.model.get_joint_qpos_addr(jn)[0]
    bq = env.sim.model.get_joint_qpos_addr(env.env.objects_dict["basket_1"].joints[0])[0]
    f = h5py.File(demo_file(t.language))
    for di, k in enumerate(sorted(f["data"], key=lambda s: int(s.split("_")[1]))):
        e = f["data"][k]; st = e["states"][:]; a = e["actions"][:]
        if di == 0:  # check image convention and state layout against a live render
            obs = env.set_init_state(st[0])
            print("img diff", np.abs(obs["agentview_image"].astype(int) - e["obs/agentview_rgb"][0]).mean(),
                  "flipped diff", np.abs(obs["agentview_image"][::-1].astype(int) - e["obs/agentview_rgb"][0]).mean(),
                  "obj pos err", np.abs(st[0, 1+qa:1+qa+3] - obs[obj+"_pos"]).max())
        oz = st[:, 1+qa+2]; ee = e["obs/ee_pos"][:]
        tg = int(np.argmax(a[:, 6] > 0)); ts = max(tg - PRE, 0)
        lifted = np.where(oz[tg:] > oz[0] + LIFT)[0]; te = tg + int(lifted[0]) if len(lifted) else len(a) - 1
        reo = np.where(a[tg:, 6] < 0)[0]; tr = tg + int(reo[0]) if len(reo) else len(a) - 1
        prop = np.concatenate([ee, e["obs/ee_ori"][:], e["obs/gripper_states"][:]], 1)
        n0 = sum(len(x) for x in D["action"])
        D["agent"].append(e["obs/agentview_rgb"][:]); D["wrist"].append(e["obs/eye_in_hand_rgb"][:])
        D["proprio"].append(prop.astype(np.float32)); D["action"].append(a.astype(np.float32))
        D["tid"].append(np.full(len(a), tid)); D["demo"].append(np.full(len(a), len(meta)))
        meta.append(dict(tid=tid, start=n0, T=len(a), tg=tg, ts=ts, te=te, tr=tr,
                         obj0=st[0, 1+qa:1+qa+3].tolist(), basket0=st[0, 1+bq:1+bq+3].tolist(),
                         ee_ts=ee[ts].tolist(), ori_ts=e["obs/ee_ori"][ts].tolist(), ori_0=e["obs/ee_ori"][0].tolist(),
                         ee_tr=ee[tr].tolist(), ee_te=ee[te].tolist(), basket_tr=st[tr, 1+bq:1+bq+3].tolist()))
    env.close(); print(tid, t.language, flush=True)
np.savez("data/libero_object.npz", **{k: np.concatenate(v) for k, v in D.items()})
json.dump(meta, open("data/libero_object_meta.json", "w"))
M = lambda k: np.array([m[k] for m in meta])
print("handoff offset ee_ts - obj0: mean", (M("ee_ts")-M("obj0")).mean(0), "std", (M("ee_ts")-M("obj0")).std(0))
print("ori_ts mean", M("ori_ts").mean(0), "std", M("ori_ts").std(0), "| ori_0 mean", M("ori_0").mean(0))
print("te - tg", (M("te")-M("tg")).mean(), (M("te")-M("tg")).max(), " lift failed:", (M("te") == M("T")-1).sum())
print("release offset ee_tr - basket: mean", (M("ee_tr")-M("basket_tr")).mean(0), "std", (M("ee_tr")-M("basket_tr")).std(0))
print("ee_te - obj0", (M("ee_te")-M("obj0")).mean(0), "T-tr", (M("T")-M("tr")).mean())
