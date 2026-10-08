"""Render agentview RGB-D for each task/init to evaluate grounding offline."""
import numpy as np, pickle
from robosuite.utils import camera_utils as CU
from vrg.common import *
import sys
R = int(sys.argv[1]); out = []
suite = get_suite()
for tid in range(suite.n_tasks):
    env, t = make_env(tid, res=R, depth=True, cams=("agentview",))
    inits = suite.get_task_init_states(tid)
    for ep in range(0, 50, 10):
        env.reset(); obs = env.set_init_state(inits[ep])
        for _ in range(5): obs, *_ = env.step(np.zeros(7))
        objs = {k[:-4]: obs[k] for k in obs if k.endswith("_pos") and not k.startswith("robot") and "_to_" not in k}
        out.append(dict(tid=tid, ep=ep, lang=t.language, img=obs["agentview_image"][::-1].copy(),
                        depth=CU.get_real_depth_map(env.sim, obs["agentview_depth"])[::-1, :, 0].copy(),
                        K=CU.get_camera_intrinsic_matrix(env.sim, "agentview", R, R),
                        T=CU.get_camera_extrinsic_matrix(env.sim, "agentview"), objs=objs))
    env.close(); print(tid, flush=True)
pickle.dump(out, open(f"data/scenes_{R}.pkl", "wb"))
