"""Ground-truth-only grounding cache (for --oracle runs before/without the VLM pass)."""
import pickle
from vrg.common import *
from vrg.pipeline import reset_episode
from vrg.ground_episodes import CALIB_EPS, EVAL_EPS
res = {}
for tid in range(10):
    env, t = make_env(tid)
    for ep in CALIB_EPS + EVAL_EPS:
        obs = reset_episode(env, tid, ep)
        res[(tid, ep)] = dict(tid=tid, ep=ep, calib=ep in CALIB_EPS, gt_obj=obs[target_name(t.language) + "_1_pos"].copy(),
                              gt_basket=obs["basket_1_pos"].copy(), obj=None, basket=None, obj_hit=False)
    env.close(); print(tid, flush=True)
pickle.dump(res, open("data/grounding_gt.pkl", "wb"))
