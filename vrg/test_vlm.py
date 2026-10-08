import numpy as np, sys
from PIL import Image, ImageDraw
from vrg.common import *
from vrg.vlm import Grounder, lift
R = 512
g = Grounder()
suite = get_suite()
print(g.parse_target(suite.get_task(3).language))
errs = []
for tid in range(suite.n_tasks):
    env, t = make_env(tid, res=R, depth=True, cams=("agentview",))
    name = target_name(t.language); inits = suite.get_task_init_states(tid)
    for ep in range(3):
        env.reset(); obs = env.set_init_state(inits[ep])
        for _ in range(5): obs, *_ = env.step(np.zeros(7))
        img = obs["agentview_image"][::-1].copy()
        for q, key in [(name.replace("_", " "), name + "_1"), ("basket", "basket_1")]:
            uv, raw = g.point(img, q)
            if uv is None: print("PARSE FAIL", raw); continue
            p = lift(env.sim, "agentview", obs["agentview_depth"], uv, R, R)
            gt = obs[key + "_pos"]
            errs.append((tid, q, *(p - gt)))
            print(f"{tid} {ep} {q:18s} uv=({uv[0]:.0f},{uv[1]:.0f}) err={np.round(p-gt,3)} |xy|={np.linalg.norm((p-gt)[:2]):.3f}", flush=True)
            if ep == 0:
                im = Image.fromarray(img); d = ImageDraw.Draw(im); d.ellipse([uv[0]-6, uv[1]-6, uv[0]+6, uv[1]+6], outline=(255,0,0), width=3)
                im.save(f"logs/vlm_{tid}_{q.replace(' ','_')}.png")
    env.close()
e = np.array([x[2:] for x in errs]); print("mean", e.mean(0), "std", e.std(0))
import pickle; pickle.dump(errs, open("logs/vlm_errs.pkl", "wb"))
