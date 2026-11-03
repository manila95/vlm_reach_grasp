"""Wrist depth for the RoboCasa grasp segments (robocasa env), from the existing human demos (no new data).

Replays each demo's stored simulator states (jumping straight to the frame, like rc_build_dataset) and renders the wrist
camera depth at 128 px for the frames of the pick skill's slice [ts-1, te+POST]. Output per task, aligned to the task npz:
  $S/robocasa/train/<task>_<split>_wdepth.npz   gidx (n,) global frame index into <task>_<split>.npz, depth (n,128,128) float16 [m]
python -m vrg.bench.rc_render_depth PickPlaceCounterToCabinet pretrain
"""
import json, sys
from pathlib import Path
import numpy as np, robosuite
import robocasa  # noqa: F401
import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.utils.dataset_registry import get_ds_meta
from robosuite.utils import camera_utils as CU

OUT = "/scratch/kaustubh/vlm_reach_grasp/robocasa/train"; POST = 10

if __name__ == "__main__":
    task, split = sys.argv[1], sys.argv[2]
    root = Path(get_ds_meta(task, split=split, source="human")["path"]); meta = json.load(open(f"{OUT}/{task}_{split}_meta.json"))
    env_meta = LU.get_env_metadata(root); kw = dict(env_meta["env_kwargs"])
    kw.update(env_name=env_meta["env_name"], has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False, renderer="mjviewer", translucent_robot=False)
    env = robosuite.make(**kw); G, D = [], []
    for m in meta:
        ep = m["ep"]; states = LU.get_episode_states(root, ep); xml = LU.get_episode_model_xml(root, ep); em = LU.get_episode_meta(root, ep)
        reset_to(env, dict(states=states[0], model=xml, ep_meta=json.dumps(em)))
        for t in range(max(m["ts"] - 1, 0), min(m["te"] + POST, m["T"] - 1) + 1):
            reset_to(env, dict(states=states[t]))
            _, d = env.sim.render(width=128, height=128, camera_name="robot0_eye_in_hand", depth=True)
            G.append(m["start"] + t); D.append(CU.get_real_depth_map(env.sim, d)[::-1, :, 0].astype(np.float16) if d.ndim == 3 else CU.get_real_depth_map(env.sim, d)[::-1].astype(np.float16))
        print(task, ep, len(G), flush=True)
    np.savez(f"{OUT}/{task}_{split}_wdepth.npz", gidx=np.array(G), depth=np.stack(D)); print("DONE", task, len(G), np.stack(D).shape, float(np.median(np.stack(D))))
