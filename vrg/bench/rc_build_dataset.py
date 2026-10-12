"""Convert RoboCasa human demos (LeRobot v2.1) for one task into an npz for policy training (robocasa env).

Per frame: 128 px upright images (left, right, wrist), state16 (base pos/quat, eef pos/quat rel. base, gripper qpos),
action12 (env order), GT target-object position (state replay), left-camera extrinsics.
Per episode meta: tg (first sustained close), ts = tg - PRE, te (target lifted LIFT m), tr (final release), offsets.

python -m vrg.bench.rc_build_dataset PickPlaceCounterToCabinet pretrain
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, imageio.v3 as iio, robosuite
from PIL import Image
import robocasa  # noqa: F401
import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.utils.dataset_registry import get_ds_meta
from robosuite.utils import camera_utils as CU
from robosuite.utils.transform_utils import quat2mat

OUT = "/scratch/kaustubh/vlm_reach_grasp/robocasa/train"
CAMS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
PRE, LIFT, RES = 20, 0.04, 128


def frames(root, cam, ep):
    f = root / f"videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4"
    v = iio.imread(f, plugin="pyav")
    return np.stack([np.array(Image.fromarray(x).resize((RES, RES), Image.BILINEAR)) for x in v])


if __name__ == "__main__":
    task, split = sys.argv[1], sys.argv[2]
    root = Path(get_ds_meta(task, split=split, source="human")["path"])
    info = json.load(open(root / "meta/info.json")); n_ep = info["total_episodes"]
    env_meta = LU.get_env_metadata(root); kw = dict(env_meta["env_kwargs"])
    kw.update(env_name=env_meta["env_name"], has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
              renderer="mjviewer", translucent_robot=False)
    env = robosuite.make(**kw)
    import pandas as pd
    D = {k: [] for k in ["left", "right", "wrist", "state", "action", "obj", "T_left", "ep"]}; meta = []; n0 = 0
    for ep in range(n_ep):
        t0 = time.time()
        df = pd.read_parquet(root / f"data/chunk-000/episode_{ep:06d}.parquet")
        state = np.stack(df["observation.state"].values).astype(np.float32)
        act = LU.get_episode_actions(root, ep).astype(np.float32)
        states = LU.get_episode_states(root, ep); xml = LU.get_episode_model_xml(root, ep); em = LU.get_episode_meta(root, ep)
        T = min(len(state), len(act), len(states))
        ims = {c: frames(root, c, ep)[:T] for c in CAMS}; T = min(T, *(len(v) for v in ims.values()))
        reset_to(env, dict(states=states[0], model=xml, ep_meta=json.dumps(em)))
        obj, Tl = [], []
        for t in range(T):
            reset_to(env, dict(states=states[t]))
            obj.append(env.sim.data.body_xpos[env.obj_body_id["obj"]].copy())
            Tl.append(CU.get_camera_extrinsic_matrix(env.sim, CAMS[0]))
        obj = np.array(obj, np.float32); succ = bool(env._check_success())
        g = act[:T, 6]; close = np.where([np.all(g[t:t + 3] > 0) for t in range(T)])[0]
        tg = int(close[0]) if len(close) else T // 2; ts = max(tg - PRE, 0)
        lift = np.where(obj[tg:, 2] > obj[0, 2] + LIFT)[0]; te = tg + int(lift[0]) if len(lift) else min(tg + 25, T - 1)
        opens = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]; tr = int(opens[-1] + 1) if len(opens) else T - 1
        bp, bR = state[:, :3], np.stack([quat2mat(q) for q in state[:, 3:7]])
        eef_w = bp + np.einsum("tij,tj->ti", bR, state[:, 7:10])
        rel = lambda t: bR[t].T @ (eef_w[t] - obj[t])          # eef relative to object, in base frame
        D["left"].append(ims[CAMS[0]][:T]); D["right"].append(ims[CAMS[1]][:T]); D["wrist"].append(ims[CAMS[2]][:T])
        D["state"].append(state[:T]); D["action"].append(act[:T]); D["obj"].append(obj); D["T_left"].append(np.array(Tl, np.float32))
        D["ep"].append(np.full(T, ep))
        meta.append(dict(ep=ep, start=n0, T=T, tg=tg, ts=ts, te=te, tr=tr, success=succ, lang=em["lang"],
                         target=[c["info"]["cat"] for c in em["object_cfgs"] if c["name"] == "obj"][0],
                         off_ts_base=rel(ts).tolist(), off_tg_base=rel(tg).tolist(), quat_ts_base=state[ts, 10:14].tolist(),
                         eef_ts_w=eef_w[ts].tolist(), obj0=obj[0].tolist(), obj_end=obj[-1].tolist(),
                         base_moved=float(np.linalg.norm(bp[:T] - bp[0], axis=1).max())))
        n0 += T
        print(task, ep, T, f"tg={tg} te={te} tr={tr} succ={succ} {time.time() - t0:.1f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{task}_{split}.npz", **{k: np.concatenate(v) for k, v in D.items()})
    json.dump(meta, open(f"{OUT}/{task}_{split}_meta.json", "w"))
    off = np.array([m["off_ts_base"] for m in meta]); print("handoff offset (base frame) mean", off.mean(0).round(3), "std", off.std(0).round(3))
