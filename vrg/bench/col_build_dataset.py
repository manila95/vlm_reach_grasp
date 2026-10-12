"""Convert Colosseum training demos (PerAct format, 100/task, no perturbations) into npz files (colosseum env, py3.8).

Per frame: front/wrist RGB (128 px), gripper pose (xyz + quat xyzw), gripper_open, ignore_collisions.
Per episode: keyframes (PerAct heuristic), first contact keyframe k0, goal keyframe (release after the first grasp, else
the final keyframe), front camera K/T and frame-0 front depth (for grounding calibration), instruction.

$VGLRUN python vrg/bench/col_build_dataset.py <task>      (no rendering needed; plain python is fine)
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, "/scratch/kaustubh/vlm_reach_grasp/colosseum")
from colosseum_helpers import load_demo, keypoints

ROOT = "/scratch/kaustubh/vlm_reach_grasp/colosseum"
TASKS = ["basketball_in_hoop", "close_box", "close_laptop_lid", "get_ice_from_fridge", "meat_on_grill", "move_hanger",
         "open_drawer", "reach_and_drag", "put_money_in_safe", "place_wine_at_rack_location", "turn_oven_on", "scoop_with_spatula"]


def first_contact(kps, gopen, T):
    """First keyframe that is a real stop (skip trivial keyframes in the first 10 steps)."""
    for k in kps:
        if k > 10: return k
    return kps[0]


def goal_keyframe(kps, gopen):
    closed = [k for k in kps if gopen[k] < 0.5]
    if closed:
        after = [k for k in kps if k > closed[0] and gopen[k] > 0.5]
        if after: return after[0]
    return kps[-1]


if __name__ == "__main__":
    task = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    ep_root = f"{ROOT}/data/train/{task}/all_variations/episodes"
    D = {k: [] for k in ["front", "wrist", "pose", "open", "ignore"]}; meta = []; n0 = 0
    for e in range(n):
        t0 = time.time()
        d = load_demo(f"{ep_root}/episode{e}", cameras=("front", "wrist"))
        T = len(d["gripper_pose"]); kps = keypoints(d["demo"])
        D["front"].append(d["front"]["rgb"]); D["wrist"].append(d["wrist"]["rgb"])
        D["pose"].append(d["gripper_pose"].astype(np.float32)); D["open"].append(d["gripper_open"]); D["ignore"].append(d["ignore_collisions"])
        k0 = first_contact(kps, d["gripper_open"], T); kg = goal_keyframe(kps, d["gripper_open"])
        np.savez_compressed(f"{ROOT}/train_npz/{task}_ep{e}_frame0.npz", depth=d["front"]["depth_m"][0].astype(np.float16),
                            K=d["front"]["intrinsics"][0], T=d["front"]["extrinsics"][0], rgb=d["front"]["rgb"][0])
        desc = d.get("variation_descriptions", ["?"])
        meta.append(dict(ep=e, start=n0, T=T, kps=[int(k) for k in kps], k0=int(k0), kgoal=int(kg),
                         variation=int(d.get("variation_number", 0)), desc=desc[0] if isinstance(desc, list) else str(desc),
                         pose_k0=d["gripper_pose"][k0].tolist(), pose_goal=d["gripper_pose"][kg].tolist(),
                         front_T=d["front"]["extrinsics"][0].tolist(), front_K=d["front"]["intrinsics"][0].tolist()))
        n0 += T
        print(task, e, T, "kps", len(kps), f"k0={k0} goal={kg}", f"{time.time() - t0:.1f}s", flush=True)
    np.savez(f"{ROOT}/train_npz/{task}.npz", **{k: np.concatenate(v) for k, v in D.items()})
    json.dump(meta, open(f"{ROOT}/train_npz/{task}_meta.json", "w"))
