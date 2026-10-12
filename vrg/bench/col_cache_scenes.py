"""Stage 1 (colosseum env): reset Colosseum episodes to their stored scenes and cache a 512 px front RGB + world point cloud.

  test : data/test/<task>_<factor>/variation0/episodes/episode{0..n-1}  (perturbed test scenes)
  calib: data/train/<task>/all_variations/episodes/episode{0..n-1}       (unperturbed training scenes, for offsets)
Each cache also stores the stored demo's keyframe gripper poses (k0 = first contact, goal) as ground truth for analysis.

$VGLRUN python vrg/bench/col_cache_scenes.py --mode test --tasks close_box --factors 0,2 --n 10
"""
import argparse, os, pickle, sys, time
import numpy as np
sys.path.insert(0, "/scratch/kaustubh/vlm_reach_grasp/colosseum")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from colosseum_helpers import make_env, keypoints, compat_pickle_load
from pyrep.objects.vision_sensor import VisionSensor
from vrg.bench.col_build_dataset import first_contact, goal_keyframe, TASKS

ROOT = "/scratch/kaustubh/vlm_reach_grasp/colosseum"


def render_front(task_env, res=512, base=128):
    cam = task_env._scene._cam_front
    cam.set_resolution([res, res]); cam.handle_explicitly()
    rgb = (cam.capture_rgb() * 255).astype(np.uint8); depth = cam.capture_depth(in_meters=True)
    pc = VisionSensor.pointcloud_from_depth_and_camera_params(depth, cam.get_matrix(), cam.get_intrinsic_matrix())
    out = dict(rgb=rgb, pc=pc.astype(np.float32), K=cam.get_intrinsic_matrix(), T=cam.get_matrix())
    cam.set_resolution([base, base]); cam.handle_explicitly()
    return out


def ep_dir(mode, task, factor):
    if mode == "calib": return f"{ROOT}/data/train/{task}/all_variations/episodes"
    d = f"{ROOT}/data/test/{task}_{factor}/variation0/episodes"
    return d if os.path.isdir(d) else f"{ROOT}/data/test/{task}_{factor}/episodes"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True, choices=["test", "calib"])
    ap.add_argument("--tasks", default=",".join(TASKS)); ap.add_argument("--factors", default="0"); ap.add_argument("--n", type=int, default=10)
    A = ap.parse_args()
    for task in A.tasks.split(","):
        for factor in ([0] if A.mode == "calib" else [int(f) for f in A.factors.split(",")]):
            d = ep_dir(A.mode, task, factor)
            if not os.path.isdir(d): print("missing", d); continue
            out_dir = f"{ROOT}/scenes/{A.mode}"; os.makedirs(out_dir, exist_ok=True)
            todo = [e for e in range(A.n) if not os.path.exists(f"{out_dir}/{task}_{factor}_{e}.pkl")]
            if not todo: continue
            env, te = make_env(task, factor, image_size=128); te.set_variation(0); te.reset()
            for e in range(A.n):   # in order (RNG alignment with the stored test episodes), skipping done ones
                t0 = time.time()
                demo = compat_pickle_load(f"{d}/episode{e}/low_dim_obs.pkl")
                vf = f"{d}/episode{e}/variation_number.pkl"
                te.set_variation(compat_pickle_load(vf) if os.path.exists(vf) else 0)
                desc, obs = te.reset_to_demo(demo)
                if e not in todo: continue
                fr = render_front(te)
                kps = keypoints(demo); go = np.array([o.gripper_open for o in demo])
                k0, kg = first_contact(kps, go, len(demo)), goal_keyframe(kps, go)
                pickle.dump(dict(task=task, factor=factor, ep=e, mode=A.mode, desc=desc[0], front=fr,
                                 gt_k0=demo[k0].gripper_pose, gt_goal=demo[kg].gripper_pose, gripper0=obs.gripper_pose),
                            open(f"{out_dir}/{task}_{factor}_{e}.pkl", "wb"))
                print(A.mode, task, factor, e, f'"{desc[0]}"', f"{time.time() - t0:.1f}s", flush=True)
            env.shutdown()
