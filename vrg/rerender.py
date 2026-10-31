"""Render-time domain randomisation data for the local skill (docs/PLAN_perceptual_robustness.md, S3, and S5 depth).

Replays every LIBERO-Object demo's simulator states over the local skill's training slice (handoff - 1 .. lift + post)
and re-renders the 128 px wrist camera:
  * K randomised variants per demo (vrg/perturb.py kinds light / texture / both / both; one draw per demo and variant,
    seeds 10_000_019 * variant + demo, disjoint from the dev-set seeds 104729 * tid + ep + 1),
  * clean metric wrist depth (float16, metres) for the depth-input variant.
Actions and proprio are unchanged, so labels stay exact. A clean re-render is compared with the dataset image (sanity).
Output: <out> {idx (n,), wrist_dr (K, n, 128, 128, 3) uint8, depth (n, 128, 128) float16, kinds}
  LIBERO_CONFIG_PATH=$S/libero_std_cfg $PY -m vrg.rerender
"""
import argparse, json
import numpy as np

KINDS = ("light", "texture", "both", "both"); CAM = "robot0_eye_in_hand"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--post", type=int, default=6); ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--out", default="/scratch/kaustubh/vlm_reach_grasp/skills/rerender.npz"); A = ap.parse_args()
    import h5py
    from robosuite.utils.camera_utils import get_real_depth_map
    from vrg.common import get_suite, make_env, demo_file
    from vrg.perturb import perturb_scene, restore_scene
    meta = json.load(open("data/libero_object_meta.json")); Z = np.load("data/libero_object.npz", mmap_mode="r")
    suite = get_suite(); idx, dr, dep, diffs = [], [[] for _ in KINDS], [], []
    for tid in range(suite.n_tasks):
        env, t = make_env(tid); env.reset(); f = h5py.File(demo_file(t.language))
        demos = sorted(f["data"], key=lambda s: int(s.split("_")[1])); ms = [(i, m) for i, m in enumerate(meta) if m["tid"] == tid]
        for k, (demo_id, m) in zip(demos, ms):
            st = f["data"][k]["states"][:]; lo, hi = max(m["ts"] - A.pad, 0), min(m["te"] + A.post, m["T"] - 1)
            gidx = m["start"] + np.arange(lo, hi + 1)
            for v, kind in enumerate((None,) + KINDS):
                if kind is None: restore_scene(env)
                else: perturb_scene(env, kind, 10_000_019 * v + demo_id)
                frames = []
                for tt in range(lo, hi + 1):
                    env.sim.set_state_from_flattened(st[tt]); env.sim.forward()
                    if kind is None:
                        rgb, d = env.sim.render(width=128, height=128, camera_name=CAM, depth=True)
                        frames.append(rgb); dep.append(get_real_depth_map(env.sim, d).astype(np.float16))
                    else:
                        frames.append(env.sim.render(width=128, height=128, camera_name=CAM))
                if kind is None: diffs.append(float(np.abs(np.stack(frames).astype(int) - Z["wrist"][gidx].astype(int)).mean()))
                else: dr[v - 1].append(np.stack(frames))
            restore_scene(env); idx.append(gidx)
        env.close(); print(tid, t.language, "clean re-render vs dataset: mean abs diff", round(np.mean(diffs[-len(ms):]), 2), flush=True)
    np.savez(A.out, idx=np.concatenate(idx), wrist_dr=np.stack([np.concatenate(x) for x in dr]), depth=np.concatenate(dep), kinds=np.array(KINDS))
    print("RERENDER DONE", len(np.concatenate(idx)), "frames; clean diff mean", np.mean(diffs), flush=True)


if __name__ == "__main__":
    main()
