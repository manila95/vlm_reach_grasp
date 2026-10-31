"""MimicGen dataset cache: hdf5 -> flat memory-mapped arrays for fast training (any env with h5py + numpy).

<cache>/<dataset>/{agent,wrist}.npy (N,84,84,3) uint8; proprio.npy (N,9) = eef pos (3), eef quat (4), gripper qpos (2);
action.npy (N,7) OSC_POSE deltas; demo.npy (N,) demo index; t.npy (N,) step in demo; index.json = demo names, offsets, lengths.
  python -m vrg.mg.data --datasets square_d0,coffee_d0
"""
import argparse, json, os
import numpy as np

S = "/scratch/kaustubh/vlm_reach_grasp/mimicgen"


def build(name, cache=f"{S}/cache"):
    import h5py
    out = f"{cache}/{name}"; os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/index.json"): return out
    f = h5py.File(f"{S}/data/core/{name}.hdf5", "r")
    demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1])); T = [f[f"data/{d}/actions"].shape[0] for d in demos]; N = sum(T)
    arr = {k: np.lib.format.open_memmap(f"{out}/{k}.npy", "w+", dtype=dt, shape=sh) for k, dt, sh in
           [("agent", np.uint8, (N, 84, 84, 3)), ("wrist", np.uint8, (N, 84, 84, 3)), ("proprio", np.float32, (N, 9)),
            ("action", np.float32, (N, 7)), ("demo", np.int32, (N,)), ("t", np.int32, (N,))]}
    o = 0
    for i, (d, n) in enumerate(zip(demos, T)):
        g = f[f"data/{d}"]; ob = g["obs"]
        arr["agent"][o:o + n] = ob["agentview_image"][()]; arr["wrist"][o:o + n] = ob["robot0_eye_in_hand_image"][()]
        arr["proprio"][o:o + n] = np.concatenate([ob["robot0_eef_pos"][()], ob["robot0_eef_quat"][()], ob["robot0_gripper_qpos"][()]], 1)
        arr["action"][o:o + n] = g["actions"][()]; arr["demo"][o:o + n] = i; arr["t"][o:o + n] = np.arange(n); o += n
    for a in arr.values(): a.flush()
    json.dump(dict(dataset=name, demos=demos, T=T, offsets=np.cumsum([0] + T[:-1]).tolist(), N=N), open(f"{out}/index.json", "w"))
    print("cached", name, N, "frames", flush=True); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", required=True); A = ap.parse_args()
    for n in A.datasets.split(","): build(n)
