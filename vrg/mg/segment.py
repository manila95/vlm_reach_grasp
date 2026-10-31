"""Contact-phase segmentation of MimicGen demos (mimicgen env: source /scratch/kaustubh/vlm_reach_grasp/mimicgen/env.sh).

A task is a sequence of object-centric subtasks, as in MimicGen's own task specs (exps/templates/robosuite/<task>.json):
each has a reference object and ends at the first 0->1 of its termination signal (the last ends with the demo). Inside
subtask k, the CONTACT PHASE starts at the first physical contact between the robot side (gripper, plus the object it
holds) and the reference object, and runs to the subtask end. The learned policy starts `pre` steps before contact (the
handoff); everything before that is free-space motion, done by the VLM-grounded reach at test time.

Demos are replayed from their recorded MuJoCo states (exact, no actions), so the segmentation is ground truth.
Output per dataset: <out>/<dataset>.json with, per demo, a list of subtasks
  dict(ref, held, start, contact, end, T_ref_eef_at_contact (4x4), eef_at_contact, ref_pose_at_contact)
and the handoff pose for any `pre` can be read from the stored per-step EE/ref poses (<out>/<dataset>_poses.npz).

  $MGPY -m vrg.mg.segment --datasets square_d0,coffee_d0 --out $S/mimicgen/segments
"""
import argparse, json, os, sys
import numpy as np

S = "/scratch/kaustubh/vlm_reach_grasp/mimicgen"
TASKS = {  # task -> MimicGen interface class, ref object -> root-body accessor
    "square": ("MG_Square", {"square_nut": lambda e: e.nuts[e.nut_to_id["square"]].root_body, "square_peg": lambda e: "peg1"}),
    "threading": ("MG_Threading", {"needle": lambda e: e.needle.root_body, "tripod": lambda e: e.tripod.root_body}),
    "coffee": ("MG_Coffee", {"coffee_pod": lambda e: e.coffee_pod.root_body, "coffee_machine": lambda e: e.coffee_machine.root_body}),
    "three_piece_assembly": ("MG_ThreePieceAssembly", {"base": lambda e: e.base.root_body, "piece_1": lambda e: e.piece_1.root_body,
                                                       "piece_2": lambda e: e.piece_2.root_body}),
    "mug_cleanup": ("MG_MugCleanup", {"object": lambda e: e.cleanup_object.root_body, "drawer": lambda e: e.drawer.root_body}),
}
GRASP_SIGNALS = ("grasp", "grasp_1", "grasp_2")


def task_spec(task):
    spec = json.load(open(f"{S}/mimicgen/mimicgen/exps/templates/robosuite/{task}.json"))["task"]["task_spec"]
    return [(v["object_ref"], v["subtask_term_signal"]) for k, v in sorted(spec.items(), key=lambda kv: int(kv[0].split("_")[1]))]


def subtree(model, root_name):
    root = model.body_name2id(root_name); ids = {root}; changed = True
    while changed:
        changed = False
        for b in range(model.nbody):
            if b not in ids and model.body_parentid[b] in ids: ids.add(b); changed = True
    return ids


def make_env(path):
    import robomimic.utils.env_utils as EnvUtils, robomimic.utils.file_utils as FileUtils, robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"], "rgb": []}})
    em = FileUtils.get_env_metadata_from_dataset(path)
    return EnvUtils.create_env_from_metadata(env_meta=em, render=False, render_offscreen=False, use_image_obs=False)


def segment_dataset(name, out, max_demos=None):
    import h5py
    import mimicgen.env_interfaces.robosuite as MGI
    task = name.rsplit("_d", 1)[0]; cls, refs = TASKS[task]; spec = task_spec(task)
    path = f"{S}/data/core/{name}.hdf5"; env = make_env(path); e = env.env; m = e.sim.model
    iface = getattr(MGI, cls)(env=e)
    f = h5py.File(path, "r"); demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))[:max_demos]
    gripper_bodies = subtree(m, e.robots[0].gripper.root_body)
    ref_bodies = {r: subtree(m, refs[r](e)) for r in refs}
    geom_body = m.geom_bodyid
    out_rec, poses = {}, {}
    for di, dk in enumerate(demos):
        g = f[f"data/{dk}"]; st = g["states"][()]; T = len(st)
        env.reset(); env.reset_to({"states": st[0], "model": g.attrs["model_file"]})
        sig = {s: np.zeros(T, int) for _, s in spec if s}; eef = np.zeros((T, 4, 4)); objp = {r: np.zeros((T, 4, 4)) for r in refs}
        touch = {r: np.zeros(T, bool) for r in refs}; held_touch = {(h, r): np.zeros(T, bool) for h in refs for r in refs if h != r}
        for t in range(T):
            env.reset_to({"states": st[t]})
            for s, v in iface.get_subtask_term_signals().items():
                if s in sig: sig[s][t] = v
            eef[t] = iface.get_robot_eef_pose()
            for r, P in iface.get_object_poses().items(): objp[r][t] = P
            d = e.sim.data
            for c in range(d.ncon):
                b1, b2 = geom_body[d.contact[c].geom1], geom_body[d.contact[c].geom2]
                for r, B in ref_bodies.items():
                    if (b1 in B and b2 in gripper_bodies) or (b2 in B and b1 in gripper_bodies): touch[r][t] = True
                for (h, r) in held_touch:
                    if (b1 in ref_bodies[h] and b2 in ref_bodies[r]) or (b2 in ref_bodies[h] and b1 in ref_bodies[r]): held_touch[(h, r)][t] = True
        subs, start, held = [], 0, None
        for k, (ref, s) in enumerate(spec):
            end = (int(np.argmax(sig[s][start:] > 0)) + start if (s and sig[s][start:].any()) else T - 1) if k < len(spec) - 1 or s else T - 1
            contact_mask = touch[ref].copy()
            if held is not None and held != ref: contact_mask |= held_touch[(held, ref)]
            idx = np.nonzero(contact_mask[start:end + 1])[0]
            contact = int(idx[0] + start) if len(idx) else None
            rec = dict(ref=ref, held=held, signal=s, start=start, contact=contact, end=end)
            if contact is not None:
                rec["T_ref_eef_at_contact"] = (np.linalg.inv(objp[ref][contact]) @ eef[contact]).tolist()
            subs.append(rec)
            held = ref if s in GRASP_SIGNALS else (held if s is None else None)
            start = end + 1
        out_rec[dk] = dict(T=T, subtasks=subs, success=bool(env.is_success()["task"]))
        poses[dk] = dict(eef=eef.astype(np.float32), **{f"obj_{r}": objp[r].astype(np.float32) for r in refs})
        if di % 100 == 0: print(name, di, T, [(s["ref"], s["start"], s["contact"], s["end"]) for s in subs], flush=True)
    os.makedirs(out, exist_ok=True)
    json.dump(dict(dataset=name, task=task, spec=spec, demos=out_rec), open(f"{out}/{name}.json", "w"))
    np.savez_compressed(f"{out}/{name}_poses.npz", **{f"{dk}/{k}": v for dk, d in poses.items() for k, v in d.items()})
    print("DONE", name, len(out_rec), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", required=True); ap.add_argument("--out", default=f"{S}/segments")
    ap.add_argument("--max_demos", type=int, default=None); A = ap.parse_args()
    for n in A.datasets.split(","): segment_dataset(n, A.out, A.max_demos)
