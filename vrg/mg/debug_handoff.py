"""Decompose composed-harness failures on training demos (mimicgen env).

For the first N D0 demos, run a policy
  T1  from the demo's own initial state (reset_to states[0])                      -> baseline from reset
  T2  from the demo's exact state at subtask k's handoff (states[t0], t0 = contact-pre)  -> can the policy start there?
  T3  from states[0], reach to the COMPUTED handoff of subtask k, then the policy  -> is the reach/handoff the problem?
Success = the subtask-k termination signal fires (k < last) or task success (k = last) within the phase budget.
  $MGPY -m vrg.mg.debug_handoff --run $S/mimicgen/runs/square_whole_s0 --k 0 --n 20
"""
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--pre", type=int, default=None); A = ap.parse_args()
    import h5py, mimicgen
    import robomimic.utils.env_utils as EnvUtils, robomimic.utils.file_utils as FileUtils, robomimic.utils.obs_utils as ObsUtils
    import mimicgen.env_interfaces.robosuite as MGI
    from vrg.mg.evaluate import Policy, handoff_offsets, reach, S, HORIZON
    from vrg.mg.segment import TASKS
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_eef_pos"], "rgb": ["agentview_image", "robot0_eye_in_hand_image"]}})
    pol = Policy(A.run); ds = pol.a["dataset"]; task = ds.rsplit("_d", 1)[0]; pre = A.pre if A.pre is not None else pol.a.get("pre", 10)
    p = f"{S}/data/core/{ds}.hdf5"; f = h5py.File(p, "r")
    env = EnvUtils.create_env_from_metadata(env_meta=FileUtils.get_env_metadata_from_dataset(p), render=False, render_offscreen=True, use_image_obs=True); e = env.env
    iface = getattr(MGI, TASKS[task][0])(env=e); SEG = json.load(open(f"{S}/segments/{ds}.json"))["demos"]
    spec, H = handoff_offsets(ds, pre, len(SEG["demo_0"]["subtasks"])); ref, signal = spec[A.k]; last = A.k == len(spec) - 1

    def run_phase(obs, tid, budget):
        pol.reset(tid=tid if pol.a["mode"] == "segment" else 0, anchor=obs["robot0_eef_pos"].copy())
        for _ in range(budget):
            obs, _, _, _ = e.step(pol.act(obs))
            if (last and e._check_success()) or (not last and iface.get_subtask_term_signals().get(signal, 0)): return True
        return False

    def obs_now(): return e._get_observations(force_update=True)
    res = {"T1": [], "T2": [], "T3": []}; errs = []
    for i in range(A.n):
        dk = f"demo_{i}"; g = f[f"data/{dk}"]; st = g["states"][()]; s = SEG[dk]["subtasks"][A.k]
        t0 = max(s["start"], s["contact"] - pre)
        if pol.a["mode"] == "whole":                                     # T1 only meaningful for whole-trajectory policies
            env.reset(); env.reset_to({"states": st[0], "model": g.attrs["model_file"]}); obs = obs_now(); pol.reset(); ok = False
            for _ in range(HORIZON[task]):
                obs, _, _, _ = e.step(pol.act(obs))
                if e._check_success(): ok = True; break
            res["T1"].append(ok)
        env.reset(); env.reset_to({"states": st[t0], "model": g.attrs["model_file"]}); res["T2"].append(run_phase(obs_now(), A.k, 2 * H[A.k]["max_len"]))
        if A.k == 0:
            env.reset(); env.reset_to({"states": st[0], "model": g.attrs["model_file"]}); obs = obs_now()
            OP = iface.get_object_poses(); T_goal = OP[ref] @ H[0]["T"]
            env.reset_to({"states": st[t0]}); true_h = iface.get_robot_eef_pose().copy(); env.reset_to({"states": st[0]}); obs = obs_now()
            errs.append(float(np.linalg.norm(T_goal[:3, 3] - true_h[:3, 3])))
            obs, ok, _ = reach(e, obs, T_goal, grip=-1.0, eef_R=lambda: iface.get_robot_eef_pose()[:3, :3])
            res["T3"].append(run_phase(obs, A.k, 2 * H[A.k]["max_len"]) if ok else False)
        if A.k > 0:                                                       # T3 for a transfer: start from the exact post-grasp state
            sp = SEG[dk]["subtasks"][A.k - 1]; tp = sp["end"] + 1
            for mode in ("hand", "held"):
                env.reset(); env.reset_to({"states": st[tp], "model": g.attrs["model_file"]}); obs = obs_now(); OP = iface.get_object_poses()
                if mode == "held" and H[A.k]["held"]:
                    G = np.linalg.inv(iface.get_robot_eef_pose()) @ OP[H[A.k]["held"]]; T_goal = OP[ref] @ H[A.k]["T_held"] @ np.linalg.inv(G)
                else:
                    T_goal = OP[ref] @ H[A.k]["T"]
                obs, ok, _ = reach(e, obs, T_goal, grip=1.0, eef_R=lambda: iface.get_robot_eef_pose()[:3, :3])
                res.setdefault(f"T3_{mode}", []).append(run_phase(obs, A.k, 2 * H[A.k]["max_len"]) if ok else False)
                res.setdefault(f"reach_{mode}", []).append(ok); errs.append(reach.last_err)
                held_ok = bool(iface.get_subtask_term_signals().get(spec[A.k - 1][1], 0)) if not ok else True
                res.setdefault(f"still_holding_after_failed_reach_{mode}", []).append(held_ok)
        print(i, {k: v[-1] for k, v in res.items() if v}, flush=True)
    print("SUMMARY", {k: f"{sum(v)}/{len(v)}" for k, v in res.items() if v}, "handoff / reach error (cm) median",
          round(np.median(errs) * 100, 1) if errs else None, flush=True)


if __name__ == "__main__":
    main()
