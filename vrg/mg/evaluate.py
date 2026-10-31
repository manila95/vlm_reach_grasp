"""Evaluate MimicGen policies: whole-trajectory (P1) vs composed reach + contact-phase policy (P2 / C1 / C2) (mimicgen env).

Protocol (MimicGen / robomimic): the test variant's env comes from its dataset's env metadata (MimicGen's official reset
distribution D0/D1/D2/O1/O2), seeded resets (np.random.seed(seed0 + i) before each reset), success = env._check_success(),
horizons from MimicGen's training configs (square / threading / coffee 400, three_piece_assembly / mug_cleanup 500).

  whole     run the policy from reset until success or the horizon.
  composed  for each subtask k of MimicGen's task spec: reach (lift -> travel at a safe height -> descend) to the handoff
            pose = reference-object pose (oracle from sim; VLM later) x mean start pose of the training segments in that
            object's frame; then run the policy until subtask k's termination signal (the last subtask: task success).
            Budget per phase = 2 x the longest training segment. With a whole-trajectory checkpoint this is control C1.
Optional handoff noise (position sigma in m, yaw sigma in rad). Per-episode record: success, phase reached, failure
cause (reach / skill_k / horizon), steps.
  $MGPY -m vrg.mg.evaluate --run $S/mimicgen/runs/sq_seg_s0 --variant square_d1 --method composed --episodes 100
"""
import argparse, json, os, sys, time
import numpy as np, torch
from scipy.spatial.transform import Rotation as Rot

S = "/scratch/kaustubh/vlm_reach_grasp/mimicgen"
HORIZON = {"square": 400, "threading": 400, "coffee": 400, "three_piece_assembly": 500, "mug_cleanup": 500}


class Policy:
    def __init__(self, run, device="cuda:0"):
        from vrg.policy import DiffusionPolicy
        ck = torch.load(f"{run}/ckpt.pt", map_location=device, weights_only=False); self.a = ck["args"]
        stats = torch.load(f"{run}/stats.pt"); self.cams = tuple(self.a["cams"].split(","))
        self.pol = DiffusionPolicy(stats, n_obs=2, horizon=16, pdim=9, n_tasks=ck["n_tasks"], cams=self.cams, crop=76).to(device)
        self.pol.load_state_dict(ck["ema"]); self.pol.eval(); self.dev = device; self.reset()

    def reset(self, tid=0, anchor=None):
        self.hist, self.queue, self.tid, self.anchor = [], [], tid, anchor

    def act(self, obs):
        p = np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]).astype(np.float32)
        if self.a["proprio"] == "rel": p[:3] -= self.anchor
        # raw robosuite frames are upside down (OpenGL); the datasets were recorded through robomimic, which flips them
        o = {"agent": obs["agentview_image"][::-1].copy(), "wrist": obs["robot0_eye_in_hand_image"][::-1].copy(), "proprio": p}
        self.hist = (self.hist + [o])[-2:]
        if len(self.hist) == 1: self.hist = self.hist * 2
        if not self.queue:
            b = {c: torch.from_numpy(np.stack([h[c] for h in self.hist]))[None].to(self.dev) for c in self.cams}
            b["proprio"] = torch.from_numpy(np.stack([h["proprio"] for h in self.hist]))[None].to(self.dev)
            b["tid"] = torch.tensor([self.tid], device=self.dev)
            self.queue = list(self.pol.act(b)[0].float().cpu().numpy())
        return self.queue.pop(0)


def handoff_offsets(dataset, pre, task_spec_len):
    """Mean start pose of the training contact segments in each subtask's reference-object frame."""
    SEG = json.load(open(f"{S}/segments/{dataset}.json")); Z = np.load(f"{S}/segments/{dataset}_poses.npz")
    pos = [[] for _ in range(task_spec_len)]; rot = [[] for _ in range(task_spec_len)]; length = [[] for _ in range(task_spec_len)]
    hpos = [[] for _ in range(task_spec_len)]; hrot = [[] for _ in range(task_spec_len)]; held = [None] * task_spec_len
    for dk, d in SEG["demos"].items():
        if any(s["contact"] is None for s in d["subtasks"]): continue      # (replay success flags are unreliable, see train.py)
        for k, s in enumerate(d["subtasks"]):
            if s["contact"] is None: continue
            t0 = max(s["start"], s["contact"] - pre); R = np.linalg.inv(Z[f"{dk}/obj_{s['ref']}"][t0]) @ Z[f"{dk}/eef"][t0]
            pos[k].append(R[:3, 3]); rot[k].append(R[:3, :3]); length[k].append(s["end"] - t0 + 1)
            if s["held"]:                                   # held-object pose relative to the reference at the handoff
                Rh = np.linalg.inv(Z[f"{dk}/obj_{s['ref']}"][t0]) @ Z[f"{dk}/obj_{s['held']}"][t0]
                hpos[k].append(Rh[:3, 3]); hrot[k].append(Rh[:3, :3]); held[k] = s["held"]
    def mean_pose(P, Rs):
        U, _, Vt = np.linalg.svd(np.sum(Rs, 0)); Rm = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt     # chordal mean rotation
        T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = np.mean(P, 0); return T
    out = []
    for k in range(task_spec_len):
        d = dict(T=mean_pose(pos[k], rot[k]), pos_std=np.std(pos[k], 0).tolist(), max_len=int(np.max(length[k])), n=len(pos[k]), held=held[k])
        if held[k]: d.update(T_held=mean_pose(hpos[k], hrot[k]), held_pos_std=np.std(hpos[k], 0).tolist())
        out.append(d)
    return SEG["spec"], out


def reach(e, obs, T_goal, grip, eef_R, frames=None, tol=0.006, max_steps=250, safe_dz=0.08):
    """Scripted transit to a pose: up to a safe height, across, then down, with orientation servoing (OSC_POSE deltas).
    eef_R(): current end-effector rotation in the SAME frame as T_goal (MimicGen's eef frame; robosuite's
    robot0_eef_quat is a different body frame, rotated 90 deg from it)."""
    eef = lambda o: o["robot0_eef_pos"]
    z_safe = T_goal[2, 3] + safe_dz                  # transit height relative to the goal: the start pose is already high, and
    wps = ([np.r_[eef(obs)[:2], z_safe]] if eef(obs)[2] < z_safe - 0.01 else []) + [np.r_[T_goal[:2, 3], z_safe], T_goal[:3, 3]]
    # (lifting above the start height put the far side of the table out of reach)
    steps = 0
    for w in wps:
        for _ in range(max_steps):
            R_cur = Rot.from_matrix(eef_R()); err_r = (Rot.from_matrix(T_goal[:3, :3]) * R_cur.inv()).as_rotvec()
            err_p = w - eef(obs)
            if np.linalg.norm(err_p) < (tol if w is wps[-1] else 0.02) and np.linalg.norm(err_r) < 0.05: break
            a = np.zeros(7); a[:3] = np.clip(err_p / 0.05 * 1.5, -1, 1); a[3:6] = np.clip(err_r / 0.5, -1, 1); a[6] = grip
            obs, _, _, _ = e.step(a); steps += 1
            if frames is not None: frames.append(obs["agentview_image"])
    ok = np.linalg.norm(T_goal[:3, 3] - eef(obs)) < 0.02
    reach.last_err = float(np.linalg.norm(T_goal[:3, 3] - eef(obs)))
    return obs, ok, steps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--variant", required=True)
    ap.add_argument("--method", choices=["whole", "composed"], required=True); ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=100000); ap.add_argument("--noise_pos", type=float, default=0.0); ap.add_argument("--noise_yaw", type=float, default=0.0)
    ap.add_argument("--pre", type=int, default=None, help="handoff = this many steps before contact (default: the run's --pre)")
    ap.add_argument("--video", type=int, default=0); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0")
    A = ap.parse_args()
    import robomimic.utils.env_utils as EnvUtils, robomimic.utils.file_utils as FileUtils, robomimic.utils.obs_utils as ObsUtils
    import mimicgen.env_interfaces.robosuite as MGI
    from vrg.mg.segment import TASKS
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_eef_pos"], "rgb": ["agentview_image", "robot0_eye_in_hand_image"]}})
    pol = Policy(A.run, A.device); train_ds = pol.a["dataset"]; task = train_ds.rsplit("_d", 1)[0]
    em = FileUtils.get_env_metadata_from_dataset(f"{S}/data/core/{A.variant}.hdf5")
    env = EnvUtils.create_env_from_metadata(env_meta=em, render=False, render_offscreen=True, use_image_obs=True); e = env.env
    iface = getattr(MGI, TASKS[task][0])(env=e)
    pre = A.pre if A.pre is not None else pol.a.get("pre", 10)
    if A.method == "composed": spec, H = handoff_offsets(train_ds, pre, len(json.load(open(f"{S}/segments/{train_ds}.json"))["spec"]))
    os.makedirs(A.out, exist_ok=True); resf = f"{A.out}/results.json"
    res = json.load(open(resf)) if os.path.exists(resf) else []; done = {r["episode"] for r in res}
    rng = np.random.default_rng(12345)
    for i in range(A.episodes):
        if i in done: continue
        np.random.seed(A.seed0 + i); obs = e.reset(); t0 = time.time(); frames = [] if i < A.video else None
        rec = dict(episode=i, seed=A.seed0 + i, method=A.method, variant=A.variant, success=False, phase=0, fail=None, steps=0)
        if A.method == "whole":
            pol.reset()
            for t in range(HORIZON[task]):
                obs, _, _, _ = e.step(pol.act(obs)); rec["steps"] += 1
                if frames is not None: frames.append(obs["agentview_image"])
                if e._check_success(): rec["success"] = True; break
            rec["fail"] = None if rec["success"] else "horizon"
        else:
            held = False
            for k, (ref, signal) in enumerate(spec):
                rec["phase"] = k
                OP = iface.get_object_poses()
                if H[k]["held"] and held:   # place the HELD object at its mean pose relative to the reference, given the current grasp
                    G = np.linalg.inv(iface.get_robot_eef_pose()) @ OP[H[k]["held"]]
                    T_goal = OP[ref] @ H[k]["T_held"] @ np.linalg.inv(G)
                else:
                    T_goal = OP[ref] @ H[k]["T"]
                if A.noise_pos or A.noise_yaw:
                    T_goal = T_goal.copy(); T_goal[:3, 3] += rng.normal(0, A.noise_pos, 3) * np.array([1, 1, 0.5])
                    T_goal[:3, :3] = Rot.from_euler("z", rng.normal(0, A.noise_yaw)).as_matrix() @ T_goal[:3, :3]
                obs, ok, n = reach(e, obs, T_goal, grip=1.0 if held else -1.0, eef_R=lambda: iface.get_robot_eef_pose()[:3, :3], frames=frames); rec["steps"] += n
                if not ok: rec["fail"] = f"reach_{k}"; rec["reach_err"] = reach.last_err; rec["reach_goal"] = T_goal[:3, 3].tolist(); break
                pol.reset(tid=k if pol.a["mode"] == "segment" else 0, anchor=obs["robot0_eef_pos"].copy())
                last = k == len(spec) - 1; fired = False
                for t in range(2 * H[k]["max_len"]):
                    obs, _, _, _ = e.step(pol.act(obs)); rec["steps"] += 1
                    if frames is not None: frames.append(obs["agentview_image"])
                    if last and e._check_success(): fired = True; break
                    if not last and iface.get_subtask_term_signals().get(signal, 0): fired = True; break
                if not fired: rec["fail"] = f"skill_{k}"; break
                held = signal in ("grasp", "grasp_1", "grasp_2")
            rec["success"] = bool(e._check_success())
            if rec["success"]: rec["fail"] = None
        rec["sec"] = time.time() - t0; res.append(rec); json.dump(res, open(resf, "w"))
        if frames:
            import imageio; os.makedirs(f"{A.out}/videos", exist_ok=True)
            imageio.mimsave(f"{A.out}/videos/{i}_{'ok' if rec['success'] else 'fail'}.mp4", [f[::-1] for f in frames], fps=20)
        print(f"ep {i} success={rec['success']} fail={rec['fail']} steps={rec['steps']} ({rec['sec']:.0f}s)", flush=True)
    print(f"== {A.out}: success {np.mean([r['success'] for r in res]):.3f} over {len(res)}", flush=True)


if __name__ == "__main__":
    main()
