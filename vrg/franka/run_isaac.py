"""Pick experiments in Isaac Lab (Isaac env; source /scratch/kaustubh/vlm_reach_grasp/isaac/env.sh first).

Episodes = (scene seed, target object): every graspable object in each random scene is a target once.
  --grounding oracle          the proposal whose ground-truth label is the target (perfect VLM, real perception)
  --grounding file:<json>     precomputed VLM choices (vrg.franka.ground_offline), matched to live proposals by 3-D point
  --grounding vlm:<model>:<mode>   run the VLM inline (as on the real robot)
  --perception gt             proposals from ground-truth instance masks (perfect segmentation) instead of depth clustering
  --save_scenes DIR           store each episode's observation for offline VLM sweeps

  $ISAACPY -m vrg.franka.run_isaac --seeds 0-49 --grounding oracle --controller cartesian --out runs/franka/oracle_cart
"""
import argparse, json, os, pickle, sys, time
import numpy as np

from vrg.franka.isaac_backend import launch
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0-9"); ap.add_argument("--grounding", default="oracle")
ap.add_argument("--perception", default="depth", choices=["depth", "gt"])
ap.add_argument("--grasp", default="pca", choices=["pca", "fixed"]); ap.add_argument("--controller", default="cartesian", choices=["cartesian", "joint", "native"])
ap.add_argument("--refine", action="store_true", help="look with the wrist camera above the object before grasping")
ap.add_argument("--n_objects", type=int, default=6); ap.add_argument("--out", required=True); ap.add_argument("--save_scenes", default=None)
ap.add_argument("--video", type=int, default=0, help="record the external camera for the first N episodes")
ap.add_argument("--reactive", default=None, help="reactive grounding (vrg.franka.reactive) with a VLM server: 'port=6102,mode=point,K=2,verify=1,R=2,tau=0'")
ap.add_argument("--speed", type=float, default=1.0, help="scale joint and straight-line speeds (1 = 30%% of joint limits, 15 cm/s)")
ap.add_argument("--settle", type=float, default=0.3, help="longest wait after each move (s)")
ap.add_argument("--settle_tol", type=float, default=0.01, help="the wait ends once every joint is within this many rad of its target")
ap.add_argument("--timing", action="store_true", help="log robot (simulated) and compute time per stage (vrg.franka.timing)")
ap.add_argument("--debug_dir", default=None, help="dump the look-pose wrist view with ground-truth masks and object poses per episode")
ap.add_argument("--episodes", default=None, help="only these episodes, all recorded: 'seed:target,...' (e.g. 103:021_bleach_cleanser)")
A = ap.parse_args()
import faulthandler   # per-episode watchdog: a hung episode (seen: RTX render deadlock) exits with code 1, run.sh restarts and resumes
app = launch()

from vrg.franka.isaac_backend import IsaacFranka, OBJECTS
from vrg.franka.perception import fit_table
from vrg.franka import episode


class GT:
    def __init__(self, env, target): self.env, self.target = env, target

    def label(self, F, props, min_frac=0.5):
        lab = {}
        masks = {cam: {n: self.env.obj_points_mask(F[cam], n) for n in self.env.scene_objs} for cam in F}
        for p in props:
            cnt, tot = {}, 0
            for cam, (_, m) in p["pix"].items():
                tot += m.sum()
                for n, om in masks[cam].items(): cnt[n] = cnt.get(n, 0) + int((m & om).sum())
            best = max(cnt, key=cnt.get) if cnt else None
            lab[p["id"]] = best if best is not None and tot and cnt[best] / tot >= min_frac else None
        return lab

    def lifted_name(self):
        up = [n for n in self.env.scene_objs if self.env.lifted(n)]
        return self.target if self.target in up else (up[0] if up else None)

    def moved(self, exclude_target=True):
        return [n for n in self.env.moved_objects() if not (exclude_target and n == self.target)]


def gt_proposals(env):
    """Perfect segmentation: one proposal per scene object from its instance mask (same dict format as perception)."""
    def f(F):
        allP = np.concatenate([fr.backproject() for fr in F.values()]); table = fit_table(allP[allP[:, 0] > 0.25])
        props = []
        for n in env.scene_objs:
            pts, pix = [], {}
            for cam, fr in F.items():
                m = env.obj_points_mask(fr, n) & (fr.depth > 0)
                if m.sum() < 20: continue
                v, u = np.nonzero(m); pts.append(fr.backproject(m)); pix[cam] = ([int(u.min()), int(v.min()), int(u.max()), int(v.max())], m)
            if not pts: continue
            Q = np.concatenate(pts); h = Q @ table[0] + table[1]
            props.append(dict(id=len(props), points=Q, center=np.median(Q, 0), top=float(np.percentile(h, 98)), extent=np.ptp(Q[:, :2], 0), pix=pix))
        return props, table
    return f


def make_ground(spec, gt_holder):
    if spec == "oracle":
        def g(F, props, desc):
            lab = gt_holder["gt"].label(F, props); c = [p for p in props if lab.get(p["id"]) == gt_holder["gt"].target]
            pid = max(c, key=lambda p: len(p["points"]))["id"] if c else None
            return pid, dict(mode="oracle", choice=pid)
        return g
    if spec.startswith("file:"):
        G = json.load(open(spec[5:]))
        def g(F, props, desc):
            r = G.get(gt_holder["key"]); xyz = None if r is None or r.get("xyz") is None else np.array(r["xyz"])
            if xyz is None or not props: return None, dict(mode="file", choice=None)
            d = [np.linalg.norm(p["center"][:2] - xyz[:2]) for p in props]; k = int(np.argmin(d))
            return props[k]["id"], dict(mode="file", model=r.get("model"), choice=props[k]["id"], match_dist=float(d[k]))
        return g
    if spec.startswith("vlm:"):
        from vrg.franka.vlm import VLM, ground
        _, model, mode = spec.split(":"); V = VLM(model, device="cuda:0")
        def g(F, props, desc):
            r = ground(V, mode, desc, F["ext"], props); return r["choice"], r
        return g
    raise ValueError(spec)


def main():
    a, b = A.seeds.split("-"); seeds = range(int(a), int(b) + 1)
    only = {(int(e.split(":")[0]), e.split(":")[1]) for e in A.episodes.split(",")} if A.episodes else None
    if only: seeds = sorted({s for s, _ in only})
    os.makedirs(A.out, exist_ok=True); resf = f"{A.out}/results.json"
    res = json.load(open(resf)) if os.path.exists(resf) else []; done = {(r["seed"], r["target_name"]) for r in res}
    env = IsaacFranka(n_objects=A.n_objects, record_video=False); env.settle_s, env.settle_tol = A.settle, A.settle_tol
    from vrg.franka import kinematics as Kin
    Kin.VEL_SCALE, Kin.LIN_SPEED = 0.3 * A.speed, 0.15 * A.speed
    RX = dict(kv.split("=") for kv in A.reactive.split(",")) if A.reactive else None
    if RX:
        from vrg.franka import reactive
        from vrg.franka.vlm_client import RemoteVLM
        vlm = RemoteVLM(int(RX["port"]))
    holder = {}; ground = make_ground(A.grounding, holder)
    if A.timing:
        from vrg.franka import timing
        TR = timing.instrument(env, vlm if RX else None); propose = gt_proposals(env) if A.perception == "gt" else None
    n_ep = 0
    for seed in seeds:
        objs = env.reset(seed=seed); targets = [n for n, _, g in objs if g]
        for tname in targets:
            n_ep += 1
            if (seed, tname) in done or (only and (seed, tname) not in only): continue
            faulthandler.cancel_dump_traceback_later(); faulthandler.dump_traceback_later(300, exit=True, file=sys.stderr)
            env.reset(seed=seed)                                 # same scene for every target
            env.record_video = n_ep <= A.video or bool(only); env.video = []
            gt = GT(env, tname); holder.update(gt=gt, key=f"{seed}|{tname}")
            if A.timing: TR.reset()
            t0 = time.time()
            if RX and RX.get("probe"):
                rec = reactive.probe(env, OBJECTS[tname][0], vlm, RX["mode"], K_app=int(RX.get("K", 4)), gt=gt)
            elif RX:
                rec = reactive.run(env, OBJECTS[tname][0], vlm, RX["mode"], K_app=int(RX.get("K", 2)), verify=bool(int(RX.get("verify", 0))),
                                   R=int(RX.get("R", 2)), tau=float(RX.get("tau", 0)), ext_modes=RX["ext"].split("+") if "ext" in RX else None, gate=RX.get("gate"), rule=RX.get("rule", "vote"), grasp_mode=A.grasp, controller=A.controller, gt=gt)
            else:
                rec = episode.run(env, OBJECTS[tname][0], ground, grasp_mode=A.grasp, controller=A.controller, gt=gt, propose=propose, refine=A.refine)
            if A.timing:
                rec["timing"] = TR.summary(); rec["timing"]["robot:return_home(planned)"] = timing.return_home_time(env.state().q, settle=A.settle)
                rec["timing"]["robot_total"] += rec["timing"]["robot:return_home(planned)"]; rec["timing_items"] = TR.items
            F, props = rec.pop("_frames"), rec.pop("_props"); look = rec.pop("_look", None)
            rec["obj_pose_init"] = {n: env.init_pose[n].tolist() for n in env.scene_objs}; rec["obj_pose_end"] = {n: env.obj_pose(n).tolist() for n in env.scene_objs}
            if A.debug_dir:
                os.makedirs(A.debug_dir, exist_ok=True); D = dict(rec=rec, ext=dict(rgb=F["ext"].rgb, depth=F["ext"].depth, K=F["ext"].K, T=F["ext"].T_base_cam),
                    ext_masks={n: np.packbits(env.obj_points_mask(F["ext"], n)) for n in env.scene_objs},
                    ext_props=[dict(id=p["id"], points=p["points"], mask=np.packbits(p["pix"]["ext"][1]) if "ext" in p["pix"] else None) for p in props])
                if look:
                    fw = look["frame"]; D["wrist"] = dict(rgb=fw.rgb, depth=fw.depth, K=fw.K, T=fw.T_base_cam)
                    D["wrist_masks"] = {n: np.packbits(env.obj_points_mask(fw, n)) for n in env.scene_objs}
                    D["wrist_props"] = [dict(id=p["id"], points=p["points"], mask=np.packbits(p["pix"]["wrist"][1])) for p in look["props"]]; D["wrist_match"] = look["match"]
                pickle.dump(D, open(f"{A.debug_dir}/{seed}_{tname}.pkl", "wb"))
            if A.save_scenes:
                os.makedirs(A.save_scenes, exist_ok=True)
                pickle.dump(dict(seed=seed, target=tname, desc=OBJECTS[tname][0], objects=objs,
                                 frames={k: dict(rgb=f.rgb, depth=f.depth, K=f.K, T=f.T_base_cam) for k, f in F.items()},
                                 masks={k: {n: np.packbits(env.obj_points_mask(f, n)) for n in env.scene_objs} for k, f in F.items()},
                                 prop_labels=rec.get("prop_labels")), open(f"{A.save_scenes}/{seed}_{tname}.pkl", "wb"))
            rec.update(seed=seed, target_name=tname, n_objects=len(objs), sec=time.time() - t0,
                       prop_labels={str(k): v for k, v in rec.get("prop_labels", {}).items()})
            if env.video:
                import imageio; os.makedirs(f"{A.out}/videos", exist_ok=True)
                imageio.mimsave(f"{A.out}/videos/{seed}_{tname}_{'ok' if rec.get('success') else 'fail'}.mp4", env.video, fps=20)
            res.append(rec); json.dump(res, open(resf, "w"), indent=0, default=float)
            print(f"seed {seed} {OBJECTS[tname][0]:16s} success={rec.get('success')} fail={rec.get('fail')} "
                  f"ground_ok={rec.get('grounding_correct')} init_ok={rec.get('initial_correct')} n_vlm={rec.get('n_vlm')} reach_err={rec.get('reach_err', float('nan')):.4f} ({rec['sec']:.0f}s)", flush=True)
    sr = np.mean([r.get("success", False) for r in res]) if res else 0
    print(f"== {A.out}: success {sr:.3f} over {len(res)} episodes", flush=True)


try:
    main()
except Exception:
    import traceback; traceback.print_exc()
finally:
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)          # Kit's shutdown hangs and ignores SIGTERM
