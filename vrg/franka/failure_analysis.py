"""Root causes of the pipeline's failures (vlmrg env). Writes runs/franka/failure_analysis.json and demo/franka/failures/*.png.

Evidence (all from logs, no re-simulation):
  * robot side: the diagnostic re-run of every failed episode of config b plus every 10th success
    (`run_isaac --episodes ... --debug_dir`, tools/isaac: runs/diag_b). It logs the look-pose wrist view with ground-truth
    masks, planned and reached TCP poses, and object poses, so each failure can be checked against the true geometry.
  * VLM side: the best reactive run (4B, 3 modes + 2 re-asks) and the descriptive-name grounding test.

Root-cause rules (first match wins):
  segmentation   merged_ext       the target's ext pixels fall mostly inside a proposal labelled as another object
                 occluded_ext     fewer than 150 target pixels visible from the external camera
  grasp_plan     merged_wrist     >5% of the matched wrist proposal's pixels belong to other objects (width inflated)
                 width_cap        clean cluster, object's true narrow side <= 7.5 cm: rejected by the planner's cap/margin
                 shape            clean cluster, object wider than the hand across its narrow side at the centroid
                 no_look          the wrist look move failed and the external-view plan was too wide
  reach / approach_collision
                 palm_neighbour   a neighbour's volume intersects the Franka Hand palm or fingers at the goal pose
  grasp_slip / grasp_miss
                 thin_grasp       closed on less than 3 cm of material (finger width after closing < 3 cm)
  grounding      lookalike        the chosen object is the same kind (box/can/bottle/tool) as the target
  python -m vrg.franka.failure_analysis
"""
import collections, glob, json, os, pickle
import numpy as np

S = "/scratch/kaustubh/vlm_reach_grasp/isaac"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META = json.load(open(f"{S}/assets/ycb/ycb_meta.json"))
KIND = {"003_cracker_box": "box", "004_sugar_box": "box", "008_pudding_box": "box", "009_gelatin_box": "box", "036_wood_block": "box",
        "061_foam_brick": "box", "002_master_chef_can": "can", "005_tomato_soup_can": "can", "007_tuna_fish_can": "can",
        "010_potted_meat_can": "can", "006_mustard_bottle": "bottle", "021_bleach_cleanser": "bottle", "011_banana": "other",
        "037_scissors": "tool", "040_large_marker": "tool", "051_large_clamp": "tool", "024_bowl": "dish", "025_mug": "dish"}
HAND_OPEN, CAP = 0.08, 0.075


def unpack(m, H, W): return np.unpackbits(m)[:H * W].reshape(H, W).astype(bool)


def obj_grid(name, T, n=7):
    lo, hi = np.array(META[name]["aabb_min"]), np.array(META[name]["aabb_max"])
    g = np.stack(np.meshgrid(*[np.linspace(lo[i], hi[i], n) for i in range(3)]), -1).reshape(-1, 3)
    return g @ T[:3, :3].T + T[:3, 3]


def hand_hits(L, w=HAND_OPEN):
    """Points L (in the TCP frame: x finger width, y closing, z toward the object) inside the palm or fingers (URDF-sized boxes)."""
    palm = (np.abs(L[:, 0]) < 0.032) & (np.abs(L[:, 1]) < 0.102) & (L[:, 2] < -0.045) & (L[:, 2] > -0.11)
    fing = (np.abs(L[:, 0]) < 0.011) & (np.abs(L[:, 1]) > w / 2) & (np.abs(L[:, 1]) < w / 2 + 0.02) & (L[:, 2] <= 0) & (L[:, 2] > -0.045)
    return palm | fing


def narrow_side(name, T):
    R = T[:3, :3]; up = int(np.argmax(np.abs(R.T @ [0, 0, 1.0]))); hor = [i for i in range(3) if i != up]
    return float(min(np.array(META[name]["size"])[hor]))


def robot_side(diag_dir):
    out = []
    for p in sorted(glob.glob(f"{diag_dir}/*.pkl")):
        X = pickle.load(open(p, "rb")); r = X["rec"]; b = os.path.basename(p)[:-4]; seed, t = int(b.split("_")[0]), b.split("_", 1)[1]
        fail = "success" if r.get("success") else r.get("fail")
        if fail == "grounding" and r.get("target_segmented") is False: fail = "segmentation"
        e = dict(seed=seed, target=t, fail=fail, cause=None, evidence={})
        Ti = np.array(r["obj_pose_init"][t])
        if fail == "segmentation":
            H, W = X["ext"]["depth"].shape; m = unpack(X["ext_masks"][t], H, W)
            ov = [(int((unpack(q["mask"], H, W) & m).sum()), q["id"]) for q in X["ext_props"] if q["mask"] is not None]
            best = max(ov) if ov else (0, None)
            other = (r.get("prop_labels") or {}).get(best[1]) if best[1] is not None else None
            e["evidence"] = dict(target_px=int(m.sum()), in_proposal_of=other)
            e["cause"] = "occluded_ext" if m.sum() < 150 else ("merged_ext" if best[0] > 0.3 * m.sum() else "other")
        elif fail == "grasp_plan" and "wrist" not in X:                 # the look move failed: grasp planned from the external view
            ns = narrow_side(t, Ti); e["evidence"] = dict(requested_cm=round(r["grasp"]["width"] * 100, 1), true_narrow_cm=round(ns * 100, 1),
                                                         look=r.get("refine", {}).get("reason"))
            e["cause"] = "width_cap" if ns <= CAP and r["grasp"]["width"] < 1.3 * (ns + 0.01) else ("shape" if ns > CAP else "no_look")
        elif fail == "grasp_plan":
            H, W = X["wrist"]["depth"].shape; M = {n: unpack(v, H, W) for n, v in X["wrist_masks"].items()}
            q = next(w for w in X["wrist_props"] if w["id"] == X["wrist_match"]); qm = unpack(q["mask"], H, W); tot = max(qm.sum(), 1)
            other = {n[4:]: round(float((qm & m).sum()) / tot, 2) for n, m in M.items() if n != t and (qm & m).sum() / tot > 0.05}
            ns = narrow_side(t, Ti)
            e["evidence"] = dict(requested_cm=round(r["grasp"]["width"] * 100, 1), true_narrow_cm=round(ns * 100, 1), other_px=other)
            e["cause"] = "merged_wrist" if other else ("width_cap" if ns <= CAP else "shape")
        elif fail in ("reach", "approach_collision"):
            Tg = np.array(r["T_goal"]); hit = []
            for n, Tn in r["obj_pose_init"].items():
                if n != t and hand_hits((obj_grid(n, np.array(Tn)) - Tg[:3, 3]) @ Tg[:3, :3]).any(): hit.append(n[4:])
            er = (np.array(r["T_reached"])[:3, 3] - Tg[:3, 3]) @ Tg[:3, :3]
            e["evidence"] = dict(neighbours_in_hand=hit, stopped_above_cm=round(-er[2] * 100, 1))
            e["cause"] = "palm_neighbour" if hit else "other"
        elif fail in ("grasp_slip", "grasp_miss"):
            w = r.get("width_after_close"); e["evidence"] = dict(width_after_close_cm=round(w * 100, 1) if w is not None else None)
            e["cause"] = "thin_grasp" if w is not None and w < 0.03 else "other"
        elif fail == "success":
            Tg = np.array(r["T_goal"]) if "T_goal" in r else None
            e["evidence"] = dict(neighbours_in_hand=[n[4:] for n, Tn in r["obj_pose_init"].items() if Tg is not None and n != t and
                                                     hand_hits((obj_grid(n, np.array(Tn)) - Tg[:3, 3]) @ Tg[:3, :3]).any()])
        out.append(e)
    return out


def vlm_side(run, names_dir):
    R = json.load(open(f"{S}/runs/{run}/results.json")); G = [r for r in R if r.get("fail") == "grounding"]
    agree = lambda r: len({h["obj"] for h in r["vlm_hist"] if h["stage"] == "ext"}) == 1
    seg = [r for r in R if r.get("target_segmented")]
    out = dict(run=run, n=len(R), grounding_failures=len(G),
               lookalike=sum(KIND.get(r["chosen_obj"]) == KIND.get(r["target_name"]) for r in G),
               pairs=collections.Counter(f"{r['target_name'][4:]} → {(r['chosen_obj'] or 'none')[4:]}" for r in G).most_common(10),
               agree=dict(n=sum(agree(r) for r in seg), correct=float(np.mean([bool(r["grounding_correct"]) for r in seg if agree(r)]))),
               disagree=dict(n=sum(not agree(r) for r in seg), correct=float(np.mean([bool(r["grounding_correct"]) for r in seg if not agree(r)]))),
               failures_flagged_by_disagreement=sum(not agree(r) for r in G),
               target_in_some_wrist_answer=sum(any(h.get("obj") == r["target_name"] for h in r["vlm_hist"] if h["stage"].startswith("approach")) for r in G))
    names = {}
    for f in glob.glob(f"{names_dir}/qwen3vl-*_*.json"):
        k = os.path.basename(f)[:-5]; D = json.load(open(f)); base = json.load(open(f"{S}/ground_eval/{k}.json"))
        names[k] = dict(ycb=float(np.mean([v["correct"] for v in base.values()])), descriptive=float(np.mean([v["correct"] for v in D.values()])))
    out["names_test"] = names
    return out


def figures(robot):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    os.makedirs(f"{ROOT}/demo/franka/failures", exist_ok=True)
    # 1. palm collisions: top-down footprints at the goal pose
    ex = [e for e in robot if e["cause"] == "palm_neighbour"][:4]
    fig, axs = plt.subplots(1, len(ex), figsize=(3.2 * len(ex), 3.3))
    for ax, e in zip(np.atleast_1d(axs), ex):
        X = pickle.load(open(f"{S}/diag_b/{e['seed']}_{e['target']}.pkl", "rb")); r = X["rec"]; Tg = np.array(r["T_goal"])
        for n, Tn in r["obj_pose_init"].items():
            g = obj_grid(n, np.array(Tn), 2); top = g[:, 2].max(); xy = g[:, :2]; c = xy.mean(0)
            hull = xy[np.argsort(np.arctan2(*(xy - c).T[::-1]))]
            col = "#1baf7a" if n == e["target"] else ("#eb6834" if n[4:] in e["evidence"]["neighbours_in_hand"] else "#c3c2b7")
            ax.add_patch(Polygon(hull, closed=True, fc=col, ec="k", alpha=0.6, lw=0.5)); ax.text(*c, f"{n[4:12]}\n{top*100:.0f} cm", fontsize=6, ha="center", va="center", clip_on=True)
        R2, o = Tg[:2, :2], Tg[:2, 3]
        for (hx, hy, cx, cy) in [(0.032, 0.102, 0, 0), (0.011, 0.01, 0, 0.05), (0.011, 0.01, 0, -0.05)]:
            box = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]) + [cx, cy]
            ax.add_patch(Polygon(box @ R2.T + o, closed=True, fc="none", ec="#a83232", lw=1.5, ls="-" if hx > 0.02 else "--"))
        ax.set_xlim(o[0] - 0.16, o[0] + 0.16); ax.set_ylim(o[1] - 0.16, o[1] + 0.16); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{e['seed']} {e['target'][4:]}: {e['fail']}\npalm hits {', '.join(e['evidence']['neighbours_in_hand'])}", fontsize=8)
    fig.text(0.5, 0.01, "green: target   orange: neighbour inside the hand   grey: other objects (labels: object top height)   red: Franka Hand palm (solid) and fingers (dashed) at the planned grasp", ha="center", fontsize=7)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.08, wspace=0.08); plt.savefig(f"{ROOT}/demo/franka/failures/palm_collisions.png", dpi=130); plt.close()
    # 2. merged clusters: external view (segmentation) and wrist view (grasp_plan)
    from PIL import Image
    tiles = []
    for cause, view in (("merged_ext", "ext"), ("merged_wrist", "wrist")):
        for e in [x for x in robot if x["cause"] == cause][:2]:
            X = pickle.load(open(f"{S}/diag_b/{e['seed']}_{e['target']}.pkl", "rb")); img = X[view]["rgb"].copy(); H, W = img.shape[:2]
            if view == "ext":
                m = unpack(X["ext_masks"][e["target"]], H, W)
                q = max(X["ext_props"], key=lambda q: (unpack(q["mask"], H, W) & m).sum() if q["mask"] is not None else -1); pm = unpack(q["mask"], H, W)
            else:
                q = next(w for w in X["wrist_props"] if w["id"] == X["wrist_match"]); pm = unpack(q["mask"], H, W); m = unpack(X["wrist_masks"][e["target"]], H, W)
            img[pm] = (0.45 * img[pm] + 0.55 * np.array([235, 104, 52])).astype(np.uint8)       # proposal: orange
            img[m & pm] = (0.45 * img[m & pm] + 0.55 * np.array([27, 175, 122])).astype(np.uint8)  # target part: green
            v, u = np.nonzero(pm); y0, y1, x0, x1 = max(v.min() - 40, 0), min(v.max() + 40, H), max(u.min() - 40, 0), min(u.max() + 40, W)
            tiles.append((Image.fromarray(img[y0:y1, x0:x1]).resize((320, int(320 * (y1 - y0) / max(x1 - x0, 1)))), f"{view}: {e['target'][4:]} merged"))
    if tiles:
        Hm = max(t.height for t, _ in tiles); canvas = Image.new("RGB", (330 * len(tiles), Hm + 22), (255, 255, 255))
        from PIL import ImageDraw; d = ImageDraw.Draw(canvas)
        for i, (t, cap) in enumerate(tiles): canvas.paste(t, (330 * i, 22)); d.text((330 * i + 4, 4), cap, fill=(20, 20, 20))
        canvas.save(f"{ROOT}/demo/franka/failures/merged_clusters.png")


def main():
    robot = robot_side(f"{S}/diag_b")
    out = dict(robot=robot, vlm=vlm_side("rx_4b_ens3_K2", f"{S}/ground_eval_descnames"))
    summ = collections.Counter((e["fail"], e["cause"]) for e in robot if e["fail"] != "success")
    out["robot_summary"] = [dict(fail=f, cause=c, n=n) for (f, c), n in sorted(summ.items())]
    out["success_with_neighbour_in_hand"] = sum(bool(e["evidence"].get("neighbours_in_hand")) for e in robot if e["fail"] == "success")
    out["n_success_controls"] = sum(e["fail"] == "success" for e in robot)
    os.makedirs(f"{ROOT}/runs/franka", exist_ok=True); json.dump(out, open(f"{ROOT}/runs/franka/failure_analysis.json", "w"), indent=1, default=str)
    for d in out["robot_summary"]: print(d)
    print("successes with a neighbour in the hand:", out["success_with_neighbour_in_hand"], "/", out["n_success_controls"])
    print(json.dumps({k: v for k, v in out["vlm"].items() if k != "pairs"}, indent=1))
    figures(robot)


if __name__ == "__main__":
    main()
