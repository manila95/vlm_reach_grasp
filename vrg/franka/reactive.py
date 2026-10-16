"""Reactive grounding: keep asking the VLM while moving, and change target when the evidence says so.

One-shot grounding (episode.run) commits to the VLM's first answer from the external camera. Here the answer is a belief
that the arm keeps testing on its way, with views that get closer and sharper:

  1. external view -> one VLM choice per mode in `ext_modes` (one vote each; e.g. point, marks, verify)
  2. approach: the joint path to the look pose above the current target is cut into K+1 segments; after each of the
     first K the wrist view is re-segmented, matched to the external proposals, and the VLM is asked again on that
     view (one more vote). The target is the vote leader (ties go to the newest, closer view); a change re-plans the
     rest of the path to the new target.
  3. look (22 cm above the target): a Yes/No check on the wrist close-up. Below `tau`, the target is rejected and the
     VLM is asked again on the external view without the rejected proposals; the arm goes to look at the new one
     (at most R retargets). If every check fails, the best-scoring visited target is grasped.
  4. grasp re-planned from the wrist view (as in look_refine) and executed.

K, verify, R, tau are the budget knobs; every VLM call's latency is logged so configs can be compared at equal cost.
`vlm` is anything with .ground(mode, obj, frame, props) and .verify(obj, frame, box) (vrg.franka.vlm_client.RemoteVLM).
"""
import collections, time
import numpy as np
from vrg.franka import kinematics as K
from vrg.franka.episode import look_pose, match_wrist, execute, move
from vrg.franka.perception import proposals, top_down_grasp


def wrist_view_props(F, props, max_d=0.05):
    """External proposals as seen by the wrist camera: wrist proposals matched by xy (nearest, within max_d), carrying
    the external proposal's id and the wrist pixels. Unmatched external proposals are simply not in view."""
    props_w, _ = proposals({"wrist": F["wrist"]}); out, used = [], set()
    for p in props:
        if not props_w: break
        d = [np.linalg.norm(q["center"][:2] - p["center"][:2]) for q in props_w]; k = int(np.argmin(d))
        if d[k] <= max_d and k not in used: used.add(k); out.append(dict(props_w[k], id=p["id"]))       # wrist pixels and points, external id
    return out


def view_quality(frame, cand, edge=3):
    """Image-only evidence of how well each candidate is seen (no ground truth): mask area as a fraction of the image,
    and whether its box is cut off by the image border. Used to tell when a closer view stops being a better view."""
    H, W = frame.rgb.shape[:2]; q = {}
    for c in cand:
        (x0, y0, x1, y1), m = c["pix"][frame.name]
        q[c["id"]] = dict(area=float(m.sum()) / (H * W), trunc=bool(x0 <= edge or y0 <= edge or x1 >= W - 1 - edge or y1 >= H - 1 - edge))
    return q


GATES = {  # "this wrist view is obscured for the approached object" (image evidence only; chosen on dev probes)
    "trunc": lambda q, n: q["trunc"],
    "lt3": lambda q, n: n < 3,
    "area5": lambda q, n: q["area"] > 0.05,
    "area8": lambda q, n: q["area"] > 0.08,
}


def leader(votes, rejected, last):
    c = {k: v for k, v in votes.items() if k not in rejected and k is not None}
    if not c: return None
    best = max(c.values()); tied = [k for k, v in c.items() if v == best]
    return last if last in tied else tied[0]


def run(robot, desc, vlm, mode, K_app=2, verify=False, R=2, tau=0.0, ext_modes=None, gate=None, rule="vote", grasp_mode="pca", controller="cartesian", gt=None):
    """mode: the model's best single mode (used for the wrist views); ext_modes: modes asked on the external view, the
    best one last so that it breaks ties (default [mode]). gate: a key of GATES; once a wrist view is obscured for the
    approached object, later views are not asked (the decision stays with the unobscured ones), and if the target is
    cut off at the look pose, the grasp is planned from its last fully visible wrist detection.
    rule: "vote" (leader of all votes so far, ties to the newest) or "last" (the newest unobscured view's pick decides). verify defaults to off: on dev the close-up Yes/No score was
    too poorly calibrated (AUC ~0.78) to reject targets with a fixed threshold."""
    ext_modes = ext_modes or [mode]
    rec = dict(target=desc, grasp_mode=grasp_mode, controller=controller, fail=None, t={}, reactive=dict(K=K_app, verify=verify, R=R, tau=tau, ext_modes=ext_modes, gate=gate, rule=rule))
    frozen, last_clear = False, {}                                  # last_clear: id -> last wrist detection not cut by the border
    hist, vlm_sec = [], 0.0
    lab = {}

    def ask(stage, frame, cand, m=None):
        nonlocal vlm_sec
        r = vlm.ground(m or mode, desc, frame, cand); vlm_sec += r["sec"]
        hist.append(dict(stage=stage, mode=m or mode, choice=r["choice"], obj=lab.get(r["choice"]), sec=r["sec"], n_cand=len(cand))); return r["choice"]

    t0 = time.time(); robot.home(); robot.gripper_move(0.08)
    F0 = robot.frames(); props, table = proposals({"ext": F0["ext"]})
    rec["_frames"], rec["_props"], rec["n_props"] = F0, props, len(props)
    if gt is not None:
        lab = gt.label(F0, props); rec["prop_labels"] = lab; rec["target_segmented"] = any(v == gt.target for v in lab.values())
    P = {p["id"]: p for p in props}
    votes, last = collections.Counter(), None
    for m in ext_modes:
        c = ask("ext", F0["ext"], props, m)
        if c is not None: votes[c] += 1; last = c
    first = leader(votes, set(), last); target = first
    rec["initial_choice"], rec["initial_obj"] = first, lab.get(first)
    rec["single_choice_obj"] = hist[-1]["obj"]                                     # the best mode alone (one-shot baseline)
    rejected, scores, retargets = set(), {}, 0
    if target is None: return _finish_grounding_fail(rec, gt, hist, vlm_sec)

    # 2. approach with K re-grounding checkpoints
    q0 = robot.state().q; Q, _ = K.plan_joint(q0, look_pose(P[target]))
    if Q is None: rec["fail"] = "ik"; return _done(rec, hist, vlm_sec)
    for k in range(K_app):
        b = len(Q) // (K_app + 1 - k)                                                # remaining path split evenly over remaining segments
        robot.follow_joint_trajectory(Q[:b], dt=0.01); Q = Q[b:]
        F = robot.frames(); cand = [c for c in wrist_view_props(F, props) if c["id"] not in rejected]
        Qv = view_quality(F["wrist"], cand)
        for c in cand:
            if not Qv[c["id"]]["trunc"]: last_clear[c["id"]] = c
        if frozen: hist.append(dict(stage=f"approach{k}", choice=None, skipped="frozen", n_cand=len(cand))); continue
        if target not in {c["id"] for c in cand} or len(cand) < 2:                  # target out of view: no evidence here
            hist.append(dict(stage=f"approach{k}", choice=None, skipped=True, n_cand=len(cand))); continue
        if gate and GATES[gate](Qv[target], len(cand)):                             # closer views only get worse from here
            frozen = True; hist.append(dict(stage=f"approach{k}", choice=None, skipped="obscured", n_cand=len(cand), quality=Qv[target])); continue
        c = ask(f"approach{k}", F["wrist"], cand); votes[c] += 1
        hist[-1].update(quality={str(i): v for i, v in view_quality(F["wrist"], cand).items()}, height=float(robot.state().T_tcp[2, 3]), target_before=target)
        new = (c if c is not None and c not in rejected else target) if rule == "last" else leader(votes, rejected, c)
        if new is not None and new != target:
            target = new; Q, _ = K.plan_joint(robot.state().q, look_pose(P[target]))
            if Q is None: rec["fail"] = "ik"; return _done(rec, hist, vlm_sec)
    if len(Q): robot.follow_joint_trajectory(Q, dt=0.01)

    # 3. look + verify, with retargeting
    g = None
    while True:
        r, _ = move(robot, look_pose(P[target]), "joint", straight=False)          # no-op if already there
        F = robot.frames(); _, qw, _ = match_wrist(F, P[target])
        if qw is None:                                                              # not visible from above: keep, grasp from ext
            hist.append(dict(stage="look", target=target, obj=lab.get(target), seen=False)); g = top_down_grasp(P[target], table, yaw_mode=grasp_mode); break
        if gate and view_quality(F["wrist"], [dict(qw, id=target)])[target]["trunc"] and target in last_clear:
            qw = last_clear[target]; hist.append(dict(stage="look", target=target, grasp_from="last_clear"))   # cut off at the look pose
        gw = top_down_grasp(qw, table, yaw_mode=grasp_mode)
        if not verify: g = gw; break
        v = vlm.verify(desc, F["wrist"], qw["pix"]["wrist"][0]); vlm_sec += v["sec"]; scores[target] = (v["score"], gw)
        hist.append(dict(stage="verify", target=target, obj=lab.get(target), score=v["score"], sec=v["sec"]))
        if v["score"] >= tau: g = gw; break
        rejected.add(target)
        if retargets >= R or len(rejected) >= len(props):                           # out of budget: best-scoring visited target
            target = max(scores, key=lambda t: scores[t][0]); g = scores[target][1]; hist.append(dict(stage="fallback", target=target, obj=lab.get(target))); break
        retargets += 1
        nxt = ask("reground", F0["ext"], [p for p in props if p["id"] not in rejected])
        if nxt is None: target = max(scores, key=lambda t: scores[t][0]); g = scores[target][1]; break
        target = nxt
    rec["retargets"] = retargets; rec["final_choice"] = target
    if gt is not None:
        rec["chosen_obj"] = lab.get(target); rec["grounding_correct"] = rec["chosen_obj"] == gt.target
        rec["initial_correct"] = rec["initial_obj"] == gt.target
        if not rec["target_segmented"]: rec["fail"] = "segmentation"
        elif not rec["grounding_correct"]: rec["fail"] = "grounding"
    rec["t"]["approach"] = time.time() - t0
    _done(rec, hist, vlm_sec)
    return execute(robot, g, rec, controller, gt)


def _done(rec, hist, vlm_sec):
    rec["vlm_hist"] = hist; rec["vlm_sec"] = vlm_sec; rec["n_vlm"] = sum(1 for h in hist if "sec" in h); return rec


def _finish_grounding_fail(rec, gt, hist, vlm_sec):
    rec["fail"] = "segmentation" if (gt is not None and not rec["target_segmented"]) else "grounding"
    rec["chosen_obj"] = None; rec["grounding_correct"] = False; rec["initial_correct"] = False
    return _done(rec, hist, vlm_sec)


def probe(robot, desc, vlm, mode, K_app=4, gt=None):
    """Signal-logging episode for designing the reactive rule on dev scenes: follow the initial choice to its look pose
    (no retargeting, no grasp) and log, at each of K checkpoints and at the look pose, the VLM's pick among the visible
    proposals and the Yes/No score of every visible proposal. Also logs Yes/No scores on the external crops."""
    rec = dict(target=desc, probe=True, fail=None); lab = {}
    robot.home(); robot.gripper_move(0.08)
    F0 = robot.frames(); props, table = proposals({"ext": F0["ext"]}); P = {p["id"]: p for p in props}
    rec["_frames"], rec["_props"], rec["n_props"] = F0, props, len(props)
    if gt is not None: lab = gt.label(F0, props); rec["prop_labels"] = lab; rec["target_segmented"] = any(v == gt.target for v in lab.values())
    views = []

    def log_view(stage, frame, cand, height):
        v = dict(stage=stage, height=height, visible=[c["id"] for c in cand], choice=None, yes={},
                 quality={str(i): x for i, x in view_quality(frame, cand).items()})
        if cand:
            r = vlm.ground(mode, desc, frame, cand); v["choice"], v["sec"] = r["choice"], r["sec"]
            for c in cand:
                s = vlm.verify(desc, frame, c["pix"][frame.name][0]); v["yes"][c["id"]] = s["score"]; v["sec_verify"] = v.get("sec_verify", 0) + s["sec"]
        views.append(v)

    log_view("ext", F0["ext"], props, None)
    first = views[0]["choice"]; rec["initial_choice"] = first
    if first is None: rec["views"] = views; return rec
    Q, _ = K.plan_joint(robot.state().q, look_pose(P[first]))
    if Q is None: rec["views"] = views; rec["fail"] = "ik"; return rec
    for k in range(K_app + 1):
        b = len(Q) // (K_app + 1 - k); robot.follow_joint_trajectory(Q[:b], dt=0.01); Q = Q[b:]
        F = robot.frames(); h = float(robot.state().T_tcp[2, 3])
        log_view("look" if k == K_app else f"approach{k}", F["wrist"], wrist_view_props(F, props), h)
    rec["views"] = views
    return rec
