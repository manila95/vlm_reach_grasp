"""Contact-onset estimation from DROID-style signals only (end-effector positions and gripper commands; no contact
sensing, no object poses), validated against the ground-truth contact onsets of vrg.mg.segment (any env with numpy).

Rules, applied per demo:
  * gripper close command (action[-1] crosses 0 upward) -> a grasp contact at that step;
  * while holding (between a close and the next open, or the end), the transport ends in a slow-down: the first step after
    the transport's speed peak where the smoothed speed falls below alpha x that peak -> a place/insert contact;
  * before the first close, if the hand slows down after a fast segment and later speeds up again without closing the
    gripper, that slow-down is a non-prehensile contact (e.g. pushing a drawer).
Each ground-truth contact onset is matched to the nearest estimate of the same kind; we report the error distribution.
  python -m vrg.mg.heuristic_segment --datasets square_d0,threading_d0
"""
import argparse, json
import numpy as np

S = "/scratch/kaustubh/vlm_reach_grasp/mimicgen"


def smooth(x, k=5):
    return np.convolve(x, np.ones(k) / k, mode="same")


def estimate(pos, grip_cmd, alpha=0.3, min_peak=0.004):
    """pos (T,3), grip_cmd (T,) in [-1, 1]. Returns list of (t, kind) with kind in {grasp, place, push}."""
    v = smooth(np.linalg.norm(np.diff(pos, axis=0, prepend=pos[:1]), axis=1))
    closed = grip_cmd > 0; ev = []
    closes = list(np.nonzero(~closed[:-1] & closed[1:])[0] + 1); opens = list(np.nonzero(closed[:-1] & ~closed[1:])[0] + 1)
    for tc in closes: ev.append((int(tc), "grasp"))
    for tc in closes:                                                  # transport while holding -> slow-down = place / insert
        to = next((o for o in opens if o > tc), len(pos) - 1)
        seg = v[tc:to]
        if len(seg) < 10: continue
        pk = int(np.argmax(seg)); peak = seg[pk]
        if peak < min_peak: continue
        slow = np.nonzero(seg[pk:] < alpha * peak)[0]
        if len(slow): ev.append((int(tc + pk + slow[0]), "place"))
    first = closes[0] if closes else len(pos) - 1                      # before any grasp: slow-down then speed-up = push
    seg = v[:first]
    if len(seg) > 20:
        pk = int(np.argmax(seg[: len(seg) // 2 + 1])); peak = seg[pk]
        slow = np.nonzero(seg[pk:] < alpha * peak)[0]
        if peak > min_peak and len(slow) and pk + slow[0] < first - 15: ev.append((int(pk + slow[0]), "push"))
    return sorted(ev)


def validate(name):
    SEG = json.load(open(f"{S}/segments/{name}.json")); I = json.load(open(f"{S}/cache/{name}/index.json"))
    P = np.load(f"{S}/cache/{name}/proprio.npy", mmap_mode="r"); A = np.load(f"{S}/cache/{name}/action.npy", mmap_mode="r")
    err = {}
    for di, dk in enumerate(I["demos"]):
        d = SEG["demos"][dk]
        if not d["success"]: continue
        o, T = I["offsets"][di], I["T"][di]; est = estimate(np.asarray(P[o:o + T, :3]), np.asarray(A[o:o + T, 6]))
        held = None
        for k, s in enumerate(d["subtasks"]):
            if s["contact"] is None: continue
            kind = "grasp" if s["signal"] in ("grasp", "grasp_1", "grasp_2") else ("place" if s["held"] else "push")
            cands = [t for t, kd in est if kd == kind and s["start"] - 5 <= t <= s["end"] + 5]
            key = f"{k}:{s['ref']}({kind})"
            err.setdefault(key, []).append(min(cands, key=lambda t: abs(t - s["contact"])) - s["contact"] if cands else None)
    print(f"== {name}")
    for key, e in err.items():
        found = [x for x in e if x is not None]
        print(f"  {key:28s} found {len(found)}/{len(e)}  error (steps) median {np.median(found) if found else float('nan'):+.0f}, "
              f"|err|<=5: {np.mean([abs(x) <= 5 for x in found]) if found else 0:.2f}, |err|<=10: {np.mean([abs(x) <= 10 for x in found]) if found else 0:.2f}")
    return err


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", required=True); A_ = ap.parse_args()
    for n in A_.datasets.split(","): validate(n)
