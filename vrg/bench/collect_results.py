"""Copy the RoboCasa / Colosseum evaluation results from /scratch into runs/eval/ so the report can read them from git.

  python -m vrg.bench.collect_results            # (vlmrg env) idempotent; re-run after more episodes finish

Writes
  runs/eval/rc_<method>_<split>/results.json     all 5 tasks merged (method: ours, e2e, saycan, hamster)
  runs/eval/col_<method>/results.json            all tasks x factors merged (method: ours, e2e, hamster)
  runs/eval/rc_grounding.json                    per-episode VLM pick grounding (hit, error) for both splits
  runs/eval/col_grounding.json                   per-episode Colosseum grounding: found?, error vs the demo's first keyframe
  demo/bench/rc_<split>_<task>.mp4               the 4 methods side by side on the same episode (ep 0)
"""
import glob, json, os, pickle
import numpy as np

S = "/scratch/kaustubh/vlm_reach_grasp"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL = f"{ROOT}/runs/eval"
TASKS = ["PickPlaceCounterToCabinet", "PickPlaceCounterToStove", "PickPlaceDrawerToCounter",
         "PickPlaceSinkToCounter", "PickPlaceToasterToCounter"]
OURS_RC = os.environ.get("OURS_RC", "ours")      # scratch eval dir prefix holding the final RoboCasa run of ours


def dump(name, rows):
    os.makedirs(f"{EVAL}/{name}", exist_ok=True)
    json.dump(rows, open(f"{EVAL}/{name}/results.json", "w"), indent=0)
    print(f"{name:24s} {sum(r['success'] for r in rows):4d}/{len(rows)}")


def robocasa():
    for m in ["ours", "e2e", "saycan", "hamster"]:
        for sp in ["pretrain", "target"]:
            src = f"{S}/robocasa/eval/{OURS_RC if m == 'ours' else m}_{sp}"
            rows = sum((json.load(open(f)) for f in sorted(glob.glob(f"{src}/*/results.json"))), [])
            if rows: dump(f"rc_{m}_{sp}", rows)
    G = pickle.load(open(f"{S}/robocasa/grounding.pkl", "rb"))
    json.dump([dict(split=sp, task=t, ep=e, hit=bool(g["pick_hit"]), err=float(g["pick_err"]), pick=g["pick_name"])
               for (sp, t, e), g in sorted(G.items())], open(f"{EVAL}/rc_grounding.json", "w"), indent=0)


def colosseum():
    for m in ["ours", "e2e", "hamster"]:
        rows = sum((json.load(open(f)) for f in sorted(glob.glob(f"{S}/colosseum/eval/{m}/w*/results.json"))), [])
        if rows: dump(f"col_{m}", rows)
    G = pickle.load(open(f"{S}/colosseum/grounding.pkl", "rb"))
    off = {t: np.array(v["off_k0"]) for t, v in json.load(open(f"{S}/colosseum/col_stats.json")).items()}
    out = []
    for (kind, task, factor, ep), g in sorted(G.items()):
        if kind != "test": continue
        ok = g.get("reach") is not None
        out.append(dict(task=task, factor=int(factor), ep=int(ep), grounded=ok,
                        err=float(np.linalg.norm(np.asarray(g["reach"]) + off[task] - np.asarray(g["gt_k0"])[:3])) if ok else None))
    json.dump(out, open(f"{EVAL}/col_grounding.json", "w"), indent=0)


def side_by_side(ep=0):
    """2x2 grid (ours | e2e / saycan | hamster) of the agentview+wrist eval clips, padded to the longest clip."""
    import imageio.v2 as iio, cv2
    os.makedirs(f"{ROOT}/demo/bench", exist_ok=True)
    for sp in ["pretrain", "target"]:
        for t in TASKS:
            clips = []
            for m in ["ours", "e2e", "saycan", "hamster"]:
                d = f"{S}/robocasa/eval/{OURS_RC if m == 'ours' else m}_{sp}/{t}/videos"
                f = sorted(glob.glob(f"{d}/{t}_e{ep}_*.mp4"))
                fr = [x for x in iio.mimread(f[0], memtest=False)] if f else [np.zeros((256, 512, 3), np.uint8)]
                ok = bool(f) and f[0].endswith("_ok.mp4")
                for x in fr:
                    cv2.rectangle(x, (0, 0), (511, 20), (0, 0, 0), -1)
                    cv2.putText(x, f"{m}: {'SUCCESS' if ok else 'fail' if f else 'no video'}", (6, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 220, 90) if ok else (240, 120, 90), 1, cv2.LINE_AA)
                clips.append(fr)
            n = max(len(c) for c in clips)
            clips = [c + [c[-1]] * (n - len(c)) for c in clips]
            out = f"{ROOT}/demo/bench/rc_{sp}_{t[9:]}.mp4"
            w = iio.get_writer(out, fps=40, quality=4, macro_block_size=8)      # 2x speed
            for i in range(0, n):
                w.append_data(np.vstack([np.hstack([clips[0][i], clips[1][i]]), np.hstack([clips[2][i], clips[3][i]])]))
            w.close(); print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    robocasa(); colosseum(); side_by_side()
