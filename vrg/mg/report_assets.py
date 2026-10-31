"""Collect MimicGen study results and make report media (vlmrg env).

* runs/mimicgen/<run>/eval_whole_d{0,1,2}/results.json : whole-trajectory evaluations. Composed evaluations are NOT
  collected while the composed harness is being debugged (vrg/mg/evaluate.py; numbers so far are invalid).
* runs/mimicgen/segment_stats.json : per dataset, contact-phase statistics; heuristic-segmenter validation.
* demo/mimicgen/contact_phases_<task>.png : one demo's frames with its subtasks and contact phases marked.
* demo/mimicgen/whole_d0_vs_d1_<task>.mp4 (+ .png still) : the whole-trajectory policy on D0 and D1, episode 0 of each
  (rule-selected), side by side, upscaled.
  python -m vrg.mg.report_assets
"""
import glob, json, os, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

S = "/scratch/kaustubh/vlm_reach_grasp/mimicgen"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS = ["square", "threading", "coffee", "three_piece_assembly", "mug_cleanup"]


def font(n):
    try: return ImageFont.truetype("DejaVuSans-Bold.ttf", n)
    except OSError: return ImageFont.load_default()


def collect():
    for f in glob.glob(f"{S}/runs/*_s[0-9]/eval_whole_d[0-9]/results.json"):
        run, ev = f.split("/")[-3], f.split("/")[-2]
        os.makedirs(f"{ROOT}/runs/mimicgen/{run}/{ev}", exist_ok=True); shutil.copy(f, f"{ROOT}/runs/mimicgen/{run}/{ev}/results.json")
    from vrg.mg.heuristic_segment import estimate
    stats = {}
    for f in sorted(glob.glob(f"{S}/segments/*_d0.json")):
        name = os.path.basename(f)[:-5]; D = json.load(open(f)); dm = D["demos"]
        if not os.path.exists(f"{S}/cache/{name}/index.json"): continue
        st = dict(n_demos=len(dm), spec=D["spec"], subtasks=[])
        T = np.array([d["T"] for d in dm.values()])
        for k, (ref, sig) in enumerate(D["spec"]):
            c = [d["subtasks"][k] for d in dm.values()]
            L = [s["end"] - s["contact"] + 1 for s in c if s["contact"] is not None]
            st["subtasks"].append(dict(ref=ref, signal=sig, contact_found=sum(s["contact"] is not None for s in c),
                                       contact_to_end_median=float(np.median(L)), subtask_len_median=float(np.median([s["end"] - s["start"] + 1 for s in c]))))
        seg = sum(sum(s["end"] - max(s["start"], s["contact"] - 10) + 1 for s in d["subtasks"] if s["contact"] is not None) for d in dm.values())
        st["segment_frame_share_pre10"] = seg / T.sum()
        # heuristic (DROID-style) validation
        I = json.load(open(f"{S}/cache/{name}/index.json")); P = np.load(f"{S}/cache/{name}/proprio.npy", mmap_mode="r"); A = np.load(f"{S}/cache/{name}/action.npy", mmap_mode="r")
        err = {}
        for di, dk in enumerate(I["demos"]):
            d = dm[dk]; o, Tn = I["offsets"][di], I["T"][di]; est = estimate(np.asarray(P[o:o + Tn, :3]), np.asarray(A[o:o + Tn, 6]))
            for k, s in enumerate(d["subtasks"]):
                if s["contact"] is None: continue
                kind = "grasp" if s["signal"] in ("grasp", "grasp_1", "grasp_2") else ("place" if s["held"] else "push")
                cands = [t for t, kd in est if kd == kind and s["start"] - 5 <= t <= s["end"] + 5]
                err.setdefault(f"{k}:{s['ref']} ({kind})", []).append(min(cands, key=lambda t: abs(t - s["contact"])) - s["contact"] if cands else None)
        st["heuristic"] = {k: dict(found=sum(x is not None for x in e) / len(e), median_err=float(np.median([x for x in e if x is not None])),
                                   within10=float(np.mean([x is not None and abs(x) <= 10 for x in e]))) for k, e in err.items()}
        stats[name] = st; print("stats", name, flush=True)
    os.makedirs(f"{ROOT}/runs/mimicgen", exist_ok=True); json.dump(stats, open(f"{ROOT}/runs/mimicgen/segment_stats.json", "w"), indent=1)


def phase_strip(task, demo=0, n=8, scale=3):
    name = f"{task}_d0"; D = json.load(open(f"{S}/segments/{name}.json")); I = json.load(open(f"{S}/cache/{name}/index.json"))
    dk = I["demos"][demo]; subs = D["demos"][dk]["subtasks"]; o, T = I["offsets"][demo], I["T"][demo]
    A = np.load(f"{S}/cache/{name}/agent.npy", mmap_mode="r")
    picks = sorted({0, T - 1} | {s["start"] for s in subs} | {s["contact"] for s in subs if s["contact"] is not None} |
                   {max(s["start"], s["contact"] - 10) for s in subs if s["contact"] is not None})
    tiles = []
    for t in picks:
        k = next(i for i, s in enumerate(subs) if s["start"] <= t <= s["end"]); s = subs[k]
        learned = s["contact"] is not None and t >= max(s["start"], s["contact"] - 10)
        im = Image.fromarray(np.asarray(A[o + t])).resize((84 * scale, 84 * scale), Image.NEAREST)
        canvas = Image.new("RGB", (im.width, im.height + 64), (251, 251, 248)); canvas.paste(im, (0, 64)); d = ImageDraw.Draw(canvas)
        col = (235, 104, 52) if learned else (42, 120, 214)
        d.rectangle([0, 56, im.width, 63], fill=col)
        d.text((6, 4), f"t={t}  {s['ref']}", fill=(20, 20, 20), font=font(18))
        d.text((6, 28), "learned contact phase" if learned else "VLM + planned reach", fill=col, font=font(17))
        tiles.append(canvas)
    W = sum(t.width for t in tiles) + 8 * (len(tiles) - 1); out = Image.new("RGB", (W, tiles[0].height), (251, 251, 248)); x = 0
    for t in tiles: out.paste(t, (x, 0)); x += t.width + 8
    os.makedirs(f"{ROOT}/demo/mimicgen", exist_ok=True); out.save(f"{ROOT}/demo/mimicgen/contact_phases_{task}.png"); print("strip", task)


def side_by_side(task, scale=4):
    import imageio.v2 as io
    clips = []
    for v, lab in ((0, "D0 (trained)"), (1, "D1 (wider placement)")):
        f = sorted(glob.glob(f"{S}/runs/{task}_whole_s0/eval_whole_d{v}/videos/0_*.mp4"))
        if not f: return
        ok = f[0].endswith("_ok.mp4"); fr = [np.asarray(x) for x in io.mimread(f[0], memtest=False)]
        clips.append((fr, f"{lab}: {'success' if ok else 'failure'}", ok))
    n = max(len(c[0]) for c in clips); out = []
    for i in range(n):
        tiles = []
        for fr, lab, ok in clips:
            im = Image.fromarray(fr[min(i, len(fr) - 1)]).resize((84 * scale, 84 * scale), Image.BICUBIC)
            c = Image.new("RGB", (im.width, im.height + 40), (251, 251, 248)); c.paste(im, (0, 40)); d = ImageDraw.Draw(c)
            d.text((8, 10), lab, fill=(27, 140, 90) if ok else (200, 60, 40), font=font(17)); tiles.append(np.asarray(c))
        out.append(np.hstack([tiles[0], np.full((tiles[0].shape[0], 8, 3), 251, np.uint8), tiles[1]]))
    p = f"{ROOT}/demo/mimicgen/whole_d0_vs_d1_{task}.mp4"; io.mimsave(p, out, fps=20, quality=7, macro_block_size=1)
    Image.fromarray(out[min(60, n - 1)]).save(p[:-4] + ".png"); print("clip", task, n)


if __name__ == "__main__":
    collect()
    for t in ["square", "coffee", "threading"]: phase_strip(t); side_by_side(t)
