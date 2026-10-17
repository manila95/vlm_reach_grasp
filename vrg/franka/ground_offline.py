"""Offline VLM grounding sweep on saved scenes (vlmrg env): every model x mode sees exactly the same observations.

Scenes come from `run_isaac --save_scenes DIR` (or, later, from the real robot's logger). For each scene the geometric
proposals are recomputed (deterministic), labelled with ground-truth masks, and each VLM picks one in each mode.
Output: <out>/<model>_<mode>.json  {"seed|target": {xyz, choice, correct, chosen_obj, raw, sec}}  (+ summary.json)

  python -m vrg.franka.ground_offline --scenes /scratch/.../isaac/scenes --models qwen3vl-2b,qwen3vl-4b --modes point,marks,verify
"""
import argparse, glob, json, os, pickle, time
import numpy as np
from vrg.franka.core import CameraFrame
from vrg.franka.perception import proposals
from vrg.franka.vlm import VLM, ground


def load_scene(path):
    S = pickle.load(open(path, "rb"))
    F = {k: CameraFrame(k, f["rgb"], f["depth"], f["K"], f["T"]) for k, f in S["frames"].items()}
    H, W = F["ext"].depth.shape
    masks = {k: {n: np.unpackbits(m)[:H * W].reshape(H, W).astype(bool) for n, m in mm.items()} for k, mm in S["masks"].items()}
    props, _ = proposals({"ext": F["ext"]})
    if S.get("prop_labels"):                                   # labels from the live run (fused ext + wrist ground truth): more
        return S, F, props, {p["id"]: S["prop_labels"].get(str(p["id"]), S["prop_labels"].get(p["id"])) for p in props}   # reliable than ext-only masks
    lab = {}
    for p in props:
        _, m = p["pix"]["ext"]; cnt = {n: int((m & om).sum()) for n, om in masks["ext"].items()}
        best = max(cnt, key=cnt.get); lab[p["id"]] = best if cnt[best] / max(m.sum(), 1) >= 0.5 else None
    return S, F, props, lab


def rescore(out, scenes_dir):
    """Recompute correct/segmented/chosen_obj of existing results with the current labels (no VLM calls)."""
    L = {}
    for p in glob.glob(f"{scenes_dir}/*.pkl"):
        S, _, props, lab = load_scene(p); L[f"{S['seed']}|{S['target']}"] = (S["target"], lab)
    summ = {}
    for f in sorted(glob.glob(f"{out}/qwen3vl-*_*.json")):
        R = json.load(open(f))
        for k, r in R.items():
            tgt, lab = L[k]; r.update(chosen_obj=lab.get(r["choice"]), correct=lab.get(r["choice"]) == tgt, segmented=tgt in lab.values())
        json.dump(R, open(f, "w"), indent=0); seg = [r for r in R.values() if r["segmented"]]
        summ[os.path.basename(f)[:-5]] = dict(n=len(R), acc=float(np.mean([r["correct"] for r in R.values()])),
                                              acc_given_segmented=float(np.mean([r["correct"] for r in seg])), sec=float(np.mean([r["sec"] for r in R.values()])))
        print(f"{os.path.basename(f):28s} acc {summ[os.path.basename(f)[:-5]]['acc']:.3f}")
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1)


# Appearance-describing names (what a person looking at the object might say), to test whether confusions come from the
# YCB category names (e.g. "potted meat can" is a SPAM can; "gelatin" and "pudding" boxes are both Jell-O) or from vision.
DESCRIPTIVE = {"003_cracker_box": "red Cheez-It cracker box", "004_sugar_box": "yellow Domino sugar box",
               "005_tomato_soup_can": "red and white Campbell's tomato soup can", "006_mustard_bottle": "yellow French's mustard bottle",
               "008_pudding_box": "brown Jell-O chocolate pudding box", "009_gelatin_box": "red Jell-O strawberry gelatin box",
               "010_potted_meat_can": "blue SPAM can", "011_banana": "banana", "021_bleach_cleanser": "white bleach bottle",
               "037_scissors": "scissors", "040_large_marker": "black marker pen", "051_large_clamp": "black clamp",
               "061_foam_brick": "red foam brick"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scenes", required=True); ap.add_argument("--models", default="qwen3vl-2b,qwen3vl-4b,qwen3vl-8b,qwen3vl-32b")
    ap.add_argument("--modes", default="point,marks,verify"); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--names", default="ycb", choices=["ycb", "descriptive"], help="how the target is named in the prompt")
    ap.add_argument("--rescore", action="store_true", help="only recompute correctness of existing results with the current labels")
    A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
    if A.rescore: return rescore(A.out, A.scenes)
    scenes = [load_scene(p) for p in sorted(glob.glob(f"{A.scenes}/*.pkl"))]
    print(f"{len(scenes)} scenes; target segmented in {sum(any(v == S['target'] for v in lab.values()) for S, _, _, lab in scenes)}", flush=True)
    summ = json.load(open(f"{A.out}/summary.json")) if os.path.exists(f"{A.out}/summary.json") else {}
    for model in A.models.split(","):
        V = VLM(model, device=A.device)
        for mode in A.modes.split(","):
            outf = f"{A.out}/{model}_{mode}.json"
            if os.path.exists(outf): continue
            R = {}
            for S, F, props, lab in scenes:
                r = ground(V, mode, DESCRIPTIVE[S["target"]] if A.names == "descriptive" else S["desc"], F["ext"], props)
                p = next((q for q in props if q["id"] == r["choice"]), None)
                R[f"{S['seed']}|{S['target']}"] = dict(model=model, mode=mode, choice=r["choice"], xyz=None if p is None else p["center"].tolist(),
                                                       chosen_obj=lab.get(r["choice"]), correct=lab.get(r["choice"]) == S["target"],
                                                       segmented=any(v == S["target"] for v in lab.values()), raw=r["raw"][:200], sec=r["sec"])
            json.dump(R, open(outf, "w"), indent=0)
            acc = np.mean([r["correct"] for r in R.values()]); seg = [r for r in R.values() if r["segmented"]]
            summ[f"{model}_{mode}"] = dict(n=len(R), acc=float(acc), acc_given_segmented=float(np.mean([r["correct"] for r in seg])) if seg else None,
                                           sec=float(np.mean([r["sec"] for r in R.values()])))
            json.dump(summ, open(f"{A.out}/summary.json", "w"), indent=1)
            print(f"{model:12s} {mode:7s} acc {acc:.3f} ({len(R)} scenes)  {summ[f'{model}_{mode}']['sec']:.2f}s/query", flush=True)
        del V
        import torch; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
