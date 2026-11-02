"""E8b: turn a grounding-study result (vrg/lplus/ground_study.py jsonl) into the grounding file the closed-loop runner reads,
so every model x strategy can be run with the robust grasp skill (vrg.lplus.run --grounding <pkl>).

Per scene: the chosen pick / place proposals of the cached scene (grounding-v6 proposals: robot self-filter, box centres)
become obj / basket = [x, y, top / 2], exactly as vrg/lplus/ground.py. Ground-truth fields are copied for analysis only.
A scene with no valid choice gets obj = None (the closed-loop run skips it; the analysis counts it as a failure).
  $LPY -m vrg.lplus.study_to_grounding --study <jsonl> --cache $S/liberoplus/scene_cache --out <pkl>
"""
import argparse, json, pickle
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--study", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True); A = ap.parse_args()
    G = {}
    for l in open(A.study):
        r = json.loads(l); tid = r["tid"]; z = np.load(f"{A.cache}/scene_{tid}.npz", allow_pickle=True)
        gt = json.loads(str(z["gt"])); tgt = str(z["target"]); ch = r["chosen"]
        pt = lambda i: None if i is None else np.array([*z["xy"][i], z["top"][i] / 2])
        G[(tid, 0)] = dict(tid=tid, ep=0, calib=False, language=str(z["language"]), name=str(z["name"]),
                           obj=pt(ch[0]), basket=pt(ch[1]), obj_hit=bool(r["pick_ok"]),
                           gt_obj=np.array(gt[tgt]) if tgt in gt else None, gt_basket=np.array(gt["basket_1"]),
                           gt_all={k: np.array(v) for k, v in gt.items()})
    pickle.dump(G, open(A.out, "wb")); print("wrote", len(G), "scenes to", A.out)


if __name__ == "__main__":
    main()
