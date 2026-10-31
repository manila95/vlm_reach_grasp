"""Merge LIBERO-Plus result shards into runs/liberoplus/<method>/results.json, plus a per-task grounding summary for ours
(target found, basket position error) in runs/liberoplus/ours/grounding.json (liberoplus env: the grounding pickle needs numpy).
  $LPY -m vrg.lplus.collect
"""
import glob, json, os

S = "/scratch/kaustubh/vlm_reach_grasp/liberoplus/runs"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    for d in sorted(glob.glob(f"{S}/*")):
        name = os.path.basename(d)
        if name.startswith(("_", "smoke")): continue
        R = {}
        for f in glob.glob(f"{d}/results_*.json"):
            for r in json.load(open(f)): R[r["tid"]] = {k: v for k, v in r.items() if k not in ("phases",)}
        if not R: continue
        os.makedirs(f"{ROOT}/runs/liberoplus/{name}", exist_ok=True)
        json.dump([R[k] for k in sorted(R)], open(f"{ROOT}/runs/liberoplus/{name}/results.json", "w"), indent=0, default=float)
        if name == "ours" and os.path.exists(f"{os.path.dirname(S)}/grounding_all.pkl"):
            import pickle, numpy as np
            G = pickle.load(open(f"{os.path.dirname(S)}/grounding_all.pkl", "rb")); out = []
            for (tid, _), g in sorted(G.items()):
                b, gb = g.get("basket"), g.get("gt_basket")
                out.append(dict(tid=tid, obj_hit=bool(g.get("obj_hit")), error=g.get("error"),
                                basket_err=float(np.linalg.norm((b - gb)[:2])) if b is not None and gb is not None else None))
            json.dump(out, open(f"{ROOT}/runs/liberoplus/ours/grounding.json", "w"), indent=0)
        print(f"{name:20s} {len(R):5d} tasks, success {sum(r['success'] for r in R.values()) / len(R):.3f}")


if __name__ == "__main__":
    main()
