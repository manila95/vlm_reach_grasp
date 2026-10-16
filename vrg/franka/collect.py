"""Copy Franka/Isaac results from /scratch into runs/franka/ (small JSON) and demo/franka/ (a few videos) for the notebook.

  python -m vrg.franka.collect
"""
import glob, json, os, shutil

S = "/scratch/kaustubh/vlm_reach_grasp/isaac"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/runs/franka"


def main():
    os.makedirs(OUT, exist_ok=True)
    for d in sorted(glob.glob(f"{S}/runs/*/results.json")):
        name = d.split("/")[-2]
        if name.startswith("dev"): continue                     # dev runs (seeds 0-9) stay on /scratch
        R = json.load(open(d))
        for r in R:                                              # keep the record small
            r.pop("grasp", None); r.get("grounding", {}).pop("scores", None)
            if r.get("fail") == "grounding" and r.get("target_segmented") is False:
                r["fail"] = "segmentation"                       # runs before 2026-10-15 22:30 labelled "target never segmented" as grounding
        os.makedirs(f"{OUT}/{name}", exist_ok=True); json.dump(R, open(f"{OUT}/{name}/results.json", "w"), indent=0)
        print(f"{name:24s} {sum(bool(r.get('success')) for r in R):4d}/{len(R)}")
        vids = sorted(glob.glob(f"{S}/runs/{name}/videos/*.mp4"))[:6]
        if vids:
            os.makedirs(f"{ROOT}/demo/franka/{name}", exist_ok=True)
            for v in vids: shutil.copy(v, f"{ROOT}/demo/franka/{name}/")
    if os.path.isdir(f"{S}/ground_eval"):
        os.makedirs(f"{OUT}/ground_eval", exist_ok=True)
        for f in glob.glob(f"{S}/ground_eval/*.json"):
            R = json.load(open(f))
            if isinstance(R, dict) and all(isinstance(v, dict) and "raw" in v for v in R.values()):
                for v in R.values(): v["raw"] = v["raw"][:120]
            json.dump(R, open(f"{OUT}/ground_eval/{os.path.basename(f)}", "w"), indent=0)


if __name__ == "__main__":
    main()
