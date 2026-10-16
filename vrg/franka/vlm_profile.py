"""Measure each VLM's footprint and per-query latency on this machine (vlmrg env): params, load time, peak GPU memory,
and latency per grounding mode on N saved scenes (after one warm-up query). Writes <out>/vlm_profile.json.
  python -m vrg.franka.vlm_profile --scenes /scratch/.../isaac/scenes_eval --out /scratch/.../isaac/ground_eval
"""
import argparse, glob, json, os, time
import numpy as np, torch
from vrg.franka.ground_offline import load_scene
from vrg.franka.vlm import VLM, ground, MODELS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scenes", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=20)
    A = ap.parse_args(); scenes = [load_scene(p) for p in sorted(glob.glob(f"{A.scenes}/*.pkl"))[:A.n]]
    outf = f"{A.out}/vlm_profile.json"; P = json.load(open(outf)) if os.path.exists(outf) else {}
    for m in MODELS:
        if m in P: continue
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        V = VLM(m, device="cuda:0"); load = time.time() - t0; w = torch.cuda.memory_allocated() / 2**30
        rec = dict(params_b=sum(p.numel() for p in V.model.parameters()) / 1e9, load_s=load, weights_gb=w, gpu=torch.cuda.get_device_name(0))
        S, F, props, _ = scenes[0]; ground(V, "marks", S["desc"], F["ext"], props)   # warm-up
        for mode in ("point", "marks", "verify"):
            ts = []
            for S, F, props, _ in scenes:
                torch.cuda.synchronize(); t = time.time(); ground(V, mode, S["desc"], F["ext"], props); torch.cuda.synchronize(); ts.append(time.time() - t)
            rec[f"{mode}_s_median"] = float(np.median(ts)); rec[f"{mode}_s_p90"] = float(np.percentile(ts, 90))
        rec["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30; rec["n_props_mean"] = float(np.mean([len(p) for _, _, p, _ in scenes]))
        P[m] = rec; json.dump(P, open(outf, "w"), indent=1); print(m, {k: round(v, 2) if isinstance(v, float) else v for k, v in rec.items()}, flush=True)
        del V; torch.cuda.empty_cache()
    print("PROFILE DONE", flush=True)


if __name__ == "__main__":
    main()
