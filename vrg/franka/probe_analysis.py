"""How informative is each VLM signal along the approach? (dev probe runs, vrg.franka.reactive.probe)

Per model and stage (ext, approach0..K-1, look): how often the true target is in view, how often the VLM's pick among
the visible proposals is the target (when it is in view), and how well the Yes/No score separates target from
non-target (AUC, and accuracy of "highest Yes among visible"). Then candidate fusion rules, scored on the same data.
  python -m vrg.franka.probe_analysis
"""
import glob, json, collections
import numpy as np

R = "/scratch/kaustubh/vlm_reach_grasp/isaac/runs"


def auc(pos, neg):
    if not pos or not neg: return np.nan
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def load(s):
    out = []
    for r in json.load(open(f"{R}/dev_probe_{s}/results.json")):
        lab = {int(k): v for k, v in r["prop_labels"].items()}; tgt = r["target_name"]
        tid = [k for k, v in lab.items() if v == tgt]
        out.append(dict(r=r, lab=lab, tgt=tgt, tid=tid[0] if tid else None, views=r.get("views", [])))
    return out


def stage_table(E):
    st = collections.OrderedDict()
    for e in E:
        for v in e["views"]:
            d = st.setdefault(v["stage"], dict(n=0, vis=0, pick=0, yes_max=0, pos=[], neg=[]))
            d["n"] += 1
            if e["tid"] is None or e["tid"] not in v["visible"]: continue
            d["vis"] += 1; d["pick"] += v["choice"] == e["tid"]
            ys = {int(k): x for k, x in v["yes"].items()}
            if ys: d["yes_max"] += max(ys, key=ys.get) == e["tid"]
            for k, x in ys.items(): (d["pos"] if k == e["tid"] else d["neg"]).append(x)
    return st


def rules(E, w_close=1.0):
    """Fused choice per episode under candidate rules (only uses information the robot has)."""
    res = collections.defaultdict(list)
    for e in E:
        V = e["views"]; ok = lambda c: c is not None and c == e["tid"]
        if not V: continue
        res["ext only"].append(ok(V[0]["choice"]))
        look = V[-1] if V[-1]["stage"] == "look" else None
        # majority of picks, ties to the latest
        votes = collections.Counter(); last = None
        for v in V:
            if v["choice"] is not None: votes[v["choice"]] += 1; last = v["choice"]
        if not votes:                                                   # no answer at all: every rule fails
            for k in ["vote (ties to latest)", f"weighted vote (1+{w_close:g}i)", "latest view decides", "summed relative Yes"]: res[k].append(False)
            continue
        best = max(votes.values()); tied = [k for k, x in votes.items() if x == best]
        res["vote (ties to latest)"].append(ok(last if last in tied else tied[0]))
        # later views weigh more: weight 1 + w_close * index
        wv = collections.Counter()
        for i, v in enumerate(V):
            if v["choice"] is not None: wv[v["choice"]] += 1 + w_close * i
        res[f"weighted vote (1+{w_close:g}i)"].append(ok(max(wv, key=wv.get)))
        # last view with >= 2 candidates in view decides
        lv = [v for v in V if v["choice"] is not None and len(v["visible"]) >= 2]
        res["latest view decides"].append(ok(lv[-1]["choice"] if lv else V[0]["choice"]))
        # summed Yes/No scores over views (relative: each view's scores minus that view's mean)
        S = collections.defaultdict(float)
        for v in V:
            ys = {int(k): x for k, x in v["yes"].items()}
            if len(ys) >= 2:
                m = np.mean(list(ys.values()))
                for k, x in ys.items(): S[k] += x - m
        res["summed relative Yes"].append(ok(max(S, key=S.get) if S else V[0]["choice"]))
    return {k: (float(np.mean(v)), len(v)) for k, v in res.items()}


def main():
    for s in ["2b", "4b", "8b", "32b"]:
        try: E = load(s)
        except FileNotFoundError: continue
        print(f"\n=== {s}  ({len(E)} episodes, target segmented in {sum(e['tid'] is not None for e in E)})")
        for k, d in stage_table(E).items():
            print(f"  {k:10s} n={d['n']:3d} target in view {d['vis']:3d}  pick acc|vis {d['pick']/max(d['vis'],1):.2f}  "
                  f"max-Yes acc|vis {d['yes_max']/max(d['vis'],1):.2f}  Yes AUC {auc(d['pos'], d['neg']):.2f}")
        for k, (a, n) in rules(E).items(): print(f"  rule {k:28s} {a:.3f} (n={n})")


if __name__ == "__main__":
    main()
