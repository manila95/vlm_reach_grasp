"""When does getting closer stop helping? Design of the "obscured view" gate on dev probes (dev_probe2_*, K=6).

A view is described only by what the robot knows: the approached object's mask area (fraction of the image) and
whether its box is cut by the image border, how many candidates are in view, and the camera height. For each gate
("this view is obscured") the decision is the pick of the LAST view before the first obscured one (the external view
if the first wrist view is already obscured), or a vote over the unobscured views. Scored on dev episodes.
  python -m vrg.franka.probe_gate
"""
import collections, json, sys
import numpy as np

R = "/scratch/kaustubh/vlm_reach_grasp/isaac/runs"


def episodes(s):
    E = []
    for r in json.load(open(f"{R}/dev_probe2_{s}/results.json")):
        lab = {int(k): v for k, v in r["prop_labels"].items()}; tid = next((k for k, v in lab.items() if v == r["target_name"]), None)
        V = r.get("views", []); app = r.get("initial_choice")
        for v in V:
            q = v.get("quality", {}).get(str(app), {}); v["f_area"] = q.get("area", np.nan); v["f_trunc"] = q.get("trunc", None)
            v["f_nvis"] = len(v["visible"]); v["f_seen"] = app in v["visible"]
        E.append(dict(tid=tid, V=V, app=app))
    return E


GATES = {"never (latest view)": lambda v: False, "approached object cut by the border": lambda v: bool(v["f_trunc"]),
         "fewer than 3 candidates in view": lambda v: v["f_nvis"] < 3, "fewer than 2 candidates in view": lambda v: v["f_nvis"] < 2}
for a in (0.05, 0.08, 0.12, 0.2): GATES[f"approached object > {int(a*100)}% of the image"] = (lambda a: lambda v: v["f_area"] > a)(a)
for h in (0.30, 0.35, 0.40): GATES[f"camera below {int(h*100)} cm"] = (lambda h: lambda v: v["height"] is not None and v["height"] < h)(h)
GATES["cut by border OR < 3 in view"] = lambda v: bool(v["f_trunc"]) or v["f_nvis"] < 3
GATES["external view only"] = lambda v: v["stage"] != "ext"


def decide(V, gate, rule):
    ok = [V[0]]
    for v in V[1:]:
        if not v["f_seen"]: continue                                 # approached object not in this view yet: no evidence, keep going
        if gate(v): break                                            # first obscured view: stop, keep the last unobscured one
        ok.append(v)
    picks = [v["choice"] for v in ok if v["choice"] is not None and (v["stage"] == "ext" or len(v["visible"]) >= 2)]
    if not picks: return None
    if rule == "last": return picks[-1]
    c = collections.Counter(picks); b = max(c.values()); tied = [k for k, x in c.items() if x == b]
    return [p for p in picks if p in tied][-1]


def main(models=("2b", "4b", "8b", "32b")):
    res = collections.defaultdict(dict)
    for s in models:
        try: E = episodes(s)
        except FileNotFoundError: continue
        # per-stage view statistics
        print(f"\n=== {s} ({len(E)} eps)")
        st = collections.OrderedDict()
        for e in E:
            for v in e["V"]:
                d = st.setdefault(v["stage"], collections.defaultdict(list))
                if e["tid"] is not None and e["tid"] in v["visible"]: d["acc"].append(v["choice"] == e["tid"])
                d["area"].append(v["f_area"]); d["trunc"].append(bool(v["f_trunc"])); d["nvis"].append(v["f_nvis"]); d["h"].append(v["height"] or np.nan)
        for k, d in st.items():
            print(f"  {k:10s} acc|vis {np.mean(d['acc']) if d['acc'] else np.nan:.2f} (n={len(d['acc']):2d})  area {np.nanmedian(d['area']):.3f}  "
                  f"cut {np.mean(d['trunc']):.2f}  in view {np.mean(d['nvis']):.1f}  height {np.nanmedian([x for x in d['h'] if x == x] or [np.nan]):.2f} m")
        for g, fn in GATES.items():
            for rule in ("last", "vote"):
                res[(g, rule)][s] = np.mean([decide(e["V"], fn, rule) == e["tid"] for e in E])
    print("\n" + f"{'gate':40s} {'rule':5s} " + " ".join(f"{m:>5s}" for m in models) + "   mean")
    for (g, rule), d in res.items():
        print(f"{g:40s} {rule:5s} " + " ".join(f"{d.get(m, np.nan):5.3f}" for m in models) + f"  {np.nanmean(list(d.values())):.3f}")


if __name__ == "__main__":
    main(tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("2b", "4b", "8b", "32b"))
