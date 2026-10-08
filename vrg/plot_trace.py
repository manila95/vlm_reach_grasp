"""Plot one traced episode (logs/trace_t7_e1.json): top-down path and height profile, coloured by phase."""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "logs/trace_t7_e1.json"
out = sys.argv[2] if len(sys.argv) > 2 else "demo/ood_compare/trace_t7_e1.png"
D = json.load(open(src)); L = D["log"]
ee = 100 * np.array([l["ee"] for l in L]); milk = 100 * np.array([l["milk"] for l in L])
PH = [("[1]", "1  scripted reach to VLM handoff", "#2a78d6"), ("[2]", "2  learned egocentric grasp", "#eb6834"),
      ("[3]", "3  scripted transport + release", "#1baf7a")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
for key, lab, col in PH:
    idx = [i for i, l in enumerate(L) if l["phase"].startswith(key)]
    idx = [idx[0] - 1] + idx
    a1.plot(ee[idx, 1], ee[idx, 0], color=col, lw=2.2, label=lab); a2.plot(idx, ee[idx, 2], color=col, lw=2.2, label=lab)
gt, vlm, ho, bk = (100 * np.array(D[k]) for k in ("gt_obj", "vlm_obj", "handoff", "gt_bk"))
pts = [(gt, "milk (true)", dict(s=110, marker="s", color="#c3c2b7", edgecolor=INK), (-10, -3)),
       (vlm, "VLM point", dict(s=70, marker="x", color=INK), (-11, 2.5)),
       (ho, "handoff", dict(s=60, marker="o", facecolor="white", edgecolor="#2a78d6", linewidths=2), (1.5, 1.5)),
       (bk, "basket centre", dict(s=90, marker="P", color=INK2), (-6, 2.5)),
       (ee[10], "start", dict(s=50, color=INK), (1.5, -3))]
for p, lab, kw, off in pts:
    a1.scatter(p[1], p[0], zorder=4, **kw); a1.text(p[1] + off[0], p[0] + off[1], lab, fontsize=9, color=INK)
a1.set_xlabel("y (cm)   ← robot's right · left →"); a1.set_ylabel("x (cm)   ← toward robot · away →"); a1.set_aspect("equal")
a1.set_xlim(-22, 34); a1.set_title("Top-down end-effector path", loc="left", fontsize=11, color=INK)
a2.plot(range(len(L)), milk[:, 2], color=INK2, lw=1.2, ls="--", label="milk height")
close = next(i for i, l in enumerate(L) if l["phase"].startswith("[2]") and all(L[j]["a"][6] > 0 for j in range(i, min(i + 5, len(L)))))
opn = next(i for i, l in enumerate(L) if l["phase"].startswith("[3]") and l["a"][6] < 0)
for i, t in ((close, "gripper closes"), (opn, "gripper opens")):
    a2.axvline(i, color="#c3c2b7", lw=1); a2.text(i + 1, 32, t, fontsize=8.5, color=INK2)
a2.set_xlabel("env step (20 Hz, steps 0–9 = settle)"); a2.set_ylabel("height z (cm)"); a2.set_ylim(0, 35)
a2.legend(loc="lower right", fontsize=8.5, frameon=False)
a2.set_title("End-effector and object height", loc="left", fontsize=11, color=INK)
fig.suptitle('"pick up the milk", shuffled layout, episode 1 — ours succeeds; end-to-end, SayCan, HAMSTER and VLA all grab the butter',
             fontsize=10, color=INK2, x=0.01, ha="left")
plt.tight_layout(); plt.savefig(out, dpi=130, facecolor="white"); print("saved", out, "close", close, "open", opn)
