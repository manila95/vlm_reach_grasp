"""Side-by-side OOD demo videos: the same shuffled scene, five methods, one grid per scene.

Inputs: runs/demo/<method>/hires/t{tid}_e{ep}.mp4 + .json (recorded by rollout.py --hires / vla_eval.py --hires).
Episodes are chosen by rule in data/demo_episodes.json, not by outcome:
  typical  = for each task, first shuffled episode whose target sits where no demo ever grasped (83% of cases)
  contrast = episodes whose target sits where some demo grasped (where HAMSTER-style can succeed)
"""
import json, os, pickle
import numpy as np, imageio
from PIL import Image, ImageDraw, ImageFont
from vrg.common import get_suite, target_name

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = [("ours", "Ours: VLM handoff + local grasp"), ("e2e", "End-to-end diffusion policy"),
           ("saycan", "SayCan-style: VLM planner + skills"), ("hamster", "HAMSTER-style: VLM path → policy"),
           ("vla", "VLA: OpenVLA-OFT 7B")]
P, HOLD = 384, 60
F = lambda sz: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
F1, F2, F3 = F(17), F(14), F(22)
GREEN, RED, WHITE, GREY = (40, 200, 90), (235, 70, 60), (255, 255, 255), (200, 200, 200)
pretty = lambda n: n.replace("_1", "").replace("_", " ") if n else n


def verdict(m):
    if m["success"]: return "SUCCESS: target in basket", GREEN
    if m.get("moved") is None: return "FAILED: nothing picked up", RED
    if m["moved"] != m["target"]: return f"FAILED: grabbed {pretty(m['moved'])}", RED
    return "FAILED: right object, not placed", RED


def panel(frame, title, phase, final=None):
    im = Image.fromarray(frame); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, P, 50], fill=(0, 0, 0)); d.text((8, 5), title, fill=WHITE, font=F1)
    d.text((8, 28), phase[:46], fill=GREY, font=F2)
    if final:
        txt, col = final; d.rectangle([0, P - 44, P, P], fill=col); d.text((10, P - 36), txt, fill=WHITE, font=F1)
    return np.array(im)


def info_card(lang, moved_cm, seen):
    im = Image.new("RGB", (P, P), (22, 22, 22)); d = ImageDraw.Draw(im); y = 16
    for t, f, c in [("OUT-OF-DISTRIBUTION LAYOUT", F3, (255, 190, 60)), ("", F2, WHITE),
                    ("Instruction:", F2, GREY), (f'"{lang}"', F2, WHITE), ("", F2, WHITE),
                    ("Object positions were shuffled:", F2, GREY), (f"target moved {moved_cm:.0f} cm from its", F2, WHITE),
                    ("training location; it now sits where", F2, WHITE),
                    ("some demo grasped." if seen else "NO demo ever grasped.", F2, (255, 190, 60)), ("", F2, WHITE),
                    ("Zero-shot: no method was trained", F2, GREY), ("on shuffled scenes.", F2, GREY), ("", F2, WHITE),
                    ("Same scene and seed for all panels.", F2, GREY), ("Inset: wrist camera.", F2, GREY)]:
        if t:
            words, line = t.split(" "), ""
            for w in words:  # wrap
                if d.textlength(line + w, font=f) > P - 28: d.text((14, y), line, fill=c, font=f); y += 21; line = ""
                line += w + " "
            d.text((14, y), line, fill=c, font=f)
        y += 21 if f is not F3 else 30
    return np.array(im)


def compose(tid, ep, seen, out):
    lang = get_suite().get_task(tid).language
    G = pickle.load(open(f"{ROOT}/data/grounding.pkl", "rb")); GS = pickle.load(open(f"{ROOT}/data/grounding_shuf.pkl", "rb"))
    ref = next(r for r in G.values() if r["tid"] == tid and not r["calib"])
    moved_cm = 100 * np.linalg.norm(GS[(tid, ep)]["gt_obj"][:2] - ref["gt_obj"][:2])
    vids, metas = [], []
    for k, _ in METHODS:
        vids.append([f for f in imageio.get_reader(f"{ROOT}/runs/demo/{k}/hires/t{tid}_e{ep}.mp4")])
        metas.append(json.load(open(f"{ROOT}/runs/demo/{k}/hires/t{tid}_e{ep}.json")))
    T = max(len(v) for v in vids) + HOLD; card = info_card(lang, moved_cm, seen); frames = []
    for t in range(T):
        tiles = []
        for (k, title), v, m in zip(METHODS, vids, metas):
            i = min(t, len(v) - 1); ph = m["phases"][i] if i < len(m["phases"]) else ""
            tiles.append(panel(v[i], title, ph, verdict(m) if t >= len(v) - 1 else None))
        tiles.append(card)
        frames.append(np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0))
    imageio.mimsave(out, frames, fps=30, quality=7)
    return {k: verdict(m)[0] for (k, _), m in zip(METHODS, metas)}


if __name__ == "__main__":
    sel = json.load(open(f"{ROOT}/data/demo_episodes.json")); os.makedirs(f"{ROOT}/demo/ood_compare", exist_ok=True)
    summary = []
    for tid, ep in [(int(t), e) for t, e in sel["typical"].items()]:
        summary.append(dict(tid=tid, ep=ep, kind="never-grasped location", **compose(tid, ep, False, f"{ROOT}/demo/ood_compare/t{tid}_e{ep}.mp4")))
    for tid, ep in sel["contrast"]:
        summary.append(dict(tid=tid, ep=ep, kind="previously-grasped location", **compose(tid, ep, True, f"{ROOT}/demo/ood_compare/contrast_t{tid}_e{ep}.mp4")))
    json.dump(summary, open(f"{ROOT}/demo/ood_compare/summary.json", "w"), indent=1)
    for s in summary: print(s)



def reel(out=f"{ROOT}/demo/ood_compare/reel_all.mp4", scale=0.6):
    """All selected scenes back to back (typical first, then contrast), downscaled for embedding."""
    sel = json.load(open(f"{ROOT}/data/demo_episodes.json")); frames = []
    files = [f"t{t}_e{e}.mp4" for t, e in sel["typical"].items()] + [f"contrast_t{t}_e{e}.mp4" for t, e in sel["contrast"]]
    for fn in files:
        for f in imageio.get_reader(f"{ROOT}/demo/ood_compare/{fn}"):
            frames.append(np.array(Image.fromarray(f).resize((int(f.shape[1] * scale) // 16 * 16, int(f.shape[0] * scale) // 16 * 16))))
    imageio.mimsave(out, frames, fps=30, quality=5)
