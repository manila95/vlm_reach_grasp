"""Grounding gallery for the notebook (vlmrg env): what the external camera sees, the target, and what each VLM picked.

Reads saved eval scenes and the offline grounding results (each model's dev-chosen mode); writes demo/franka/grounding/*.png.
  python -m vrg.franka.grounding_figs
"""
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from vrg.franka.ground_offline import load_scene

S = "/scratch/kaustubh/vlm_reach_grasp/isaac"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/demo/franka/grounding"
MODELS = ["qwen3vl-2b", "qwen3vl-4b", "qwen3vl-8b", "qwen3vl-32b"]
COL = {"qwen3vl-2b": (235, 104, 52), "qwen3vl-4b": (237, 161, 0), "qwen3vl-8b": (168, 50, 50), "qwen3vl-32b": (42, 120, 214)}
SCENES = ["101|009_gelatin_box", "100|061_foam_brick", "134|010_potted_meat_can", "127|003_cracker_box"]


def panel(key, best, G, scale=2):
    seed, target = key.split("|")
    Sc, F, props, lab = load_scene(f"{S}/scenes_eval/{seed}_{target}.pkl")
    img = F["ext"].rgb.copy(); H, W = img.shape[:2]
    box = np.array([p["pix"]["ext"][0] for p in props])                  # crop to the objects, with margin
    x0, y0 = max(int(box[:, 0].min()) - 40, 0), max(int(box[:, 1].min()) - 40, 0)
    x1, y1 = min(int(box[:, 2].max()) + 40, W), min(int(box[:, 3].max()) + 40, H)
    im = Image.fromarray(img).convert("RGB"); dr = ImageDraw.Draw(im)
    for p in props:                                                        # proposal outlines: target green, others white
        m = p["pix"]["ext"][1]; edge = m & ~ndimage.binary_erosion(m, iterations=2)
        a = np.array(im); a[edge] = (40, 200, 90) if lab.get(p["id"]) == target else (255, 255, 255); im = Image.fromarray(a)
    im = im.crop((x0, y0, x1, y1)).resize(((x1 - x0) * scale, (y1 - y0) * scale), Image.BICUBIC); dr = ImageDraw.Draw(im)
    try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except OSError: font = ImageFont.load_default()
    picks, lines = {}, []
    for m in MODELS:
        r = G[m].get(key)
        if r is None: continue
        ok = r["correct"]; lines.append((f"{m.split('-')[1].upper()} ({best[m]}): {'✓' if ok else '✗ ' + (str(r['chosen_obj'])[4:] if r['chosen_obj'] else 'nothing')}", COL[m]))
        if r["choice"] is not None: picks.setdefault(r["choice"], []).append(m)
    for pid, ms in picks.items():                                          # one dot per model on its chosen proposal
        p = next(q for q in props if q["id"] == pid); v, u = np.nonzero(p["pix"]["ext"][1])
        cx, cy = (np.median(u) - x0) * scale, (np.median(v) - y0) * scale
        for k, m in enumerate(ms):
            x = cx + (k - (len(ms) - 1) / 2) * 16; dr.ellipse([x - 7, cy - 7, x + 7, cy + 7], fill=COL[m], outline=(0, 0, 0))
    bar = 24 + 20 * len(lines); out = Image.new("RGB", (im.width, im.height + bar), (250, 250, 248)); out.paste(im, (0, bar)); d = ImageDraw.Draw(out)
    d.text((6, 4), f'"pick up the {Sc["desc"]}"  (seed {seed})', fill=(20, 20, 20), font=font)
    for k, (t, c) in enumerate(lines): d.text((6, 24 + 20 * k), t, fill=c, font=font)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    best = json.load(open(f"{S}/ground_eval/best_mode_dev.json"))
    G = {m: json.load(open(f"{S}/ground_eval/{m}_{best[m]}.json")) for m in MODELS}
    for key in SCENES:
        panel(key, best, G).save(f"{OUT}/{key.replace('|', '_')}.png"); print("wrote", key)


if __name__ == "__main__":
    main()
