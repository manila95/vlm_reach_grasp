"""VLM grounding for the Franka track, with swappable local models for the size sweep.

Three ways to turn "pick up the <object>" into one of the geometric proposals (perception.proposals):
  point    the VLM points at the object in the full external image; the proposal under / nearest the point wins
  marks    proposals are drawn as numbered marks on the image (set-of-marks); the VLM answers with a number
  verify   each proposal is cropped and scored with the VLM's Yes/No logit; highest score wins (as on LIBERO)
All three return the same record so the evaluation can attribute failures to grounding vs grasping.
"""
import re, time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

MODELS = {   # name -> HF id (all local, Apache-2.0); the sweep runs smallest -> largest
    "qwen3vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
}


class VLM:
    def __init__(self, name="qwen3vl-8b", device="cuda:0"):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.name, self.hf = name, MODELS.get(name, name)
        # one GPU per model (32B bf16 is ~66 GB): splitting Qwen3-VL-32B over two GPUs with device_map="auto" made every
        # generation NaN ("!!!!", token 0) on 2026-10-15 while the same weights on one GPU are fine
        self.model = AutoModelForImageTextToText.from_pretrained(self.hf, dtype=torch.bfloat16, device_map=device).eval()
        self.proc = AutoProcessor.from_pretrained(self.hf)
        tok = self.proc.tokenizer
        self.yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")]
        self.no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")]

    def _inputs(self, img, text):
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(img)}, {"type": "text", "text": text}]}]
        return self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                             return_tensors="pt").to(self.model.device)

    @torch.no_grad()
    def ask(self, img, text, max_new_tokens=64):
        inp = self._inputs(img, text)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    @torch.no_grad()
    def yes_score(self, img, text):
        lg = self.model(**self._inputs(img, text)).logits[0, -1].float()
        return (torch.logsumexp(lg[self.yes], 0) - torch.logsumexp(lg[self.no], 0)).item()


# ---------------------------------------------------------------- helpers
def crop(img, box, margin=0.5, out=448, min_side=60):
    x0, y0, x1, y1 = box; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; s = max(x1 - x0, y1 - y0, min_side) * (1 + margin)
    H, W = img.shape[:2]; a, b = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0)); c, d = int(min(cx + s / 2, W)), int(min(cy + s / 2, H))
    return np.array(Image.fromarray(img[b:d, a:c]).resize((out, max(int(out * (d - b) / max(c - a, 1)), 1)), Image.BICUBIC))


def draw_marks(img, props, cam):
    im = Image.fromarray(img).convert("RGB"); dr = ImageDraw.Draw(im)
    try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(14, img.shape[0] // 30))
    except OSError: font = ImageFont.load_default()
    marks = {}
    for k, p in enumerate(props, 1):
        if cam not in p["pix"]: continue
        _, mask = p["pix"][cam]; v, u = np.nonzero(mask); cx, cy = int(np.median(u)), int(np.median(v))
        tb = dr.textbbox((cx, cy), str(k), font=font, anchor="mm"); dr.rectangle([tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2], fill=(0, 0, 0))
        dr.text((cx, cy), str(k), fill=(255, 255, 255), font=font, anchor="mm"); marks[k] = p["id"]
    return np.array(im), marks


def parse_point(text, W, H, family="qwen3"):
    m = re.search(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", text)
    if m is None: return None
    x, y = float(m.group(1)), float(m.group(2))
    return (x / 1000 * W, y / 1000 * H) if family == "qwen3" else (x, y)      # Qwen3-VL uses 0..1000 coordinates


# ---------------------------------------------------------------- grounding
def ground(vlm, mode, obj, frame, props):
    """Pick the proposal that is `obj`. Returns dict(choice (proposal id or None), raw, sec, scores, uv)."""
    t0 = time.time(); img = frame.rgb; H, W = img.shape[:2]; cam = frame.name
    vis = [p for p in props if cam in p["pix"]]
    rec = dict(mode=mode, model=vlm.name, choice=None, raw="", scores=None, uv=None)
    if mode == "point":
        r = vlm.ask(img, f'Point to the {obj} in the image. Output its point coordinates in JSON format like '
                         f'[{{"point_2d": [x, y], "label": "{obj}"}}].')
        rec["raw"] = r; uv = parse_point(r, W, H)
        if uv is not None and vis:
            rec["uv"] = uv; u, v = int(np.clip(uv[0], 0, W - 1)), int(np.clip(uv[1], 0, H - 1))
            inside = [p for p in vis if p["pix"][cam][1][v, u]]
            if inside: rec["choice"] = inside[0]["id"]
            else:                                            # nearest proposal mask in pixels
                dist = [np.min(np.hypot(*(np.array(np.nonzero(p["pix"][cam][1]))[::-1] - np.array([[u], [v]])))) for p in vis]
                rec["choice"] = vis[int(np.argmin(dist))]["id"]; rec["snapped_px"] = float(min(dist))
    elif mode == "marks":
        mimg, marks = draw_marks(img, vis, cam)
        r = vlm.ask(mimg, f"The objects on the table are labelled with white numbers on black boxes. "
                          f"Which number is on the {obj}? Answer with the number only.", max_new_tokens=8)
        rec["raw"] = r; m = re.search(r"\d+", r)
        if m and int(m.group(0)) in marks: rec["choice"] = marks[int(m.group(0))]
    elif mode == "verify":
        q = f"This is a close-up of one item on a table. Is this item the {obj}? Answer Yes or No."
        sc = {p["id"]: vlm.yes_score(crop(img, p["pix"][cam][0]), q) for p in vis}
        rec["scores"] = sc
        if sc: rec["choice"] = max(sc, key=sc.get)
    else:
        raise ValueError(mode)
    rec["sec"] = time.time() - t0
    return rec
