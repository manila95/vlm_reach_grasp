"""E8 grounding study on cached LIBERO-Plus dev scenes (vrg/lplus/cache_scenes.py): models x strategies, scored offline.

Backends
  qwen:<hf id>       local Qwen3-VL (text answers; yes/no logits for the "verify" strategy)
  gemini:<model>     Gemini API (key from $GEMINI_API_KEY; images sent as PNG)
Strategies (each returns the chosen proposal for the pick target and for the receptacle)
  verify     current system: parse the instruction (parser v2), yes/no score per proposal crop (Qwen only)
  marks      numbered boxes drawn on the image + the full instruction: "which number is the object to pick / the container?"
  recursive  marks, then a second round on zoomed crops of the 3 most likely candidates with the full instruction,
             then a third round with the crops reordered; majority of the rounds (re-asks when rounds disagree)
A choice is correct when it is the proposal nearest the ground-truth target (basket). Output: one JSON line per scene in
<out>/<backend>__<strategy>.jsonl (resumable), with the answer, correctness, number of calls and seconds.
  $LPY -m vrg.lplus.ground_study --backend gemini:gemini-3.8-flash --strategy marks --cache $S/liberoplus/scene_cache --out $S/liberoplus/ground_study
"""
import argparse, base64, glob, io, json, os, re, time, urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def png_b64(img):
    b = io.BytesIO(); Image.fromarray(img).save(b, format="PNG"); return base64.b64encode(b.getvalue()).decode()


class Gemini:
    def __init__(self, model):
        self.model = model; self.key = os.environ["GEMINI_API_KEY"]

    def ask(self, imgs, text, retries=6):
        parts = [{"inline_data": {"mime_type": "image/png", "data": png_b64(i)}} for i in imgs] + [{"text": text}]
        body = json.dumps({"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.0}}).encode()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        for a in range(retries):
            try:
                req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
                d = json.load(urllib.request.urlopen(req, timeout=180))
                return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"] if not p.get("thought"))
            except Exception as e:
                if a == retries - 1: raise
                time.sleep(min(60, 5 * 2 ** a))


class Qwen:
    def __init__(self, model):
        from vrg.vlm import Grounder
        self.g = Grounder(model, device="cuda:0")

    def ask(self, imgs, text):
        if len(imgs) == 1: return self.g.ask(imgs[0], text)
        w = sum(i.shape[1] for i in imgs) + 10 * (len(imgs) - 1); h = max(i.shape[0] for i in imgs)
        canvas = np.full((h, w, 3), 255, np.uint8); x = 0
        for i in imgs: canvas[:i.shape[0], x:x + i.shape[1]] = i; x += i.shape[1] + 10
        return self.g.ask(canvas, text)


def draw_marks(img, boxes, order):
    im = Image.fromarray(img.copy()); d = ImageDraw.Draw(im); f = ImageFont.truetype(FONT, 30)
    for n, i in enumerate(order, 1):
        x0, y0, x1, y1 = [int(v) for v in boxes[i]]
        d.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=4)
        d.rectangle([x0, max(y0 - 36, 0), x0 + 40, max(y0 - 36, 0) + 34], fill=(255, 40, 40)); d.text((x0 + 6, max(y0 - 36, 0)), str(n), fill=(255, 255, 255), font=f)
    return np.asarray(im)


def ints(s, k):
    m = re.search(r"\{.*\}", s, re.S)
    try:
        d = json.loads(m.group(0)); return [d.get(x) for x in k]
    except Exception:
        return [None] * len(k)


def marks_round(be, sc, order, ranked=False):
    img = draw_marks(sc["img"], sc["boxes"], order)
    if ranked:
        q = (f'A robot received the instruction: "{sc["language"]}". Numbered red boxes mark the objects in the scene. '
             'Answer in JSON: {"pick": [three box numbers, most likely first, for the object the robot must pick up], '
             '"place": the box number of the container it must be put into}.')
    else:
        q = (f'A robot received the instruction: "{sc["language"]}". Numbered red boxes mark the objects in the scene. '
             'Which box is the object the robot must pick up, and which box is the container it must be put into? '
             'Answer in JSON: {"pick": number, "place": number}.')
    r = be.ask([img], q); p, pl = ints(r, ["pick", "place"])
    to_idx = lambda n: order[int(n) - 1] if isinstance(n, (int, float, str)) and str(n).strip().isdigit() and 1 <= int(n) <= len(order) else None
    if ranked:
        p = [to_idx(n) for n in (p if isinstance(p, list) else [p])]; p = [x for x in p if x is not None]
    else:
        p = to_idx(p)
    return p, to_idx(pl), r


def crops_round(be, sc, cands, perm):
    from vrg.vlm import zoom
    tiles = []; letters = "ABCD"
    for n, i in enumerate(perm):
        c = np.asarray(Image.fromarray(zoom(sc["img"], sc["boxes"][cands[i]])).resize((300, 300)))
        im = Image.fromarray(c.copy()); ImageDraw.Draw(im).text((8, 4), letters[n], fill=(255, 40, 40), font=ImageFont.truetype(FONT, 40)); tiles.append(np.asarray(im))
    q = (f'A robot received the instruction: "{sc["language"]}". The images are close-ups of candidate objects, labelled '
         f'{", ".join(letters[:len(perm)])}. Which one is the object the robot must pick up? Read labels and packaging. '
         'Answer in JSON: {"pick": letter}.')
    r = be.ask(tiles, q); (a,) = ints(r, ["pick"])
    if isinstance(a, str) and a.strip()[:1].upper() in letters[:len(perm)]: return cands[perm[letters.index(a.strip()[:1].upper())]], r
    return None, r


def run_scene(be, strategy, sc, rng):
    n = len(sc["boxes"]); calls = 0
    if strategy == "verify":
        from vrg.vlm import zoom
        pick, place, _ = be.g.parse_target(sc["language"], version=2); calls += 2
        q = 'This is a close-up of one item in a robot workspace. Is this item the %s? Read any label on the packaging. Answer Yes or No.'
        sp = [be.g.yes_score(zoom(sc["img"], b), q % pick) for b in sc["boxes"]]
        sb = [be.g.yes_score(zoom(sc["img"], b), q % place) for b in sc["boxes"]]; calls += 2 * n
        return int(np.argmax(sp)), int(np.argmax(sb)), calls, {"pick": pick, "place": place}
    order = list(rng.permutation(n))
    if strategy == "marks":
        p, pl, r = marks_round(be, sc, order); return p, pl, 1, {"raw": r[-300:]}
    if strategy == "recursive":
        ranked, pl, r1 = marks_round(be, sc, order, ranked=True); calls = 1
        if not ranked: return None, pl, calls, {"raw": r1[-300:]}
        cands = list(dict.fromkeys(ranked))[:3]
        if len(cands) == 1: return cands[0], pl, calls, {"rounds": [int(cands[0])], "raw": r1[-300:]}
        votes = [cands[0]]                                   # round 1: the top-ranked mark
        for perm in (list(range(len(cands))), list(rng.permutation(len(cands)))):
            c, _ = crops_round(be, sc, cands, perm); calls += 1   # rounds 2-3: zoomed crops, full instruction
            if c is not None: votes.append(c)
            if len(votes) >= 2 and votes.count(votes[-1]) >= 2: break   # two rounds agree: stop
        best = max(set(votes), key=votes.count)
        return best, pl, calls, {"rounds": [int(v) for v in votes], "raw": r1[-300:]}
    raise ValueError(strategy)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backend", required=True); ap.add_argument("--strategy", required=True)
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    A = ap.parse_args()
    kind, model = A.backend.split(":", 1); be = Gemini(model) if kind == "gemini" else Qwen(model)
    os.makedirs(A.out, exist_ok=True); f = f"{A.out}/{A.backend.replace('/', '_').replace(':', '-')}__{A.strategy}.jsonl"
    done = {json.loads(l)["tid"] for l in open(f)} if os.path.exists(f) else set()
    files = sorted(glob.glob(f"{A.cache}/scene_*.npz"), key=lambda p: int(p.split("_")[-1][:-4]))
    if A.limit: files = files[:A.limit]
    for p in files:
        tid = int(p.split("_")[-1][:-4])
        if tid in done: continue
        z = np.load(p, allow_pickle=True); sc = {k: z[k] for k in z.files}; sc["language"] = str(sc["language"]); gt = json.loads(str(sc["gt"]))
        if len(sc["boxes"]) == 0: continue
        true_p = int(np.argmin([np.linalg.norm(np.array(xy) - np.array(gt[str(sc["target"])][:2])) for xy in sc["xy"]]))
        true_b = int(np.argmin([np.linalg.norm(np.array(xy) - np.array(gt["basket_1"][:2])) for xy in sc["xy"]]))
        t0 = time.time()
        try: ch_p, ch_b, calls, info = run_scene(be, A.strategy, sc, np.random.default_rng(tid))
        except Exception as e: print(tid, "ERROR", repr(e)[:200], flush=True); continue
        rec = dict(tid=tid, pick_ok=bool(ch_p == true_p), place_ok=bool(ch_b == true_b), calls=calls, sec=time.time() - t0,
                   chosen=[None if ch_p is None else int(ch_p), None if ch_b is None else int(ch_b)], info=info)
        open(f, "a").write(json.dumps(rec, default=str) + "\n")
        print(tid, "pick", rec["pick_ok"], "place", rec["place_ok"], f"{rec['sec']:.1f}s", flush=True)
    R = [json.loads(l) for l in open(f)]
    print(f"STUDY {A.backend} {A.strategy}: pick {np.mean([r['pick_ok'] for r in R]):.3f} place {np.mean([r['place_ok'] for r in R]):.3f} "
          f"n={len(R)} calls/scene {np.mean([r['calls'] for r in R]):.1f} sec/scene {np.mean([r['sec'] for r in R]):.1f}", flush=True)


if __name__ == "__main__":
    main()
