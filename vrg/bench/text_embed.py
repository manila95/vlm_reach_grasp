"""CLIP text embeddings for every instruction / skill string (vlmrg env) -> robocasa/text_emb.pkl (merged, never overwritten)."""
import glob, json, os, pickle, torch
from transformers import CLIPModel, CLIPTokenizer
ROOT = "/scratch/kaustubh/vlm_reach_grasp/robocasa"
os.environ.setdefault("HF_HOME", "/scratch/kaustubh/vlm_reach_grasp/cache/hf")
from vrg.bench.rc_train import skill_texts
S = set()
for f in glob.glob(f"{ROOT}/train/*_meta.json"):
    for m in json.load(open(f)): S.add(m["lang"]); S.update(skill_texts(m))
for gp in (f"{ROOT}/grounding.pkl", f"{ROOT}/grounding_dev.pkl"):
  if os.path.exists(gp):
    for r in pickle.load(open(gp, "rb")).values(): S.add(r["lang"]); S.add(r["saycan"]["pick_text"]); S.add(r["saycan"]["place_text"])
for f in glob.glob(f"{ROOT}/scenes/*/*.pkl") + glob.glob(f"{ROOT}/scenes_dev/*/*.pkl"): S.add(pickle.load(open(f, "rb"))["info"]["lang"])
out = f"{ROOT}/text_emb.pkl"; E = pickle.load(open(out, "rb")) if os.path.exists(out) else {}
todo = sorted(s for s in S if s not in E)
m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/scratch/kaustubh/vlm_reach_grasp/cache/hf").eval(); tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/scratch/kaustubh/vlm_reach_grasp/cache/hf")
with torch.no_grad():
    for i in range(0, len(todo), 256):
        b = todo[i:i + 256]; o = m.text_model(**tok(b, padding=True, truncation=True, return_tensors="pt")); x = m.text_projection(o.pooler_output)
        x = torch.nn.functional.normalize(x, dim=-1).numpy()
        for s_, v in zip(b, x): E[s_] = v
pickle.dump(E, open(out, "wb")); print("strings", len(E), "new", len(todo))
