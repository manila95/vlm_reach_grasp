"""CLIP text embeddings for Colosseum instructions (vlmrg env) -> text_emb.pkl (numpy) + text_emb_py38.pkl (plain lists)."""
import glob, json, os, pickle, torch
from transformers import CLIPModel, CLIPTokenizer
ROOT = "/scratch/kaustubh/vlm_reach_grasp/colosseum"; C = "/scratch/kaustubh/vlm_reach_grasp/cache/hf"
S = {m["desc"] for f in glob.glob(f"{ROOT}/train_npz/*_meta.json") for m in json.load(open(f))}
for f in glob.glob(f"{ROOT}/scenes/*/*.pkl"): S.add(pickle.load(open(f, "rb"))["desc"])
m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=C).eval(); tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir=C)
S = sorted(S)
with torch.no_grad():
    o = m.text_model(**tok(S, padding=True, truncation=True, return_tensors="pt")); x = torch.nn.functional.normalize(m.text_projection(o.pooler_output), dim=-1).numpy()
E = dict(zip(S, x)); pickle.dump(E, open(f"{ROOT}/text_emb.pkl", "wb"))
pickle.dump({k: [float(v) for v in e] for k, e in E.items()}, open(f"{ROOT}/text_emb_py38.pkl", "wb"), protocol=4)
print(len(S), "instructions:", S[:8])
