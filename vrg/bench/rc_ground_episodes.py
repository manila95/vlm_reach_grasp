"""Stage 2 (vlmrg env): VLM parsing + grounding for every cached RoboCasa scene -> grounding.pkl, plus SayCan plans.

Key (split, task, ep) -> dict(pick_name, place_name, place_is_object, pick, place, pick_score, saycan={...}, gt_err...)
"""
import glob, json, os, pickle, sys
import numpy as np, torch
from vrg.vlm import Grounder, zoom
from vrg.bench.rc_ground import parse, ground, ground_place, nearest, P_VER

ROOT = "/scratch/kaustubh/vlm_reach_grasp/robocasa"
SAYCAN_PROMPT = ("Robot: Hi there, I'm a robot operating in a kitchen. You can ask me to do tasks.\n"
                 "Human: {instr}\nRobot: I would: 1. ")


@torch.no_grad()
def lm_logprob(g, prompt, cont):
    tok = g.proc.tokenizer
    p = tok(prompt, return_tensors="pt").input_ids; c = tok(cont, add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([p, c], 1).to(g.model.device)
    lp = torch.log_softmax(g.model(input_ids=ids).logits[0, :-1].float(), -1); tgt = ids[0, 1:]
    return lp[torch.arange(len(tgt)), tgt][-c.shape[1]:].sum().item()


@torch.no_grad()
def lm_logprob_batch(g, prompt, conts):
    """Sum log p(continuation | prompt) for many continuations in one padded forward pass."""
    tok = g.proc.tokenizer; p = tok(prompt, return_tensors="pt").input_ids[0]
    cs = [tok(c, add_special_tokens=False, return_tensors="pt").input_ids[0] for c in conts]
    L = len(p) + max(len(c) for c in cs); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(cs), L), pad); att = torch.zeros((len(cs), L), dtype=torch.long)
    for i, c in enumerate(cs): x = torch.cat([p, c]); ids[i, :len(x)] = x; att[i, :len(x)] = 1
    ids, att = ids.to(g.model.device), att.to(g.model.device)
    lp = torch.log_softmax(g.model(input_ids=ids, attention_mask=att).logits.float(), -1)
    out = []
    for i, c in enumerate(cs):
        pos = torch.arange(len(p) - 1, len(p) - 1 + len(c), device=ids.device)
        out.append(lp[i, pos, ids[i, pos + 1]].sum().item())
    return np.array(out)


def saycan_plan(g, scene, instr, vocab):
    """LM usefulness over the fixed pick-skill vocabulary x VLM affordance (is that object visible) for the top-3."""
    lm = lm_logprob_batch(g, SAYCAN_PROMPT.format(instr=instr), [f"pick up the {v}." for v in vocab])
    p = np.exp(lm - lm.max()); p /= p.sum(); aff = np.zeros(len(vocab))
    for k in np.argsort(-p)[:3]:
        best, _ = ground(g, scene, vocab[k], generic=False)
        aff[k] = 1 / (1 + np.exp(-best["score"])) if best else 0.0
    j = int(np.argmax(p * aff)) if aff.max() > 0 else int(np.argmax(p))
    return dict(pick_text=f"pick up the {vocab[j]}", lm=p.tolist(), aff=aff.tolist(), choice=vocab[j])


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/grounding.pkl"
    res = pickle.load(open(out, "rb")) if os.path.exists(out) else {}
    vocab = sorted({m["target"].replace("_", " ") for f in glob.glob(f"{ROOT}/train/*_meta.json") for m in json.load(open(f))})
    print("saycan vocab", len(vocab), flush=True)
    g = Grounder()
    scenes = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/scenes"   # e.g. grounding_dev.pkl + scenes_dev
    for f in sorted(glob.glob(f"{scenes}/*/*.pkl")):
        s = pickle.load(open(f, "rb")); key = (s["split"], s["task"], s["ep"])
        if key in res: continue
        lang = s["info"]["lang"]; pick, place, isobj = parse(g, lang)
        bp, _ = ground(g, s, pick); bq, _ = ground_place(g, s, place, isobj, pick, s["base_pos"])
        r = dict(pick_name=pick, place_name=place, place_is_object=isobj, lang=lang,
                 pick=None if bp is None else bp["point"], pick_score=None if bp is None else bp["score"], pick_cam=bp and bp["cam"],
                 place=None if bq is None else bq["point"], gt_obj=s["gt"]["obj"])
        r["pick_err"] = None if bp is None else float(np.linalg.norm(bp["point"] - s["gt"]["obj"]))
        r["pick_hit"] = bp is not None and (nearest(s["gt"], bp["point"])[0] == "obj" or r["pick_err"] < 0.05)
        from vrg.bench.rc_train import place_clause
        r["saycan"] = dict(saycan_plan(g, s, lang, vocab), place_text=place_clause(lang))
        res[key] = r
        print(key, pick, "|", place, "| hit", r["pick_hit"], "| saycan:", r["saycan"]["choice"], flush=True)
        pickle.dump(res, open(out, "wb"))
    for sp in ("pretrain", "target"):
        v = [r["pick_hit"] for k, r in res.items() if k[0] == sp]; print(sp, "pick grounding", sum(v), "/", len(v))
