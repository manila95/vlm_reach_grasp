"""SayCan-style planning with a VLM: score(skill) = p_LM(skill | instruction, history) x affordance(skill, scene).

Skills (= the language-conditioned policy library in runs/saycan_50):
    0..9  "pick up the <object>"        (object vocabulary = the 10 LIBERO-Object targets)
    10    "place it in the basket"
    11    "done"
Affordance for a pick skill = sigmoid of the max VLM yes/no score over depth proposals ("is this the <object>?"), i.e.
"is that object in the scene"; for "place" = 1 if something is held (checked online), "done" = 0.5.
To save compute, visual affordance is evaluated for the top-3 language candidates (the rest get 0).
Cached per (task, episode, layout) -> data/saycan_plans[_shuf].pkl
"""
import argparse, pickle, numpy as np, torch
from vrg.common import *
from vrg.pipeline import reset_episode, render_rgbd
from vrg.vlm import Grounder, propose, zoom
from vrg.ground_episodes import CALIB_EPS, EVAL_EPS

OBJS = [target_name(get_suite().get_task(i).language).replace("_", " ") for i in range(10)]
SKILLS = [f"pick up the {o}" for o in OBJS] + ["place it in the basket", "done"]
PROMPT = ("Robot: Hi there, I'm a robot operating on a floor with groceries and a basket. You can ask me to do tasks.\n"
          "Human: {instr}\nRobot: I would: {hist}")


@torch.no_grad()
def lm_scores(g, instr, hist):
    """log p(skill text | prompt) under the VLM's language model (text-only)."""
    tok = g.proc.tokenizer; out = []
    h = "".join(f"{i + 1}. {s}, " for i, s in enumerate(hist)) + f"{len(hist) + 1}. "
    pre = PROMPT.format(instr=instr, hist=h)
    pid = tok(pre, return_tensors="pt").input_ids
    for s in SKILLS:
        sid = tok(s + ".", add_special_tokens=False, return_tensors="pt").input_ids
        ids = torch.cat([pid, sid], 1).to(g.model.device)
        lp = torch.log_softmax(g.model(input_ids=ids).logits[0, :-1].float(), -1)
        tgt = ids[0, 1:]; ll = lp[torch.arange(len(tgt)), tgt][-sid.shape[1]:]
        out.append(ll.sum().item())
    return np.array(out)


def softmax(x): e = np.exp(x - x.max()); return e / e.sum()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shuffle", action="store_true"); A = ap.parse_args()
    g = Grounder(); res = {}
    for tid in range(10):
        env, t = make_env(tid)
        lm1 = softmax(lm_scores(g, t.language, []))
        lm2 = {k: softmax(lm_scores(g, t.language, [SKILLS[k]])) for k in np.argsort(-lm1)[:3] if k < 10}
        for ep in EVAL_EPS:
            obs = reset_episode(env, tid, ep, A.shuffle)
            img, depth, K, T = render_rgbd(env); props = propose(depth, K, T)
            aff = np.zeros(len(SKILLS)); aff[11] = 0.5
            for k in np.argsort(-lm1)[:3]:
                if k >= 10: continue
                q = ('This is a close-up of one item in a robot workspace. Is this item the %s? '
                     'Read any label on the packaging. Answer Yes or No.' % OBJS[k])
                aff[k] = 1 / (1 + np.exp(-max(g.yes_score(zoom(img, p["box"]), q) for p in props)))
            comb = lm1 * aff; first = int(np.argmax(comb))
            res[(tid, ep)] = dict(lm1=lm1, aff=aff, combined=comb, first=first, lm2=lm2.get(first), correct=first == tid)
            print(tid, ep, SKILLS[first], "| correct", first == tid, flush=True)
        env.close()
    pickle.dump(dict(skills=SKILLS, plans=res), open(f"data/saycan_plans{'_shuf' if A.shuffle else ''}.pkl", "wb"))
    print("first-skill accuracy", np.mean([r["correct"] for r in res.values()]))
