"""Train policies on RoboCasa pick-and-place demos (robocasa env).

modes
  local_pick   egocentric grasp skill: wrist cam + (eef quat rel. base, gripper, eef z rel. base); slice [ts, te+post]; arm 7-D
  local_place  egocentric place skill: same inputs; slice [tr-PRE_PLACE, tr+POST_PLACE]; arm 7-D
  e2e          end-to-end: left/right/wrist cams + state16 + instruction (CLIP text); full demos; 12-D
  saycan       language-conditioned skills: "pick up the <obj>" (start -> lifted) and "<place clause>" (lifted -> end); 12-D
  hamster      end-to-end with the demo's own keypoint path drawn on the left camera; no text; 12-D
"""
import argparse, json, os, pickle, time, copy, re
import numpy as np, torch
from vrg.policy import DiffusionPolicy
from vrg.bench.robocasa_env import TASKS

ROOT = "/scratch/kaustubh/vlm_reach_grasp/robocasa"
PRE_PLACE, POST_PLACE = 20, 10


def ego_proprio(state):
    """state16: base pos(3) quat(4) | eef pos rel base(3) quat rel base(4) | gripper(2) -> egocentric 7-D."""
    return np.concatenate([state[:, 10:14], state[:, 14:16], state[:, 9:10]], 1).astype(np.float32)


def place_clause(lang):
    m = re.search(r"(place it .*?)\.?$", lang, re.I)
    return m.group(1) if m else lang


def skill_texts(meta_e):
    return f"pick up the {meta_e['target'].replace('_', ' ')}", place_clause(meta_e["lang"])


def hamster_images(Z, meta, res=128):
    """Draw each demo's keypoint path (start, pre-grasp, grasp, lifted, release) on its left-camera frames."""
    from vrg.paths import project, draw_path
    from robosuite.utils.transform_utils import quat2mat
    f = res / 2 / np.tan(np.radians(30)); K = np.array([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1]])
    st, left, T_left = Z["state"], Z["left"], Z["T_left"]  # each Z[k] re-reads the whole array from the npz: read once
    out = np.empty_like(left)
    bp = st[:, :3]; eef = bp + np.einsum("tij,tj->ti", np.stack([quat2mat(q) for q in st[:, 3:7]]), st[:, 7:10])
    for m in meta:
        s = m["start"]; kp = eef[[s, s + m["ts"], s + m["tg"], s + m["te"], s + m["tr"]]]
        for t in range(m["T"]):
            uv = project(kp, K, T_left[s + t])
            out[s + t] = draw_path(left[s + t][::-1], uv)[::-1]
    return out


def load(tasks, mode, post, n_demos, pre=0):
    D, metas, n0 = {k: [] for k in ["left", "right", "wrist", "proprio", "action", "text_key"]}, [], 0
    starts, ends, idx, conds = [], [], [], []
    for task in tasks:
        Z = np.load(f"{ROOT}/train/{task}_pretrain.npz"); meta = json.load(open(f"{ROOT}/train/{task}_pretrain_meta.json"))
        meta = [m for m in meta if m["success"]][:n_demos]
        keep = np.zeros(len(Z["state"]), bool)
        for m in meta: keep[m["start"]:m["start"] + m["T"]] = True
        remap = np.cumsum(keep) - 1
        need = ("wrist",) if mode.startswith("local") else ("left", "right", "wrist")
        for k in need:
            v = hamster_images(Z, meta) if (mode == "hamster" and k == "left") else Z[k]
            D[k].append(v[keep]); del v
        D["proprio"].append(ego_proprio(Z["state"])[keep] if mode.startswith("local") else Z["state"][keep])
        D["action"].append((Z["action"][:, :7] if mode.startswith("local") else Z["action"])[keep])
        for m in meta:
            s, T = n0 + remap[m["start"]], m["T"]; pick_t, place_t = skill_texts(m)
            if mode == "local_pick": segs = [(max(m["tg"] - pre, 0) if pre else m["ts"], min(m["te"] + post, T - 1), "")]
            elif mode == "local_place": segs = [(max(m["tr"] - PRE_PLACE, 0), min(m["tr"] + POST_PLACE, T - 1), "")]
            elif mode == "saycan": segs = [(0, min(m["te"] + post, T - 1), pick_t), (m["te"], T - 1, place_t)]
            elif mode == "e2e": segs = [(0, T - 1, m["lang"])]
            else: segs = [(0, T - 1, "")]
            for lo, hi, txt in segs:
                for t in range(lo, hi + 1): idx.append(s + t); starts.append(s); ends.append(s + T); conds.append(txt)
            metas.append(dict(m, task=task))
        n0 += int(keep.sum())
    D = {k: np.concatenate(v) for k, v in D.items() if v}
    return D, metas, np.array(idx), np.array(starts), np.array(ends), conds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["local_pick", "local_place", "e2e", "saycan", "hamster"])
    ap.add_argument("--tasks", default=",".join(TASKS)); ap.add_argument("--n_demos", type=int, default=100)
    ap.add_argument("--steps", type=int, default=60000); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--post", type=int, default=6)
    ap.add_argument("--pre", type=int, default=0, help="local_pick: start slices this many steps before the grasp (0: dataset ts = tg-20)")
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0")
    A = ap.parse_args(); os.makedirs(A.out, exist_ok=True); json.dump(vars(A), open(f"{A.out}/args.json", "w"), indent=1)
    dev = torch.device(A.device); torch.manual_seed(0)
    D, metas, idx, starts, ends, conds = load(A.tasks.split(","), A.mode, A.post, A.n_demos, A.pre)
    cams = ("wrist",) if A.mode.startswith("local") else ("left", "right", "wrist")
    use_text = A.mode in ("e2e", "saycan")
    print(A.mode, "samples", len(idx), "frames", len(D["action"]), "demos", len(metas), flush=True)
    G = {c: torch.from_numpy(D[c]).to(dev) for c in cams}
    G["proprio"] = torch.from_numpy(D["proprio"]).to(dev); G["action"] = torch.from_numpy(D["action"]).to(dev)
    del D; import gc; gc.collect()  # host copies are no longer needed once the data is on the GPU
    TXT = None
    if use_text:
        E = pickle.load(open(f"{ROOT}/text_emb.pkl", "rb"))
        keys = sorted(set(conds)); kid = {k: i for i, k in enumerate(keys)}
        TXT = torch.tensor(np.stack([E[k] for k in keys]), device=dev); cid = torch.tensor([kid[c] for c in conds], device=dev)
    idx, starts, ends = (torch.tensor(x, device=dev) for x in (idx, starts, ends))
    H, NO = 16, 2
    sel = torch.unique(torch.minimum(torch.maximum(idx[:, None] + torch.arange(-1, H - 1, device=dev), starts[:, None]), ends[:, None] - 1).flatten())
    stats = {"action_min": G["action"][sel].min(0).values.cpu(), "action_max": G["action"][sel].max(0).values.cpu(),
             "proprio_min": G["proprio"][sel].min(0).values.cpu(), "proprio_max": G["proprio"][sel].max(0).values.cpu()}
    torch.save(stats, f"{A.out}/stats.pt")
    adim, pdim = G["action"].shape[1], G["proprio"].shape[1]
    pol = DiffusionPolicy(stats, n_obs=NO, horizon=H, adim=adim, pdim=pdim, cams=cams, text_dim=512 if use_text else 0).to(dev)
    ema = copy.deepcopy(pol).eval(); [p.requires_grad_(False) for p in ema.parameters()]
    opt = torch.optim.AdamW(pol.parameters(), lr=A.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 500) * 0.5 * (1 + np.cos(np.pi * s / A.steps)))

    def batch(bs):
        j = torch.randint(0, len(idx), (bs,), device=dev); t, s, e = idx[j], starts[j], ends[j]
        to = (t[:, None] + torch.arange(-(NO - 1), 1, device=dev)).clamp(min=s[:, None])
        ta = (t[:, None] + torch.arange(-(NO - 1), H - NO + 1, device=dev)).clamp(min=s[:, None], max=e[:, None] - 1)
        obs = {c: G[c][to] for c in cams}; obs["proprio"] = G["proprio"][to]
        if TXT is not None: obs["text"] = TXT[cid[j]]
        return obs, G["action"][ta]

    t0 = time.time(); log = open(f"{A.out}/log.txt", "a")
    start = 1; SP = f"{A.out}/state.pt"
    if os.path.exists(SP):  # resume an interrupted run
        st = torch.load(SP, map_location=dev); pol.load_state_dict(st["pol"]); ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["step"] + 1
        torch.set_rng_state(st["rng"]); print("resumed at step", start, flush=True)
    t0 = time.time()
    for step in range(start, A.steps + 1):
        obs, act = batch(A.bs)
        with torch.autocast("cuda", dtype=torch.bfloat16): loss = pol.loss(obs, act)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step(); sched.step()
        with torch.no_grad():
            d = min(0.9999, (1 + step) / (10 + step))
            for pe, p in zip(ema.parameters(), pol.parameters()): pe.lerp_(p, 1 - d)
        if step % 500 == 0:
            msg = f"step {step} loss {loss.item():.4f} {(time.time() - t0) / (step - start + 1) * 1000:.0f}ms/it"; print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if step % 5000 == 0:
            torch.save({"pol": pol.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "rng": torch.get_rng_state()}, SP + ".tmp"); os.replace(SP + ".tmp", SP)
        if step % 20000 == 0 or step == A.steps:
            torch.save({"ema": ema.state_dict(), "args": vars(A), "adim": adim, "pdim": pdim, "cams": cams, "text": use_text}, f"{A.out}/ckpt.pt")
