"""Train whole-trajectory vs contact-segment diffusion policies on a MimicGen dataset (mimicgen env).

  --mode whole     every frame of every demo (P1 / C1 policy); cameras and absolute proprio as given
  --mode segment   only contact phases: for subtask k, steps [max(start_k, contact_k - pre), end_k] (P2 / C2 policy);
                   conditioned on the subtask index
  --proprio abs    eef pos + quat + gripper in the robot base frame (global)
            rel    eef position relative to the segment's start (handoff) position + quat + gripper (egocentric; P2)
Same network (vrg.policy.DiffusionPolicy, 84 px images, crop 76), steps and batch size for every mode, so the comparison
is at equal gradient steps. Resumable (state.pt).
  $MGPY -m vrg.mg.train --dataset square_d0 --mode segment --cams wrist --proprio rel --pre 10 --out $S/mimicgen/runs/sq_seg
"""
import argparse, copy, json, os, time
import numpy as np, torch

S = "/scratch/kaustubh/vlm_reach_grasp/mimicgen"
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--mode", choices=["whole", "segment"], required=True)
ap.add_argument("--cams", default="agent,wrist"); ap.add_argument("--proprio", choices=["abs", "rel"], default="abs")
ap.add_argument("--pre", type=int, default=10, help="segment mode: steps before first contact where the learned phase starts")
ap.add_argument("--demos", type=int, default=1000); ap.add_argument("--steps", type=int, default=60000); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True); json.dump(vars(A), open(f"{A.out}/args.json", "w"), indent=1)
torch.manual_seed(A.seed); np.random.seed(A.seed); dev = torch.device(A.device); cams = tuple(A.cams.split(","))
from vrg.policy import DiffusionPolicy
H, NO = 16, 2

# ---------------------------------------------------------------- data
C = f"{S}/cache/{A.dataset}"; I = json.load(open(f"{C}/index.json"))
rng = np.random.default_rng(A.seed); use = np.sort(rng.choice(len(I["demos"]), min(A.demos, len(I["demos"])), replace=False))
idx, starts, ends, conds, anchor = [], [], [], [], []      # sample t, clamp window [start, end), conditioning id, anchor row (handoff)
if A.mode == "segment":
    SEG = json.load(open(f"{S}/segments/{A.dataset}.json"))
    n_sub = len(SEG["spec"])
for i in use:
    o, T = I["offsets"][i], I["T"][i]
    if A.mode == "whole":
        win = [(0, T - 1, 0)]
    else:
        subs = SEG["demos"][I["demos"][i]]["subtasks"]
        # MimicGen core datasets hold only successful demos; replaying their stored states does not always reproduce the
        # success check (datasets were post-processed for robosuite 1.4), so keep every demo with all contacts found
        if any(s["contact"] is None for s in subs): continue
        win = [(max(s["start"], s["contact"] - A.pre), s["end"], k) for k, s in enumerate(subs) if s["contact"] is not None]
    for lo, hi, k in win:
        for t in range(lo, hi + 1): idx.append(o + t); starts.append(o + lo); ends.append(o + hi + 1); conds.append(k); anchor.append(o + lo)
print(f"{A.mode}: {len(use)} demos -> {len(idx)} samples", flush=True)
G = {c: torch.from_numpy(np.load(f"{C}/{c}.npy", mmap_mode="r")[:]).to(dev) for c in cams}
P = torch.from_numpy(np.load(f"{C}/proprio.npy")).to(dev); ACT = torch.from_numpy(np.load(f"{C}/action.npy")).to(dev)
idx, starts, ends, conds, anchor = (torch.tensor(x, device=dev) for x in (idx, starts, ends, conds, anchor))


def proprio(rows, anc):
    """rows (B, k) frame indices; anc (B,) handoff row of each sample's segment. rel: position relative to the handoff."""
    p = P[rows].clone()
    if A.proprio == "rel": p[..., :3] -= P[anc][:, None, :3]
    return p


sel = torch.unique(torch.minimum(torch.maximum(idx[:, None] + torch.arange(-1, H - 1, device=dev), starts[:, None]), ends[:, None] - 1).flatten())
pr = proprio(idx[:, None], anchor)[:, 0]                  # proprio stats over the samples themselves
stats = {"action_min": ACT[sel].min(0).values.cpu(), "action_max": ACT[sel].max(0).values.cpu(),
         "proprio_min": pr.min(0).values.cpu(), "proprio_max": pr.max(0).values.cpu()}
torch.save(stats, f"{A.out}/stats.pt")
n_tasks = n_sub if A.mode == "segment" else 0
pol = DiffusionPolicy(stats, n_obs=NO, horizon=H, pdim=9, n_tasks=n_tasks, cams=cams, crop=76).to(dev)
ema = copy.deepcopy(pol).eval(); [p.requires_grad_(False) for p in ema.parameters()]
opt = torch.optim.AdamW(pol.parameters(), lr=A.lr, weight_decay=1e-6)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 500) * 0.5 * (1 + np.cos(np.pi * s / A.steps)))
step0 = 0
if os.path.exists(f"{A.out}/state.pt"):
    st = torch.load(f"{A.out}/state.pt", map_location=dev, weights_only=False)
    pol.load_state_dict(st["pol"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); step0 = st["step"]
    print("resumed at", step0, flush=True)


def batch(bs):
    j = torch.randint(0, len(idx), (bs,), device=dev)
    t, s, e = idx[j], starts[j], ends[j]
    to = (t[:, None] + torch.arange(-(NO - 1), 1, device=dev)).clamp(min=s[:, None])
    ta = (t[:, None] + torch.arange(-(NO - 1), H - NO + 1, device=dev)).clamp(min=s[:, None], max=e[:, None] - 1)
    obs = {c: G[c][to] for c in cams}; obs["proprio"] = proprio(to, anchor[j]); obs["tid"] = conds[j]
    return obs, ACT[ta]


t0 = time.time(); log = open(f"{A.out}/log.txt", "a")
for step in range(step0 + 1, A.steps + 1):
    obs, act = batch(A.bs)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = pol.loss(obs, act)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step(); sched.step()
    with torch.no_grad():
        d = min(0.9999, (1 + step) / (10 + step))
        for pe, p in zip(ema.parameters(), pol.parameters()): pe.lerp_(p, 1 - d)
    if step % 500 == 0:
        msg = f"step {step} loss {loss.item():.4f} {(time.time() - t0) / (step - step0) * 1000:.0f}ms/it"; print(msg, flush=True); log.write(msg + "\n"); log.flush()
    if step % 5000 == 0 or step == A.steps:
        torch.save(dict(pol=pol.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step), f"{A.out}/state.pt")
torch.save({"ema": ema.state_dict(), "args": vars(A), "n_tasks": n_tasks}, f"{A.out}/ckpt.pt")
print("TRAIN DONE", flush=True)
