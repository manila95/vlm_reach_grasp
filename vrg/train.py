"""Train a diffusion policy on LIBERO-Object.

--mode local : windows from the VLM handoff point (ts) until the object is lifted (te); no task conditioning,
               a single object-agnostic grasp skill shared by all 10 tasks.
--mode full  : whole demonstrations, conditioned on task id (standard multi-task imitation baseline).
"""
import argparse, json, os, time, copy
import numpy as np, torch
from vrg.policy import DiffusionPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["local", "full", "saycan", "hamster"], required=True)
ap.add_argument("--demos_per_task", type=int, default=50)
ap.add_argument("--steps", type=int, default=60000)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--cams", default="agent,wrist")
ap.add_argument("--pre", type=int, default=None, help="local: start the slice this many steps before the grasp (default: dataset ts = 20)")
ap.add_argument("--post", type=int, default=6, help="local: extra steps past lift included")
ap.add_argument("--mask_xy", action="store_true", help="drop absolute EEF x,y from proprio (egocentric policy)")
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:0")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True); json.dump(vars(A), open(f"{A.out}/args.json", "w"), indent=1)
dev = torch.device(A.device); torch.manual_seed(0); np.random.seed(0)
cams = tuple(A.cams.split(","))

Z = np.load("data/libero_object.npz"); meta = json.load(open("data/libero_object_meta.json"))
H, NO = 16, 2
idx, starts, ends, conds = [], [], [], []  # sample index t, demo start, demo end (exclusive), conditioning id
for m in meta:
    if sum(1 for x in meta[:meta.index(m)] if x["tid"] == m["tid"]) >= A.demos_per_task: continue
    s, T = m["start"], m["T"]
    if A.mode == "local": segs = [(m["ts"] if A.pre is None else max(m["tg"] - A.pre, 0), min(m["te"] + A.post, T - 1), 0)]
    elif A.mode == "saycan":  # SayCan-style skill library: "pick up the <obj>" (id = object/task) and "place it in the basket" (id 10)
        segs = [(0, min(m["te"] + A.post, T - 1), m["tid"]), (m["te"], T - 1, 10)]
    else: segs = [(0, T - 1, m["tid"])]
    for lo, hi, c in segs:
        for t in range(lo, hi + 1): idx.append(s + t); starts.append(s); ends.append(s + T); conds.append(c)
conds = torch.tensor(conds, device=dev)
idx, starts, ends = map(lambda x: torch.tensor(x, device=dev), (idx, starts, ends))
print(f"{A.mode}: {len(idx)} samples")

AG = np.load("data/agent_path.npy") if A.mode == "hamster" else Z["agent"]  # HAMSTER: path drawn on agentview
G = {"agent": torch.from_numpy(AG).to(dev), "wrist": torch.from_numpy(Z["wrist"]).to(dev),
     "proprio": torch.from_numpy(Z["proprio"]).to(dev), "action": torch.from_numpy(Z["action"]).to(dev),
     "tid": torch.from_numpy(Z["tid"]).long().to(dev)}
if A.mask_xy: G["proprio"][:, :2] = 0
sel = torch.unique(torch.cat([torch.minimum(torch.maximum(idx[:, None] + torch.arange(-1, H - 1, device=dev), starts[:, None]), ends[:, None] - 1).flatten()]))
stats = {"action_min": G["action"][sel].min(0).values.cpu(), "action_max": G["action"][sel].max(0).values.cpu(),
         "proprio_min": G["proprio"][sel].min(0).values.cpu(), "proprio_max": G["proprio"][sel].max(0).values.cpu()}
torch.save(stats, f"{A.out}/stats.pt")

pol = DiffusionPolicy(stats, n_obs=NO, horizon=H, n_tasks={"full": 10, "saycan": 11}.get(A.mode, 0), cams=cams).to(dev)
ema = copy.deepcopy(pol).eval(); [p.requires_grad_(False) for p in ema.parameters()]
opt = torch.optim.AdamW(pol.parameters(), lr=A.lr, weight_decay=1e-6)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 500) * 0.5 * (1 + np.cos(np.pi * s / A.steps)))
print("params(M)", sum(p.numel() for p in pol.parameters()) / 1e6)


def batch(bs):
    j = torch.randint(0, len(idx), (bs,), device=dev)
    t, s, e = idx[j], starts[j], ends[j]
    to = (t[:, None] + torch.arange(-(NO - 1), 1, device=dev)).clamp(min=s[:, None])  # obs frames
    ta = (t[:, None] + torch.arange(-(NO - 1), H - NO + 1, device=dev)).clamp(min=s[:, None], max=e[:, None] - 1)
    obs = {c: G[c][to] for c in cams}; obs["proprio"] = G["proprio"][to]; obs["tid"] = conds[j]
    return obs, G["action"][ta]


t0 = time.time(); log = open(f"{A.out}/log.txt", "a")
for step in range(1, A.steps + 1):
    obs, act = batch(A.bs)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = pol.loss(obs, act)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step(); sched.step()
    with torch.no_grad():
        d = min(0.9999, (1 + step) / (10 + step))
        for pe, p in zip(ema.parameters(), pol.parameters()): pe.lerp_(p, 1 - d)
    if step % 500 == 0:
        msg = f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {(time.time()-t0)/step*1000:.0f}ms/it"
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
    if step % 20000 == 0 or step == A.steps:
        torch.save({"ema": ema.state_dict(), "args": vars(A)}, f"{A.out}/ckpt.pt")
