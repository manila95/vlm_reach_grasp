"""Tiny policy server (vlmrg env) so the Python-3.8 Colosseum sim can query GPU policies.

python -m vrg.bench.policy_server --port 6001 --ckpt runs/.../ckpt.pt [--device cuda:0]
Request: dict(obs={cam: (H,W,3) uint8, "proprio": (P,), ["text": (512,)]}) -> reply: (H, A) float32 unnormalised actions.
"""
import argparse, os
from multiprocessing.connection import Listener
import numpy as np, torch
from vrg.policy import DiffusionPolicy


def load(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False); st = torch.load(os.path.join(os.path.dirname(ckpt), "stats.pt"))
    from vrg.bench.col_train import H
    pol = DiffusionPolicy(st, n_obs=1, horizon=H, n_act=H, adim=ck["adim"], pdim=ck["pdim"], cams=ck["cams"], text_dim=512 if ck["text"] else 0).to(device)
    pol.load_state_dict(ck["ema"]); pol.eval(); return pol, ck


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0"); A = ap.parse_args()
    pol, ck = load(A.ckpt, A.device); print("serving", A.ckpt, "on", A.port, flush=True)
    with Listener(("localhost", A.port), authkey=b"vrg") as L:
        while True:
            conn = L.accept()
            try:
                while True:
                    req = conn.recv()
                    if req == "close": break
                    o = req["obs"]; b = {k: torch.from_numpy(np.asarray(v))[None, None].to(A.device) for k, v in o.items() if k != "text"}
                    if "text" in o: b["text"] = torch.from_numpy(np.asarray(o["text"], np.float32))[None].to(A.device)
                    with torch.no_grad(): a = pol.act(b)[0].float().cpu().numpy()
                    conn.send(a)
            except EOFError: pass
            finally: conn.close()
