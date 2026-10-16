"""VLM server (vlmrg env): the robot process (Isaac, or the real arm's PC) asks a GPU process for grounding over a socket.

  python -m vrg.franka.vlm_server --model qwen3vl-2b --port 6102 [--device cuda:0]
Requests (multiprocessing.connection, authkey b"vrg"), answered in order by one model, several clients allowed:
  dict(op="ground", mode, obj, cam, rgb, props=[(id, box, packed mask)])  -> vrg.franka.vlm.ground record (+ sec)
  dict(op="verify", obj, rgb, box)                                        -> dict(score=Yes-vs-No logit, sec)
"""
import argparse, threading, time
from multiprocessing.connection import Listener
from types import SimpleNamespace
import numpy as np
from vrg.franka.vlm import VLM, ground, crop, verify_question


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--device", default="cuda:0"); A = ap.parse_args()
    V = VLM(A.model, device=A.device); lock = threading.Lock()

    def answer(req):
        t0 = time.time()
        if req["op"] == "ground":
            H, W = req["rgb"].shape[:2]
            props = [dict(id=i, pix={req["cam"]: (box, np.unpackbits(m)[:H * W].reshape(H, W).astype(bool))}) for i, box, m in req["props"]]
            r = ground(V, req["mode"], req["obj"], SimpleNamespace(name=req["cam"], rgb=req["rgb"]), props)
        elif req["op"] == "verify":
            r = dict(score=V.yes_score(crop(req["rgb"], req["box"]), verify_question(req["obj"])))
        else:
            raise ValueError(req["op"])
        r["sec"] = time.time() - t0; return r

    def handle(conn):
        with conn:
            while True:
                try: req = conn.recv()
                except (EOFError, ConnectionResetError): return
                if req == "close": return
                with lock: r = answer(req)
                conn.send(r)

    with Listener(("localhost", A.port), authkey=b"vrg") as L:
        print(f"serving {A.model} on {A.port}", flush=True)
        while True: threading.Thread(target=handle, args=(L.accept(),), daemon=True).start()


if __name__ == "__main__":
    main()
