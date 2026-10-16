"""Client for vrg.franka.vlm_server (numpy only, so it runs in the Isaac env and on the robot PC)."""
import time
from multiprocessing.connection import Client
import numpy as np


class RemoteVLM:
    def __init__(self, port, host="localhost", wait=900):
        t0 = time.time()
        while True:
            try: self.conn = Client((host, port), authkey=b"vrg"); break
            except (ConnectionRefusedError, OSError):
                if time.time() - t0 > wait: raise
                time.sleep(5)

    def ground(self, mode, obj, frame, props):
        """Same contract as vrg.franka.vlm.ground: props need id and pix[frame.name]; only those visible in frame are sent."""
        P = [(p["id"], p["pix"][frame.name][0], np.packbits(p["pix"][frame.name][1])) for p in props if frame.name in p["pix"]]
        self.conn.send(dict(op="ground", mode=mode, obj=obj, cam=frame.name, rgb=frame.rgb, props=P)); return self.conn.recv()

    def verify(self, obj, frame, box):
        self.conn.send(dict(op="verify", obj=obj, rgb=frame.rgb, box=box)); return self.conn.recv()
