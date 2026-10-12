"""Offline check of instruction parsing + placement grounding on cached RoboCasa scenes (vlmrg env)."""
import glob, pickle, numpy as np
from vrg.vlm import Grounder
from vrg.bench.rc_ground import parse, ground_place, ground, nearest
RECEPT = {"PickPlaceCounterToStove": "container", "PickPlaceSinkToCounter": "container", "PickPlaceToasterToCounter": "plate"}
g = Grounder()
for f in sorted(glob.glob("/scratch/kaustubh/vlm_reach_grasp/robocasa/scenes/*/*.pkl")):
    s = pickle.load(open(f, "rb")); t = s["task"]
    pick, place, isobj = parse(g, s["info"]["lang"])
    best, _ = ground_place(g, s, place, isobj, pick, s["base_pos"])
    if best is None: print(s["split"], t, s["ep"], "NO PLACE", pick, place); continue
    p = best["point"]
    if t in RECEPT: info = f"err to {RECEPT[t]} = {100*np.linalg.norm(p - s['gt'][RECEPT[t]]):.1f}cm"
    else:
        fx = s["info"]["fixtures"]["cab" if "Cabinet" in t else "counter"]
        info = f"dist to {fx[0]} centre = {100*np.linalg.norm(p - fx[1]):.1f}cm, z={p[2]:.2f} (fixture z {fx[1][2]:.2f})"
    print(f"{s['split']:8s} {t[9:]:18s} ep{s['ep']} parse=({pick} | {place} | obj={isobj}) {info} reach={np.linalg.norm((p - s['base_pos'])[:2]):.2f}m", flush=True)
