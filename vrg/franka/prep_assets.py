"""One-time asset prep: copy YCB props from the Isaac asset server to /scratch and make them physics-enabled.

Adds RigidBodyAPI + MassAPI on the root and convex-decomposition colliders on every mesh, flattens to one local file
(no network needed afterwards), and records each object's size (axis-aligned box, metres) in ycb_meta.json.

  DISPLAY=:147 OMNI_KIT_ACCEPT_EULA=YES $ISAACPY -m vrg.franka.prep_assets
"""
import json, os
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf

OUT = "/scratch/kaustubh/vlm_reach_grasp/isaac/assets/ycb"
# approximate real masses (kg), YCB paper / object sheets
MASS = {"002_master_chef_can": 0.414, "003_cracker_box": 0.411, "004_sugar_box": 0.514, "005_tomato_soup_can": 0.349,
        "006_mustard_bottle": 0.603, "007_tuna_fish_can": 0.171, "008_pudding_box": 0.187, "009_gelatin_box": 0.097,
        "010_potted_meat_can": 0.370, "011_banana": 0.066, "019_pitcher_base": 0.178, "021_bleach_cleanser": 1.131,
        "024_bowl": 0.147, "025_mug": 0.118, "035_power_drill": 0.895, "036_wood_block": 0.729, "037_scissors": 0.082,
        "040_large_marker": 0.016, "051_large_clamp": 0.125, "052_extra_large_clamp": 0.202, "061_foam_brick": 0.028}


def main():
    os.makedirs(OUT, exist_ok=True); meta = {}
    for name, mass in MASS.items():
        src = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/{name}.usd"
        stage = Usd.Stage.Open(src)
        if stage is None: print("missing", name, flush=True); continue
        flat = stage.Flatten(); st = Usd.Stage.Open(flat)
        root = st.GetDefaultPrim()
        UsdPhysics.RigidBodyAPI.Apply(root); PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
        n_mesh = 0
        for p in Usd.PrimRange(root):
            if p.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(p); mc = UsdPhysics.MeshCollisionAPI.Apply(p)
                mc.CreateApproximationAttr("convexDecomposition"); n_mesh += 1
                PhysxSchema.PhysxCollisionAPI.Apply(p).CreateContactOffsetAttr(0.002)
        bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(root).ComputeAlignedRange()
        lo, hi = list(bb.GetMin()), list(bb.GetMax())
        dst = f"{OUT}/{name}.usd"; st.GetRootLayer().Export(dst)
        meta[name] = dict(usd=dst, mass=mass, aabb_min=lo, aabb_max=hi, size=[h - l for l, h in zip(lo, hi)], n_mesh=n_mesh,
                          scale=float(UsdGeom.LinearUnitsAttr if False else UsdGeom.GetStageMetersPerUnit(stage)))
        print(name, "size(m)", [round(s, 3) for s in meta[name]["size"]], "metersPerUnit", meta[name]["scale"], flush=True)
    json.dump(meta, open(f"{OUT}/ycb_meta.json", "w"), indent=1)


main()
app.close()
