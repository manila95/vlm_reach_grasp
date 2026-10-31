"""Our own appearance perturbations for LIBERO scenes: the dev set of docs/PLAN_perceptual_robustness.md.

Render-only changes (no physics change), seeded per episode, written independently of LIBERO-Plus (none of its generators
or assets are used), so selecting on these scenes does not tune on the LIBERO-Plus test set.
  light:   per light, diffuse x U(0.15, 2.5) with a random colour tint, ambient U(0, 0.35), specular U(0, 0.5),
           direction tilted up to 60 deg from vertical, shadows on with p = 0.5; headlight x U(0.3, 1.5)
  texture: table / floor / wall materials: plain random colour (texture removed), or their texture swapped among the
           scene's surface textures, each tinted by a random colour; texture repeat x U(0.5, 3)
  both:    light + texture
"""
import numpy as np

SURF_MATS = ("floorplane", "table_texture", "table_mat", "walls_mat", "living_room")
SURF_TEX = ("texplane", "tex-table", "tex-wall", "textable", "tex-living_room")


def _tint(rng, lo=0.7, hi=1.3):
    c = rng.uniform(lo, hi, 3); return c / c.max()


FIELDS = ("light_diffuse", "light_ambient", "light_specular", "light_dir", "light_castshadow", "mat_rgba", "mat_texid", "mat_texrepeat")


def _model(env):
    m = env.sim.model; return m._model if hasattr(m, "_model") else m


def restore_scene(env):
    """Back to the unperturbed appearance (the first model seen; draws never compound)."""
    mm = _model(env)
    if not hasattr(env, "_vrg_base"):
        env._vrg_base = {k: getattr(mm, k).copy() for k in FIELDS}
        env._vrg_base["hl"] = (mm.vis.headlight.ambient.copy(), mm.vis.headlight.diffuse.copy())
    B = env._vrg_base
    for k in FIELDS: getattr(mm, k)[:] = B[k]
    mm.vis.headlight.ambient[:], mm.vis.headlight.diffuse[:] = B["hl"]
    return mm, B


def perturb_scene(env, kind, seed):
    import mujoco
    if kind in (None, "", "none"): return
    rng = np.random.default_rng(seed); mm, B = restore_scene(env)
    if kind in ("light", "both"):
        for i in range(mm.nlight):
            mm.light_diffuse[i] = B["light_diffuse"][i] * rng.uniform(0.15, 2.5) * _tint(rng)
            mm.light_ambient[i] = rng.uniform(0, 0.35) * _tint(rng)
            mm.light_specular[i] = rng.uniform(0, 0.5)
            th, ph = np.deg2rad(rng.uniform(0, 60)), rng.uniform(0, 2 * np.pi)
            mm.light_dir[i] = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), -np.cos(th)]
            mm.light_castshadow[i] = rng.random() < 0.5
        s = rng.uniform(0.3, 1.5); mm.vis.headlight.ambient[:] = B["hl"][0] * s; mm.vis.headlight.diffuse[:] = B["hl"][1] * s
    if kind in ("texture", "both"):
        tex = [mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_TEXTURE, n) for n in SURF_TEX]; tex = [t for t in tex if t >= 0]
        for n in SURF_MATS:
            j = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_MATERIAL, n)
            if j < 0: continue
            if rng.random() < 0.4:
                mm.mat_texid[j] = -1; mm.mat_rgba[j, :3] = rng.uniform(0.05, 0.95, 3)
            else:
                if tex: mm.mat_texid[j] = tex[rng.integers(len(tex))]
                mm.mat_rgba[j, :3] = rng.uniform(0.35, 1.0) * _tint(rng, 0.5, 1.5)
                mm.mat_texrepeat[j] = B["mat_texrepeat"][j] * rng.uniform(0.5, 3)
