"""Evaluate the Spatial-Forcing OpenVLA-OFT 7B checkpoint fine-tuned on LIBERO-Object under our protocol.

Reuses the repo's own initialize_model / run_episode unchanged; only (a) restricts to our 20 eval init states per task
and (b) optionally applies the same seeded layout shuffle right after set_init_state (before the 10 settle steps).
Run from ~/projects/Spatial-Forcing/openvla-SF with the `spatialforcing` env:
  PYTHONPATH=.:~/projects/vlm_reach_grasp python ~/projects/vlm_reach_grasp/vrg/vla_eval.py [--shuffle] --out DIR
"""
import argparse, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--shuffle", action="store_true"); ap.add_argument("--out", required=True)
ap.add_argument("--n_eps", type=int, default=20); ap.add_argument("--hires", type=int, default=0)
ap.add_argument("--eps", default=""); A = ap.parse_args()

sys.argv = sys.argv[:1]
from experiments.robot.libero import run_libero_eval as R
from vrg.pipeline import shuffle_layout
from vrg.ground_episodes import EVAL_EPS

cfg = R.GenerateConfig(pretrained_checkpoint="ckpts/spatial-forcing-7b-finetuned-libero-object", task_suite_name="libero_object",
                       center_crop=True, num_trials_per_task=A.n_eps)
model, action_head, proprio_projector, noisy_action_projector, processor = R.initialize_model(cfg)
resize_size = R.get_image_resize_size(cfg)
suite = R.benchmark.get_benchmark_dict()[cfg.task_suite_name]()


class ShuffleEnv:
    """Delegating wrapper: set_init_state() then (optionally) permute object positions with our per-episode seed."""
    def __init__(self, env, tid, language): self._env, self.tid, self.lang, self.ep = env, tid, language, None
    def __getattr__(self, k): return getattr(self._env, k)
    def set_init_state(self, s):
        obs = self._env.set_init_state(s)
        if A.shuffle:
            shuffle_layout(self._env, 7919 * self.tid + self.ep, self.lang)
            obs = self._env.env._get_observations(force_update=True)
        e = self._env.env  # remember object positions for the "which object moved" diagnostic
        self.pos0 = {n: e.sim.data.body_xpos[e.obj_body_id[n]].copy() for n in e.objects_dict if n != "basket_1"}
        return obs

    def step(self, a):
        out = self._env.step(a)
        if A.hires:
            H = A.hires; e = self._env
            f = e.sim.render(width=H, height=H, camera_name="agentview")[::-1].copy()
            f[-H // 3 - 6:-6, -H // 3 - 6:-6] = e.sim.render(width=H // 3, height=H // 3, camera_name="robot0_eye_in_hand")[::-1]
            self.frames.append(f)
        return out

    def moved(self):
        e = self._env.env
        d = {n: float(np.linalg.norm(e.sim.data.body_xpos[e.obj_body_id[n]] - p)) for n, p in self.pos0.items()}
        m = max(d, key=d.get); return m if d[m] > 0.05 else None


os.makedirs(A.out, exist_ok=True); res = []
for tid in range(suite.n_tasks):
    task = suite.get_task(tid); inits = suite.get_task_init_states(tid)
    env, desc = R.get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res); W = ShuffleEnv(env, tid, task.language)
    for ep in ([int(e) for e in A.eps.split(",")] if A.eps else EVAL_EPS[:A.n_eps]):
        W.ep = ep; W.frames = []; t0 = time.time()
        ok, _ = R.run_episode(cfg, W, desc, model, resize_size, processor, action_head, proprio_projector, noisy_action_projector, inits[ep])
        from vrg.common import target_name
        res.append(dict(tid=tid, ep=ep, success=bool(ok), fail=None, steps=0, sec=time.time() - t0, moved=W.moved(), target=target_name(task.language) + "_1"))
        if A.hires:
            import imageio; os.makedirs(f"{A.out}/hires", exist_ok=True)
            imageio.mimsave(f"{A.out}/hires/t{tid}_e{ep}.mp4", W.frames[10:], fps=30)  # drop the 10 settle steps (others start after settle)
            json.dump(dict(res[-1], phases=["language-conditioned VLA"] * len(W.frames[10:])), open(f"{A.out}/hires/t{tid}_e{ep}.json", "w"))
        print(f"[vla] task {tid} ep {ep}: success={ok} ({time.time() - t0:.1f}s)", flush=True)
    env.close()
json.dump(res, open(f"{A.out}/results.json", "w"), indent=1)
print(f"== vla success {np.mean([r['success'] for r in res]):.3f} ({len(res)} eps)")
