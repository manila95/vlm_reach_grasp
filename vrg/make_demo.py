"""Live end-to-end demo video: VLM parse + propose/verify grounding -> reach -> learned grasp -> transport.

python -m vrg.make_demo --tasks 0,6,8 --ep 1 --ckpt runs/local_50/ckpt.pt
"""
import argparse, os, pickle
import numpy as np, imageio
from PIL import Image, ImageDraw, ImageFont
from vrg.common import *
from vrg.pipeline import *
from vrg.vlm import Grounder, ground_propose_verify
from vrg.rollout import calib_bias, LIFT_DONE, GRASP_MAX, TRANSPORT_Z

R = 512
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except Exception: FONT = ImageFont.load_default()


def caption(img, lines, color=(255, 255, 255)):
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 30 * len(lines) + 8], fill=(0, 0, 0))
    for i, l in enumerate(lines): d.text((10, 6 + 30 * i), l, fill=color, font=FONT)
    return np.array(im)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", default="0,6,8"); ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--ckpt", default="runs/local_50/ckpt.pt"); ap.add_argument("--out", default="demo")
    ap.add_argument("--shuffle", action="store_true", help="OOD shuffled layout")
    ap.add_argument("--e2e", default=None, help="also record this end-to-end policy on the same scene (side by side)")
    ap.add_argument("--tag", default="")
    A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
    g = Grounder(); runner = PolicyRunner(A.ckpt)
    ob_bias, bk_bias = calib_bias(pickle.load(open("data/grounding.pkl", "rb")))
    e2e = PolicyRunner(A.e2e) if A.e2e else None
    all_frames = []
    for tid in map(int, A.tasks.split(",")):
        env, task = make_env(tid)  # policy obs stay at 128px (training resolution); video frames rendered separately
        frames = []; phase = ["", ""]
        obs = reset_episode(env, tid, A.ep, A.shuffle)

        def hires():
            return np.concatenate([env.sim.render(width=R, height=R, camera_name=c)[::-1] for c in ("agentview", "robot0_eye_in_hand")], 1)

        def rec(o):
            f = hires()
            frames.append(caption(f, [f'"{task.language}"', phase[0]], (255, 255, 255)))

        def step(a):
            o, r, d, i = env.step(a); rec(o); return o, d

        # --- VLM: decompose instruction, ground pick & place targets
        pick, place, _ = g.parse_target(task.language)
        img, depth, K, T = render_rgbd(env)
        bo, props = ground_propose_verify(g, img, depth, K, T, pick)
        bb, _ = ground_propose_verify(g, img, depth, K, T, place)
        im = Image.fromarray(img); d = ImageDraw.Draw(im)
        for p in props:
            d.rectangle(p["box"], outline=(160, 160, 160), width=2); d.text((p["box"][0], p["box"][3] + 2), f'{p["score"]:+.1f}', fill=(255, 255, 255), font=FONT)
        d.rectangle(bo["box"], outline=(255, 40, 40), width=6); d.rectangle(bb["box"], outline=(40, 140, 255), width=6)
        title = np.array(im.resize((2 * R, 2 * R)).crop((0, int(0.25 * 2 * R), 2 * R, int(0.25 * 2 * R) + R)))
        title = caption(title, [f'VLM parse: pick="{pick}"  place="{place}"', f'depth proposals + VLM yes/no logit for "{pick}"; red = pick, blue = place'])
        frames += [title] * 75
        p_obj = np.array([*bo["xy"], bo["top"] / 2]) - ob_bias; p_bk = np.array([*bb["xy"], bb["top"] / 2]) - bk_bias
        # --- reach
        phase[0] = "[1] VLM-guided reach (scripted OSC)"
        obs, done = move_to(env, obs, p_obj + HANDOFF_OFF, HANDOFF_ORI, max_steps=100, cb=rec)
        # --- learned grasp
        phase[0] = "[2] learned local grasp (egocentric diffusion policy, wrist cam)"
        runner.reset(); z_close = None
        for _ in range(GRASP_MAX):
            a = runner(obs); a[6] = 1.0 if a[6] > 0 else -1.0
            if obs["robot0_joint_pos"][3] > -1.0:  # v5: posture-aware roll/pitch lock near the elbow limit
                a[3:5] = np.clip(ori_error(obs, HANDOFF_ORI) / 0.5, -0.5, 0.5)[:2]
            obs, done = step(a); z = obs["robot0_eef_pos"][2]
            if a[6] > 0 and z_close is None: z_close = z
            if a[6] < 0: z_close = None
            if z_close is not None and z > z_close + LIFT_DONE: break
        # --- transport + release
        phase[0] = "[3] VLM-guided transport + release (scripted)"
        obs, d = transport_and_release(env, obs, p_bk, cb=rec, release_steps=25, retract="auto"); done = done or d
        phase[0] = "SUCCESS" if done else "FAILED"
        for _ in range(30): rec(obs)
        ok = done
        if e2e is not None:  # same scene, end-to-end baseline
            ef = []; obs = reset_episode(env, tid, A.ep, A.shuffle); e2e.reset(); d = False
            tag = "End-to-end diffusion policy (task-id conditioned)"
            for k in range(300):
                a = e2e(obs, tid); a[6] = 1.0 if a[6] > 0 else -1.0
                obs, r, d, _ = env.step(a)
                ef.append(caption(hires(), [tag, f"step {k}"]))
                if d: break
            for _ in range(30): ef.append(caption(ef[-1], [tag, "SUCCESS" if d else "FAILED (timeout)"]))
            ours = [f for f in frames[75:]]
            n = max(len(ours), len(ef)); pad = lambda L: L + [L[-1]] * (n - len(L))
            side = [np.concatenate([a, b], 0) for a, b in zip(pad(ours), pad(ef))]
            imageio.mimsave(f"{A.out}/compare_task{tid}{A.tag}.mp4", side, fps=30)
            print(tid, "e2e success", d, flush=True)
        imageio.mimsave(f"{A.out}/demo_task{tid}{A.tag}.mp4", frames, fps=30)
        print(tid, task.language, "success", ok, flush=True)
        all_frames += frames; env.close()
    imageio.mimsave(f"{A.out}/demo_all{A.tag}.mp4", all_frames, fps=30)


if __name__ == "__main__":
    main()
