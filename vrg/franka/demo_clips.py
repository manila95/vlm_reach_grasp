"""Side-by-side demo clips for the notebook: the same episode (same scene and target) under two configs.

Inputs are recordings from `run_isaac --episodes ...` (see tools/isaac/queue_demo.sh); writes demo/franka/compare/*.mp4.
  python -m vrg.franka.demo_clips
"""
import glob, os
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

S = "/scratch/kaustubh/vlm_reach_grasp/isaac/runs"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/demo/franka/compare"
PAIRS = [  # name, episode, (run, label) left, (run, label) right
    ("look_bleach", "103_021_bleach_cleanser", ("demo_a", "a. external camera only"), ("demo_b", "b. + wrist look-then-grasp")),
    ("look_mustard", "104_006_mustard_bottle", ("demo_a", "a. external camera only"), ("demo_b", "b. + wrist look-then-grasp")),
    ("ik_pudding", "106_008_pudding_box", ("demo_e", "e. Isaac differential IK"), ("demo_b", "b. our IK (planned)")),
    ("vlm_gelatin", "101_009_gelatin_box", ("demo_vlm2b", "Qwen3-VL-2B (point)"), ("demo_vlm32b", "Qwen3-VL-32B (marks)")),
]


def frames(run, ep):
    f = glob.glob(f"{S}/{run}/videos/{ep}_*.mp4")[0]
    return [np.asarray(x) for x in imageio.mimread(f, memtest=False)], f.endswith("_ok.mp4")


def label(fr, text, ok):
    im = Image.new("RGB", (fr.shape[1], fr.shape[0] + 30), (250, 250, 248)); im.paste(Image.fromarray(fr), (0, 30)); d = ImageDraw.Draw(im)
    try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError: font = ImageFont.load_default()
    d.text((8, 6), text, fill=(20, 20, 20), font=font)
    d.text((fr.shape[1] - 90, 6), "success" if ok else "failure", fill=(27, 140, 90) if ok else (200, 60, 40), font=font)
    return np.asarray(im)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, ep, (r1, l1), (r2, l2) in PAIRS:
        (A, okA), (B, okB) = frames(r1, ep), frames(r2, ep)
        n = max(len(A), len(B)); A += [A[-1]] * (n - len(A)); B += [B[-1]] * (n - len(B))   # hold the last frame of the shorter
        V = [np.hstack([label(a, l1, okA), label(b, l2, okB)]) for a, b in zip(A, B)]
        imageio.mimsave(f"{OUT}/{name}.mp4", V, fps=20, quality=6, macro_block_size=1); print("wrote", name, n, "frames")
    clamp = glob.glob(f"{S}/demo_b/videos/114_051_large_clamp_*.mp4")
    if clamp:
        A, ok = frames("demo_b", "114_051_large_clamp")
        imageio.mimsave(f"{OUT}/slip_clamp.mp4", [label(a, "b. clamp: grasped, slips when lifting", ok) for a in A], fps=20, quality=6, macro_block_size=1)


if __name__ == "__main__":
    main()
