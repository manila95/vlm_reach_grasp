# VLM-guided reaching + learned local grasping (LIBERO-Object PoC)

A vision-language model shrinks the manipulation search space, and imitation learning only solves the last few centimetres.

1. **VLM (Qwen3-VL-8B, local, zero-shot)** parses the instruction (`pick=milk, place=basket`), then grounds both objects in one RGB-D image: depth-based object proposals, each zoomed in and verified with a yes/no query.
2. **Scripted reach** moves the gripper to a *handoff pose* above the object (offset and orientation taken from demo statistics).
3. **Learned local grasp**: a diffusion policy trained only on the ~1 s grasp-and-lift slice of the demos, with **egocentric inputs** (wrist camera, height, orientation, gripper; no absolute x/y), shared by all 10 tasks.
4. **Scripted transport + release** over the VLM-localized basket.

The full write-up, with every number, CI, ablation and embedded videos, is in **[`report/VLM_Reach_LocalGrasp_Report.ipynb`](report/VLM_Reach_LocalGrasp_Report.ipynb)**. A standalone HTML version is in `report/`; GitHub may not render a notebook this large, so download the HTML or open the notebook locally.

## Results (LIBERO-Object, 10 tasks × 20 episodes)

*Shuffled* = the 6 grocery items swap floor positions (seeded per episode). In 83% of shuffled episodes the target sits where no demo ever grasped. **All shuffled results are zero-shot**: no component was trained or tuned on shuffled scenes (see the report for the one calibration constant and the held-out confirmation).

| Method | Standard layouts | Shuffled layouts |
|---|---|---|
| End-to-end diffusion policy (task id) | 98.5% | 0.0% |
| VLA: OpenVLA-OFT 7B (Spatial-Forcing checkpoint) | 99.0% | 6.0% |
| SayCan-style (VLM planner + language-conditioned skills) | 97.0% | 0.0% |
| HAMSTER-style (VLM path + path-conditioned policy) | 91.0% | 13.5% |
| VLM + scripted grasp | 87.0% | 82.0% |
| **Ours: VLM handoff + egocentric local grasp (v5)** | **97.5%** | **78.0%** |

Held-out split (250 untouched episodes): ours v5 96.0% / 75.2%, end-to-end 97.6% / 0%.

How to read this:
* On standard layouts all learned methods are tied: LIBERO-Object puts each task's objects in nearly the same place every time.
* Under shuffle, methods whose learned policy must find the object grab **whatever now sits where the target used to be** (75–93% of episodes moved a wrong object).
* Most of our robustness comes from the *scripted* VLM-guided reach (it works at any location). The scripted grasp alone is as good on this rigid-object benchmark (82%); the learned grasp's advantage appears under localization error (2–4 cm noise: 84–90% vs 41% at 2 cm). See report §4.1c, §4.2.
* SayCan-style / HAMSTER-style are controlled re-implementations (same architecture, data and budget), not the original systems.

Side-by-side OOD videos (same scene, all five methods): `demo/ood_compare/` (`reel_all.mp4` has all 12 rule-selected scenes).

## Layout

```
vrg/                    all code
  common.py             LIBERO env helpers
  vlm.py                Qwen3-VL wrapper, depth proposals, zoom + yes/no verification
  pipeline.py           skills: move_to, handoff constants, transport, policy runner, layout shuffle
  policy.py             diffusion policy (ResNet18-GN + spatial softmax, 1D conditional UNet, DDIM)
  train.py              training: --mode local | full | saycan | hamster
  rollout.py            evaluation of every method (+ ablations, noise, held-out, high-res recording)
  ground_episodes.py    VLM grounding pass (cached per episode)
  saycan_plan.py        SayCan-style LM x affordance planner
  paths.py, build_hamster.py   HAMSTER-style path prompts
  vla_eval.py           wrapper around the Spatial-Forcing OpenVLA-OFT LIBERO eval
  compose_demo.py       side-by-side OOD comparison videos
report/build_report.py  generates the notebook from runs/eval/*/results.json
runs/eval/*/            per-episode results of every evaluation
data/                   small caches the report needs (datasets/checkpoints are not in the repo)
```

## Setup

Tested on Linux, 2× RTX PRO 6000 (96 GB), CUDA 12.8, Python 3.10.

```bash
# environment (key versions): torch 2.7 (cu128), robosuite 1.4.1, mujoco 2.3.7, transformers >= 4.57 (5.19 used),
# diffusers 0.33, qwen-vl-utils, LIBERO (github.com/Lifelong-Robot-Learning/LIBERO, installed editable)
source env.sh          # MUJOCO_GL=egl + NVIDIA EGL vendor (needed for headless rendering); sets $PY

# data: LIBERO-Object demos (~7 GB)
huggingface-cli download yifengzhu-hf/LIBERO-datasets --repo-type dataset --include "libero_object/*" --local-dir data/libero_raw
$PY -m vrg.build_dataset           # -> data/libero_object.npz + phase labels
$PY -m vrg.build_hamster           # HAMSTER-style path-overlaid images (only for that baseline)
```

## Reproduce

```bash
# policies (~1 h each on one GPU)
$PY -m vrg.train --mode local --cams wrist --mask_xy --steps 60000 --out runs/local_ego_50 --device cuda:0   # ours
$PY -m vrg.train --mode full    --steps 100000 --out runs/full_50    --device cuda:1                         # end-to-end
$PY -m vrg.train --mode saycan  --steps 100000 --out runs/saycan_50  --device cuda:0
$PY -m vrg.train --mode hamster --steps 100000 --out runs/hamster_50 --device cuda:1

# VLM stages (cached)
$PY -m vrg.ground_episodes; $PY -m vrg.ground_episodes --shuffle --out data/grounding_shuf.pkl
$PY -m vrg.saycan_plan;     $PY -m vrg.saycan_plan --shuffle

# evaluation (add --shuffle [--grounding data/grounding_shuf.pkl] for OOD)
V5="--auto_retract --lock_ori auto --retries 2 --j4_thresh=-1.0 --calib data/grounding.pkl"
$PY -m vrg.rollout --method hier_vlm $V5 --ckpt runs/local_ego_50/ckpt.pt --out runs/eval/v5strict_vlm
$PY -m vrg.rollout --method full    --ckpt runs/full_50/ckpt.pt    --out runs/eval/full_50
$PY -m vrg.rollout --method saycan  --ckpt runs/saycan_50/ckpt.pt  --out runs/eval/saycan
$PY -m vrg.rollout --method hamster --ckpt runs/hamster_50/ckpt.pt --out runs/eval/hamster

# report
python report/build_report.py && jupyter nbconvert --to notebook --execute --inplace report/VLM_Reach_LocalGrasp_Report.ipynb
```

The `run_*.sh` scripts are the exact queues used for the experiments in the report.

## Acknowledgements

LIBERO (benchmark and demos), Qwen3-VL, Diffusion Policy (Chi et al.), SayCan (Ahn et al.), HAMSTER (Li et al.), and the Spatial-Forcing / OpenVLA-OFT checkpoint used as the VLA baseline.
