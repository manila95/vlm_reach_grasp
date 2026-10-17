# Development log (condensed)

This project was developed with Claude Code over two sessions (2026-10-07/08). This is a curated summary of what was tried, what failed, and why decisions were made. The raw session transcript is not included (size; it contains machine-specific and personal details).

## Day 1: building the pipeline
- **Setup.** LIBERO-Object chosen: each scene has 7 objects and the instruction names the target, so finding the right object is the core difficulty. Headless rendering needed the NVIDIA EGL vendor forced (`env.sh`).
- **Demo statistics → handoff.** The gripper closes ~51 steps into a ~149-step demo. 20 steps before closing, the end effector is ≈15 cm above the object (+2.5 cm in x). That pose became the scripted reach target and the start of the local skill's training slice.
- **Grounding.** Asking Qwen3-VL directly for a point/box found the target 42–64% of the time (labels are a few pixels tall; Qwen3-VL-32B was no better). **Depth proposals (bird's-eye clustering) + zoomed yes/no verification** reached 50/50, then 98.5% / 98% on the eval scenes.
- **Scripted controller bugs.** Transport pushed the basket; releasing above the near edge dropped objects on the rim. Fixed with a high transport and release over the basket centre (scripted oracle grasp 0% → 90%).
- **Policies.** Our own diffusion-policy implementation. Local skill: handoff→lift slice (34% of frames), no task id. Baseline: full demos + task id, more steps.
- **First results.** All ≈94–98% on standard layouts. A layout shuffle was added as the OOD test: the end-to-end policy dropped to 0% (it cleanly picks the object at the memorized location).
- **Locality matters.** The first local skill (agentview + wrist + absolute x/y) also dropped under shuffle (48%). The **egocentric** version (wrist only, x/y masked) reached 64.5% → the skill must be truly local.

## Night 1: baselines and improvements
- **Baselines** (same architecture/data/budget): SayCan-style (VLM LM × affordance planner + language-conditioned skills), HAMSTER-style (keypoint path drawn on the image + path-conditioned policy), and the public Spatial-Forcing OpenVLA-OFT 7B checkpoint via the authors' eval code. Under shuffle: 0%, 13.5%, 6%.
- **Failure diagnosis.** Remaining shuffled failures were kinematic: near the workspace edge the learned grasp's wrist tilt drives joint 4 to its limit (straight elbow) and transport stalls. Stronger OSC null-space gains didn't help. Iterations: v2 full orientation lock (+shuffle, but broke tomato sauce, whose grasp needs yaw), v3 roll/pitch lock, v4/v5 posture-aware lock + retract + grasp retry. **v5: 97.5% / 78% (strict zero-shot), confirmed on a held-out split (96% / 75.2%).**
- **Handoff sweep.** The skill's basin is a few cm (98% → ~50% at +10 cm height or 6 cm lateral). A longer training slice widens the basin but leaks layout (shuffled 39.5%).
- **Data efficiency.** Egocentric skill with 10 demos/task: 89.5% / 75%. End-to-end with 10 demos/task: 91.5% / 0%.

## Day 2: scrutiny
- **Zero-shot check.** The grounding bias had been fit on shuffled calibration scenes; re-run with standard-scene calibration (strict) → within noise; strict numbers are the headline.
- **What the shuffle tests.** Targets move 10–39 cm and in 83% of episodes land where no demo ever grasped. Splitting results shows HAMSTER-style *does* follow paths (79–91% at previously grasped locations, 0% elsewhere): its failure is spatial extrapolation. Ours is flat across both subsets, because the scripted reach makes location irrelevant. The README and report state that most of the OOD robustness comes from that scripted reach.
- **Side-by-side demos.** Rule-selected scenes (no outcome cherry-picking) with all five methods on the same seed: `demo/ood_compare/`.

## Benchmark expansion: RoboCasa365 + Colosseum (2026-10-10 → 15)
- Pipelines in `vrg/bench/`; results in section 6 of the main report. Jobs now go through `tools/launch_job.sh` (per-job systemd unit with a RAM cap), after an OOM kill on 2026-10-10 took down every job and the session.
- **Compared with the papers:**
  - Colosseum end-to-end: 50% with no perturbation (PerAct 51% on the same 12 tasks), 21% under perturbation (PerAct 43%).
  - Our HAMSTER-style: 17% vs the paper's 57% (HAMSTER+3D-DA). The likely cause is that it was trained on perfect demo paths and tested on noisy VLM paths.
  - RoboCasa end-to-end: 10% vs the paper's DP at 15.7%, which used 60× more data.

## Franka in Isaac Lab (2026-10-15 → 17): `report/Franka_Isaac_Experiments.ipynb`
- **Why.** The user will move to a real Franka (franky + RealSense ext/wrist RGB-D + Franka Hand, local VLMs only). Everything above a `Robot` interface (perception, VLM, grasp, IK, episode) is shared between the Isaac backend and an untested franky backend.
- **Setup lessons:**
  - Isaac Sim 5.1 is needed on the Blackwell GPUs.
  - "No device could be created" came from conda's activate.d forcing Mesa EGL; `isaac/env.sh` fixes it.
  - CPU PhysX is 36× faster than GPU PhysX for this scene.
  - Kit ignores SIGTERM, so use `os._exit` and SIGKILL.
- **Robot side, with a perfect VLM (257 eval eps):**
  - A wrist "look" before grasping takes success from 49% to 78%.
  - A grasp angle from PCA is required: a fixed angle gives 28%.
  - Our planned IK (goal IK with limits, null space and restarts, then min-jerk / straight line) beats Isaac's one-step differential IK, 78% vs 56%.
- **VLM size (Qwen3-VL 2B → 32B):** 75% → 81% grounding, and about 85% pick success once grounding is right. Most remaining failures are the VLM's.
  - Bugs found and fixed along the way:
    - Splitting the 32B model over two GPUs gave NaN outputs.
    - The external camera's segmentation lagged its RGB by one render, so offline labels missed the target in 6/257 scenes.
- **Reactive grounding (the user's idea: more inference instead of a bigger model).**
  - Dev probes chose the design: the arm follows its first choice and logs every view, without acting on them.
  - The best policy pools 3 grounding modes on the external view with 2 wrist re-asks along the approach. With it, 4B reaches 86% / 72%, beating 32B one-shot (81% / 68%).
  - The user then suggested that close views get obscured and that the last unobscured detection should be used. Per-view accuracy does peak at a camera height of ≈42 cm and drops closer in for 2B–8B.
  - But "last unobscured view decides" was worse than one-shot on eval: single views are noisy, and a fixed area gate fires at the wrong time. Pooled votes win.

- **Failure modes (2026-10-17).** Every failed perfect-VLM episode was re-run with full diagnostics: wrist view with ground-truth masks, planned and reached poses, object poses. 74/77 reproduced. Causes:
  - **Touching objects merge into one bird's-eye cluster** (≈ 20 eps): no proposal for the target, or an inflated grasp width.
  - **The Franka palm (≈ 20 cm along the closing direction, 4.5 cm above the fingertips) hits taller neighbours** (12 of 15 `reach`/collision failures vs 2 of 22 successes). The "reach" label was really a collision that didn't move anything 3 cm.
  - **A 7.5 cm planner cap + 1 cm margin rejects a standing cracker box** that fits the 8 cm hand (the grasp angle is accurate, median 0.7°).
  - **VLM:** 22/23 errors are same-kind look-alikes. Descriptive names ("blue SPAM can") add +4 to +14 points, more than 2B → 32B. When the three modes agree, grounding is 99% right.

## Open next steps
Learned target-relative reach (replace the scripted reach), re-grounding at phase boundaries (the VLM is currently queried once), straight-line (norm-scaled) reach, a learned place skill, layout-randomized training for the baselines, multiple seeds, harder suites (LIBERO-90, RoboCasa).
