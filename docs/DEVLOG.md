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

## Open next steps
Learned target-relative reach (replace the scripted reach), re-grounding at phase boundaries (the VLM is currently queried once), straight-line (norm-scaled) reach, a learned place skill, layout-randomized training for the baselines, multiple seeds, harder suites (LIBERO-90, RoboCasa).
