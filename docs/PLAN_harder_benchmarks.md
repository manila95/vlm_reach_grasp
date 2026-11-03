# Plan: the same comparison on harder benchmarks (2026-11-03)

Why: LIBERO-Object is saturated (VLAs 88–89%, ours 94% with a stronger VLM). Harder environments show whether
"modular + better VLM" holds when manipulation is harder than a tabletop pick-and-place.

## Decision (user, 2026-11-03)
Start with **no training on our side**: VLM grounding + scripted reach + **scripted grasp** (+ scripted transport/place),
against **released VLA checkpoints run zero-shot as released**.

## Benchmark 1: RoboCasa365 (kitchens, PandaOmron mobile manipulator; robocasa 1.0.1 is installed)
- Tasks: our 5 Atomic-Seen pick-and-place tasks (counter→cabinet, counter→stove, drawer→counter, sink→counter,
  toaster→counter), 50 episodes each, pretrain split, then widen to the other pick-place atomic tasks.
- Released baselines (`robocasa/robocasa365_checkpoints`, trained on 300 tasks × 100 human demos):
  Atomic-Seen average (65 tasks, official): DP 15.7, π0 34.6, π0.5 39.6, GR00T N1.5 43.0.
  The comparison runs these checkpoints on **our** 5 tasks, same episodes, same success check.
- Harness: robocasa-benchmark/openpi fork (`examples/robocasa/main.py`, config `pi05_pretrain_human300`,
  `pi0_robocasa_pretrain_human300`, replan every 5 steps, task horizon from `get_task_horizon`); GR00T `scripts/run_eval.py`.
- Ours (new, no training): the VLM from the E8 study grounds the pick and the place target from RGB-D; scripted base/arm
  reach, scripted top-down grasp (grasp height from `rc_pick_top`), scripted transport and release.
  The earlier learned-skill run (17% / 11% on pretrain / target kitchens) is the reference for what scripted must beat.
- Metrics: success with Wilson CIs, paired tests against each VLA, per-task and per-phase failure labels, execution time
  (robot time + model time), as on LIBERO-Plus.

## Next candidates (after RoboCasa)
- Colosseum (RLBench perturbation suite, the analogue of LIBERO-Plus; pipeline exists).
- Other benchmarks LeRobot ships: RoboTwin 2.0, VLABench, RoboCerebra, RoboMME (released checkpoints to be checked).
- SimplerEnv (WidowX / Google robot): released VLAs exist, but embodiments need a new reach/grasp backend.

## Steps
1. Download official π0.5, π0 and GR00T N1.5 multitask checkpoints (parameters only; running).
2. Build the fork's env; serve π0.5 and evaluate the 5 tasks × 50 episodes (reproduce the official protocol on our tasks).
3. GR00T N1.5 the same way.
4. Ours with scripted grasp and each grounder (offline grounding study on cached RoboCasa scenes first).
5. Analysis, slides, report.
