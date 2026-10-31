# Plan: learning only the contact-rich phase vs whole trajectories (2026-10-29)

## Question
Does it help to learn **only the contact-rich manipulation phase**, with a VLM plus a planned reach getting the robot to the
right place, instead of learning whole trajectories from the same demonstrations? "Help" means better generalization
(spatial, object, visual, viewpoint, initial state), better data efficiency, and no loss in distribution.

**Hypotheses**
* **H1:** a policy trained only on contact-phase segments, with object-relative / egocentric inputs and composed with
  VLM-guided reaching, generalizes better than a whole-trajectory policy trained on the same demos with the same model and compute.
* **H2:** in distribution they are on par.
* **H3:** the gain comes from the policy only having to model the variance of the contact phase. It does not come merely
  from being placed at a good start pose (control C1 below separates the two).

**What we already know.**
* **LIBERO-Object (grasp only):**
  * the egocentric local skill scores 97.5% standard / 78% shuffled, against 98.5% / 0% for the whole-trajectory policy;
  * but the scripted grasp alone gets 82% shuffled, so much of the robustness came from the scripted reach;
  * the learned grasp mattered under localization noise.
* **Contact-rich tasks:** a scripted controller cannot do the contact phase, so the learned segment carries the result.
  This is the regime that tests H1.

## Closest prior work and how this differs
* **HITL-TAMP** (Mandlekar, Garrett, Xu, Fox, CoRL 2023). TAMP handles the free-space parts; humans teleoperate only the
  contact-rich parts. Policies trained on those segments beat training on full demos, in robosuite.
* **SkillMimicGen / SkillGen** (Garrett et al., 2024).
  * It segments demos into alternating motion and skill segments, adapts skills in the object frame, and connects them
    with motion planning.
  * The deployed Hybrid Skill Policy alternates learned contact-rich skills with planned motion.
  * Its stated guidance: learning should focus on the contact-rich segments, where motion planning is ineffective.

**What is new here.** Both rely on TAMP with known object models and poses, are evaluated mainly on success and data
efficiency in sim, and use small from-scratch policies. This project instead:
1. replaces the planner's object knowledge with **open-world VLM grounding from RGB-D + language**, which DROID-scale
   deployment needs;
2. measures **generalization along standard shift axes** (MimicGen D0→D2, LIBERO-Plus/PRO, RoboCasa, PolaRiS), not only
   in-distribution success;
3. compares **segment-only vs whole-trajectory fine-tuning of pretrained VLAs** (small → large), including on real-world
   DROID data;
4. includes the **C1 control** (a whole-trajectory policy started at the same handoff) and egocentric inputs, to explain
   *why* any gain occurs.

SkillMimicGen's Hybrid Skill Policy, re-implemented with oracle object poses, is the natural upper reference for the
composed system on MimicGen.

## The unit we learn: the contact phase
* **Contact phase:** from a pre-contact handoff to the end of the interaction with one object or part.
  * The handoff is k steps, or d cm, before first contact with the task-relevant object.
  * The end of the interaction is a release, or the object/part reaching its goal state.
* **Multi-stage tasks give several phases.** Coffee, for example, has pick pod → insert pod → close lid.
* **Free-space motion between phases** (approach, transit) is done by a VLM-grounded planned reach to the next handoff.
  This matches MimicGen's object-centric subtask decomposition.
* **Handoff pose at test time:** the interaction frame times a start offset.
  * The VLM grounds the *interaction target*: object or part, such as a handle, hole, knob or lid. Depth gives a 3-D
    point; an orientation comes from depth (PCA / normal) or from the VLM.
  * The start offset is the mean, or a sample, of the segment start poses in the target frame, taken from training data.
* **Segment boundaries:**
  * **Sim datasets:** ground-truth contact (contact sensors / object-pose change) and the datasets' own subtask
    annotations. MimicGen source datasets carry subtask-termination signals (to verify).
  * **DROID (real, no contact labels):** gripper close/open events, the end-effector slow-down before contact, and
    optionally VLM-based temporal segmentation of the wrist video. The heuristic is validated first on sim datasets,
    where the ground truth is known.

## Policies compared (same architecture, same demos, same training budget)
| | Trained on | Inputs | At test time |
|---|---|---|---|
| **P1 whole-trajectory** (standard baseline) | full demos | standard (ext + wrist + proprio), language/task-conditioned | runs alone |
| **P2 contact-segment (ours)** | contact-phase segments only | egocentric: wrist camera + proprio relative to the handoff; no absolute position | VLM + reach to each handoff, then P2 |
| C1 whole-trajectory, started at the oracle handoff | full demos | as P1 | placed at the same handoff P2 gets, so it receives the same help. **Isolates "training on segments" from "starting at the handoff".** |
| C2 contact-segment with global inputs | segments | ext camera + absolute pose | as P2. Isolates egocentric inputs (on LIBERO, locality mattered: 48% vs 64.5% shuffled) |
| C3 scripted contact phase (where definable, e.g. grasp, push) | – | – | lower bound: shows when learning is needed at all |
| C4 segment-length sweep (handoff 2 / 5 / 10 / 20 cm before contact) | segments | as P2 | basin vs layout leakage (LIBERO: longer slices leaked layout) |

**Two model scales, so the answer does not depend on one architecture:**
1. Our small diffusion policy trained from scratch: about 1 h per training, cheap, used for the full grid.
2. A pretrained VLA fine-tuned both ways from the same base: π0.5-base / π0-base via openpi, or SmolVLA via LeRobot if
   compute is tight. This tests whether pretraining erases the benefit.

## Generalization axes
* **Spatial:** layouts and initial object poses outside training (MimicGen D0 → D1/D2, LIBERO shuffle / LIBERO-Plus layout, RoboCasa layouts).
* **Object instance:** unseen objects in the same category (RoboCasa, LIBERO-PRO).
* **Visual:** lighting, textures, background (LIBERO-Plus, Colosseum).
* **Camera viewpoint:** LIBERO-Plus, Colosseum.
* **Robot initial state:** LIBERO-Plus. Expected to be trivial for P2, since the reach normalizes it.
* **Distractors / clutter.**
* **Handoff noise:** inject 1–4 cm and 5–15° errors at the handoff. This measures the composed system's main weakness
  (VLM localization) and the size of each policy's basin.
* **Data efficiency:** 10 / 50 / 200 / 1000 demos per task.

## Benchmarks, in order
**A. MimicGen (robosuite 1.4, our LIBERO stack) — primary, start here.**
* Contact-rich tasks with 1000 demos each and built-in spatial-generalization splits:
  * Square (insertion);
  * Threading (needle);
  * Coffee (pod insert + lid);
  * Three-Piece Assembly;
  * plus Stack, Nut Assembly, and Mug Cleanup (drawer).
* Train on D0, test on D0 / D1 / D2.
* MimicGen's own subtask decomposition gives the segments and the reference objects.
* The oracle handoff is exact in sim, so VLM and reach errors can be added separately.

**B. Reuse the platforms we already run.**
* LIBERO-90 / LIBERO-Long: drawers, stove knobs, microwave. Evaluated with the LIBERO-Plus (7 perturbation dimensions)
  and LIBERO-PRO (objects, initial states, instructions, environments) generalization suites.
* RoboCasa365 contact-rich atomic tasks: doors, drawers, knobs, levers, with held-out kitchens and objects.
* Colosseum / RLBench: insertion and stacking under the 12 perturbation factors.

**C. DROID (real data) — the standard-dataset result.**
* **Data:** a DROID subset filtered for contact-rich language (open/close drawer, insert, wipe, stack, put in, pour),
  segmented with the validated heuristic.
* **Training:** fine-tune the same base VLA two ways, whole trajectories vs contact segments, with joint-position actions
  (simulators handle them; our zero-shot π0.5 test showed joint-velocity policies do not transfer to sim).
* **Evaluation:**
  * PolaRiS: Gaussian-splat real-to-sim DROID scenes in Isaac Sim, r ≈ 0.9 with real-world success and 0.98 with RoboArena;
  * sim-evals: DROID Isaac scene, zero-shot for joint-position DROID policies;
  * IsaacLab-Arena pick-and-place variants;
  * then an A/B on the real Franka (RoboArena-style pairwise).
* **Embodiment caveat:** DROID uses a Robotiq gripper and ZED cameras. Evaluate in DROID-matched sims, and on our
  Franka-Hand cell only with the tool retargeting already built (`vrg/franka/vla.py`).

**D. Optional second embodiment.** Bridge V2 + SimplerEnv (WidowX; visual-matching and variant-aggregation protocols).
Its tasks are mostly pick-and-place, so this is lower priority for the contact-rich question.

## Measurement
* Success with Wilson CIs, and ≥ 3 training seeds per configuration (diffusion sampling noise: lesson from LIBERO).
* Per-axis drop relative to in-distribution.
* Failure attribution with the existing taxonomy: VLM grounding, handoff/reach, contact phase, transit.
* Two fairness controls:
  * **equal gradient steps:** segments are about 30–40% of the frames, so P2 sees fewer frames;
  * **equal compute.**
* Episode time, with the timing harness from the Franka track.

## Phases and compute (2× 96 GB GPUs, shared)
| Phase | Work | Compute | Time |
|---|---|---|---|
| 0 | Segmentation tooling (sim ground truth + DROID heuristic, validated on sim); handoff-offset statistics; VLM interaction-point grounding (part-level: handle / hole / knob) | CPU + VLM | 2–3 days |
| 1 | MimicGen: 4 contact-rich tasks × {P1, P2, C1, C2} × 3 seeds, D0 → D0/D1/D2, handoff-noise and data-efficiency sweeps, C4 on 2 tasks | ≈ 50–80 GPU-h (small diffusion policy) | ≈ 1 week |
| 2 | LIBERO-90/Long + LIBERO-Plus/PRO and RoboCasa contact-rich subsets, same recipe; existing pipelines | ≈ 60 GPU-h | ≈ 1 week |
| 3 | DROID subset → two fine-tunes of one VLA (whole vs segments), eval in PolaRiS / sim-evals | full π0.5 fine-tune is heavy (openpi: full DROID ≈ 2 days on 8×H100; LoRA did poorly for them). A subset with 2×96 GB is feasible only for a smaller model (SmolVLA) or a short π0.5 fine-tune on a subset | 2–3 weeks |
| 4 | Real Franka A/B on 3–4 contact-rich tasks | robot time | 1 week |

## Risks
* **Interaction frames, not points:** insertion and articulation need the right approach orientation. The VLM plus depth
  must supply it.
  * Mitigation: start-pose statistics in the object frame, plus orientation from depth.
  * The oracle-handoff runs bound the cost.
* **Segment boundaries in real data** are heuristic. Validate on sim, and report sensitivity to the boundary choice (C4).
* **Fairness:** C1 is essential; without it, a P2 win could just mean "starting close helps".
* **Benchmark leakage:** MimicGen data is generated in the same simulator, a known critique of sim-generated benchmarks.
  The DROID tier answers it with real data.
* **Sim-eval fidelity for DROID policies:** use joint-position models and DROID-matched scenes (PolaRiS / sim-evals).
  Do not use our own cell for that tier.

## Decisions (2026-10-30)
1. **Models.** Start with a small VLA for the proof of concept, then move to larger VLAs.
   * Proof of concept: our diffusion policy (fast grid) and SmolVLA (≈ 0.45 B; LeRobot).
   * Larger: MolmoAct (AI2, ≈ 7 B, built on Molmo). It predicts depth tokens and an image-space trajectory trace
     before its actions, so it is a natural comparison for "VLM reach + learned contact phase". Also π0.5 later.
2. **Target deployment:** the real Franka, across DROID's task distribution. The sim tiers are stepping stones; the
   DROID tier and the real A/B are the deliverable.
3. **Start with Phase 0 → Phase 1 on MimicGen,** with a standard evaluation protocol:
   * MimicGen/robomimic task definitions, horizons and success checks;
   * cross-distribution: train D0, test D0/D1/D2;
   * the MimicGen-paper setting (train and test on D1) as the in-distribution reference.

4. **No VLA training from scratch (user, 2026-10-30).** Pretrained VLAs are evaluated as released. "Ours" is the VLM +
   reach + a small imitation or conditional-imitation policy for the contact phase. This gives two comparisons:
   * **(A) Controlled study (same architecture, small IL policies).** Whole-trajectory vs contact-segment, with controls
     C1–C4, on MimicGen (D0 → D0/D1/D2) and LIBERO. This answers *whether learning only the contact phase helps*.
   * **(B) Against the field.** Our composed system vs off-the-shelf pretrained VLA checkpoints (OpenVLA-OFT, π0 / π0.5,
     SmolVLA, MolmoAct; each with its authors' benchmark fine-tune where released).
     * Sim: **LIBERO + LIBERO-Plus / LIBERO-PRO perturbation suites**, the standard VLA generalization evaluation, with public checkpoints.
     * DROID tier: released DROID VLAs (π0.5-DROID, π0-FAST-DROID, …) vs ours, in PolaRiS / sim-evals scenes, then the
       real Franka. Ours on DROID uses a small segment policy trained on DROID contact segments.

### Environment notes
* MimicGen datasets must be replayed with **MuJoCo 2.3.2**. 2.3.7 (our LIBERO env) changes the arm dynamics.
* So MimicGen runs in its own env, `/scratch/kaustubh/vlm_reach_grasp/envs/mimicgen` (source `mimicgen/env.sh`):
  robosuite 1.4.1, robomimic d0b37cf, torch cu128.

## Earlier open questions
1. Model for the DROID tier: π0.5 (closest to the state of the art, but compute-heavy) vs SmolVLA / our diffusion
   policy (feasible on our GPUs).
2. Is the real-Franka A/B in scope, and which contact-rich tasks (drawer, peg insertion, wiping, lid)?
3. Start with Phase 0 → Phase 1 on MimicGen (recommended).
