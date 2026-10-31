# Plan: perceptual robustness of the VLM + reach + local-skill system (2026-10-31)

Trigger: on LIBERO-Plus (libero_object, 2,518 tasks) our v5 system scores 69.9% vs π0.5-LIBERO's 89.1%.
- It wins on geometry: robot initial state 98% vs 69%; sensor noise 99% vs 95%.
- It collapses on appearance: light 1% vs 99%; background texture 19% vs 98%.

This plan lists the issues the data shows, the candidate fixes for each, and the experiments that decide between them without tuning on the test set.

## 1. What the data says (LIBERO-Plus run, `runs/liberoplus/ours`)

| Category | Success | Main failure | Object xy error at failure | Basket error (median) |
|---|---|---|---|---|
| Light | 1% | grasp 282/297 | 1.6 cm (same as successes elsewhere) | 25 cm |
| Background | 19% | grasp 115, place 85 | 1.3–2.1 cm | 23 cm |
| Camera | 66% | place 132 | 2.6 cm | 6 cm (place fails: 3.5 cm) |
| Language | 88% | "place" 44 | **17.5 cm**: a wrong object was grasped | 7 cm |
| Layout | 82% | place 59, grasp 15 | 2.6 cm | 7 cm |

## 2. Issues and hypotheses

**I1. The learned grasp skill is not invariant to illumination and texture.**
- Under light and background changes, the handoff is as accurate as in successful episodes, yet the grasp fails.
- The skill sees one 128 px wrist RGB stream. It was trained on 50 demos per task under one lighting and one table texture, with random crop as the only augmentation.
- Hypothesis: the failure is a covariate shift in pixel statistics, not a wrong pose.
- Alternatives to rule out:
  - the scripted reach behaves differently in these scenes (it shouldn't: it's state-based);
  - the perturbed scenes change physics or object meshes.

**I2. Basket grounding fails under appearance shifts.**
- Basket error is 23–25 cm under light and background changes, vs ≈ 7 cm on standard scenes.
- The object grounding stays at 1.6 cm in the same scenes. So the difference lies in the basket proposals (a large, low-texture object whose depth cluster can merge with a retextured table) or in the VLM's yes/no verification on dark crops.
- To check: proposals vs verification, on logged per-proposal data.

**I3. The basket offset is fixed rather than measured.**
- There is a ≈ 7 cm basket bias even on standard scenes. It is absorbed by a calibration fit on standard scenes and by the basket's size.
- Under new viewpoints the bias can change.
- Root cause to find: box vs cluster centre, or top surface vs body origin.

**I4. Wrong-object grasps are labelled "place".**
- Language failures (17.5 cm object error) and some Camera failures are grounding misses that only show up at the place check.
- The failure taxonomy must separate wrong-object, grasp, transport-drop and release-miss.

**I5. Camera-view place failures with correct grounding (132).**
- The cause is unknown: a drop during transport, or a release that misses.
- Needs videos and the new taxonomy before any fix.

**I6. (MimicGen) The in-distribution cost of composition.**
- C1 also loses 37 points on D0.
- Not addressed here; it is tracked in `PLAN_contact_segment_vs_whole.md`.

## 3. Candidate solutions (logic first)

**For I1, ordered from "remove the nuisance from the input" to "learn invariance":**

| Id | Fix | Why it should work | Risk |
|---|---|---|---|
| S0 | baseline (crop only) | reference | — |
| S1 | photometric augmentation: brightness, contrast, saturation, colour tint, gamma, noise, blur; the same draw for both obs frames | lighting shifts are mostly global photometric transforms | cannot model shadows or texture changes |
| S2 | texture randomisation: random convolution (RandConv) mixed with the image | breaks reliance on surface texture and colour | may hurt fine geometry cues |
| S3 | **render-time domain randomisation**: replay each demo's simulator states and re-render the wrist view under randomised lights (direction, diffuse, ambient, specular, shadows) and randomised table/floor materials (colours, procedural textures) | physically consistent shadows and specularities; the same actions, so labels stay exact | must not use LIBERO-Plus's assets or generators (leakage); ~25k frames × K renders |
| S4 | per-frame input normalisation (per-channel standardisation of the wrist image) at train and test | removes global brightness and contrast analytically, at no cost | discards some useful absolute colour |
| S5 | **depth (or RGB-D) wrist input** | depth is invariant to light and texture by construction, and the real Franka has a wrist RealSense | depth noise on real sensors; thin and transparent objects |
| S6 | pretrained visual encoder (ImageNet / DINOv2 / R3M ResNet), fine-tuned | pretraining on diverse images gives robustness for free | heavier; may overfit with 50 demos |

**For I2/I3 (grounding), measured offline: grounding needs no rollouts.**
- G1: photometric normalisation (auto-exposure / gamma) of the image given to the VLM.
- G2: basket proposals from the depth structure (a rim of points around a hollow interior) instead of colour-agnostic clusters merged with the table.
- G3: pooled multi-mode votes, as in the Franka track (several grounding modes; agreement implies 99% accuracy).
- G4: measure the bias instead of calibrating it (e.g. the cluster centroid at rim height).
- G5: re-ground the basket at the phase boundary, from the post-grasp viewpoint.

**For I4/I5:** a failure taxonomy first (wrong object / no grasp / drop in transport / release miss) from object poses logged per phase. Then decide.

## 4. Data hygiene (decided before any result)

Three evaluation pools, never mixed:

1. **Own dev set (selection).** The 10 standard LIBERO-Object tasks × the `CALIB_EPS` and `EVAL_EPS` initial states. Perturbations come from our own generator, not LIBERO-Plus's:
   - light: random diffuse and ambient scale 0.15–2.5, direction ±60°, coloured light;
   - table and floor: random RGB, procedural checker and noise textures.

   Our generator's parameters are seeded separately and documented.
2. **LIBERO-Plus dev split (sanity only).** A stratified 20% of LIBERO-Plus (category × difficulty, seed 0), fixed now in `data/liberoplus_split.json`. Used only to confirm that dev-set gains transfer, never to choose hyperparameters.
3. **LIBERO-Plus test split (final, once).** The remaining 80%. Every method, including the released VLAs, is reported on this split, with Wilson CIs. The full-2,518 numbers stay in the report for comparability.

Regression gates, on the standard protocol in the vlmrg env:
- standard LIBERO-Object (200 episodes) must stay within CI of v5 (97.5%);
- the shuffled layout (200) must not drop;
- the held-out split (250) is checked once at the end.

## 5. Experiments

**E0, attribution (no training).** On the LIBERO-Plus dev split, swap one component at a time:

| Condition | Grounding | Grasp | Answers |
|---|---|---|---|
| A | VLM | learned skill | the current system |
| B | ground truth | learned skill | how much is grounding (I2–I4)? |
| C | VLM | scripted grasp | is the learned skill the bottleneck (I1)? |
| D | ground truth | scripted grasp | the ceiling of the robot side |

Prediction under H1: on Light, B ≈ A (low) while C ≫ A. If not, I1 is not the main cause and the plan changes.

**E1, an offline perception probe (fast and cheap; used to screen the variants).**
- Render held-out demo states (5 demos per task, not in training) under the dev perturbations.
- Measure the skill's predicted-action error against the demo actions, and the success-relevant first-chunk error.
- Compare against the same states rendered normally.
- This ranks S0–S6 in minutes before any rollouts.

**E2, closed-loop selection on the own dev set.**
- Variants: S0, S1, S2, S1+S2, S3, S1+S3, S4, S5 (depth), S5 (RGB-D), S6.
- 2 training seeds each for the top 3 after E1. Diffusion sampling is unseeded, so compare full-suite numbers with CIs.
- Metrics: success on standard, light-dev and texture-dev scenes; the failure taxonomy.
- Decision rule (fixed now): maximise mean perturbed-dev success, subject to the standard-scene success staying inside v5's CI.

**E3, grounding fixes offline.** G1–G5 on own-dev scenes and the LIBERO-Plus dev split. Metrics: basket and object error, and hit rate, using the logged proposals. Choose by the same rule, with standard-scene grounding error as the gate.

**E4, combine and transfer.**
- The chosen skill and the chosen grounding go through the regression gates (standard, shuffle).
- Then a sanity check on the LIBERO-Plus dev split.
- Then **one** run on the LIBERO-Plus test split, reported per category and difficulty against all baselines on the same split.

**E5, carry it into the contact-segment study.**
- Retrain MimicGen P2 with the winning recipe and evaluate D0/D1/D2, plus relit and retextured MimicGen scenes from the same generator.
- Question: does photometric invariance keep P2's off-distribution advantage, and does it change the P1 vs P2 gap?
- P1 also gets the same augmentation for a fair comparison.

**E6 (real-arm relevance).** If S5 (depth) wins or ties, it is the natural choice for the real Franka (wrist RealSense). Test it with simulated depth noise (dropout, edge noise).

## 6. Expected outcomes and what each would mean

- **C ≫ A on Light, and S1/S3 fix it on dev:** the claim becomes "a local skill is robust to geometry by construction and to appearance via augmentation". That's a cheap fix, and it strengthens the segment-study argument.
- **S5 (depth) alone fixes it:** a stronger claim (invariance by input design), aligned with the real arm.
- **Nothing fixes Light without hurting standard scenes:** report the trade-off honestly, and keep the released VLAs' appearance robustness as their advantage (pretraining scale).

## 7. Cost (one 96 GB GPU's share)

| Step | Cost |
|---|---|
| E0 | ≈ 4 × 140 rollouts |
| E1 | minutes per variant |
| Skill training | ≈ 65 min per run, ~12 runs |
| S3 rendering | ≈ 25k frames × 4 variants |
| E2 rollouts | ≈ 600 per variant |
| E3 | offline grounding (≈ 18 s per scene with the 8B VLM) |
| E4 | one 2,000-task test run |

Total ≈ 1.5–2 days of compute alongside the running queues.
