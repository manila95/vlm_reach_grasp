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

## Results log

**E0, attribution (2026-10-31; LIBERO-Plus dev split, 505 tasks; success %):**

| Category | A VLM + learned | B GT + learned | C VLM + scripted | D GT + scripted | π0.5 |
|---|---|---|---|---|---|
| Light | 2 | 5 | 20 | 100 | 100 |
| Background | 24 | 45 | 47 | 100 | 98 |
| Camera | 70 | 99 | 52 | 100 | 89 |
| Layout | 73 | 93 | 62 | 87 | 87 |
| Language | 90 | 99 | 82 | 100 | 92 |
| Robot init | 99 | 99 | 82 | 99 | 65 |
| Sensor noise | 100 | 100 | 82 | 100 | 96 |
| **All** | **70** | **82** | **63** | **98** | **89** |

- **I1 is confirmed.** With perfect grounding the learned skill still gets 5% under light (scripted 100%) and 45% under background (scripted 100%).
- **Grounding costs 12 points overall** (A→B), and all of Camera (70→99), Layout (73→93) and Language (90→99). Camera is a grounding problem, not a skill problem.
- **The robot side is not the limit.** With perfect perception the modular system scores 98%, above π0.5's 89%. The whole gap to π0.5 is perception: grounding plus the skill's appearance robustness.
- **The learned skill still beats the scripted grasp under VLM noise** (A 70 vs C 63), as on LIBERO. So the goal is to make it robust, not to replace it.

**I4, failure taxonomy (full LIBERO-Plus run, ours; from the logged moved object vs target):**

| Category | Grasp, nothing moved | Wrong object | Right object, not placed | Other |
|---|---|---|---|---|
| Light | 282 | 0 | 11 | 0 |
| Background | 96 | 5 | 59 | 40 |
| Camera | 0 | 54 | 73 | 9 |
| Language | 0 | 30 | 14 | 0 |
| Layout | 10 | 22 | 30 | 12 |

- Light is pure perception: the skill never makes contact.
- Background mixes the skill (96) and the basket (59 carried but not placed; basket error 23 cm).
- Camera mixes wrong-object grounding (54) and an unexplained place miss with the right object (73; basket error only 3.5 cm). That needs videos (I5).
- Language failures are mostly grounding (30 wrong object).

**E2, wave 1 (own dev set, GT grounding, 200 episodes per cell; 2026-11-01):**

| Skill | Clean | Light | Texture | Both |
|---|---|---|---|---|
| v5 | 99.0 | 33.0 | 27.0 | 12.5 |
| photo | 91.5 | 95.5 | 77.5 | 80.0 |
| conv | 87.5 | 90.0 | 82.5 | 75.5 |
| photo+conv | 94.5 | 93.5 | 84.5 | 83.5 |
| norm | **99.0** | 95.0 | 80.0 | 68.0 |
| norm+photo | 96.0 | 97.0 | 80.0 | 74.5 |
| re-render (dr) | 89.5 | 91.5 | 93.0 | **94.0** |

- Every variant fixes most of the light failure.
- Only norm passes the clean gate (v5 CI 96.4–99.7).
- The clean drops are concentrated in milk (task 7): a tall carton that catches the basket rim at release, depending on grasp height. Milk is fragile for every skill (v5 25–55% under perturbation). So single-seed differences are inflated.
- Before deciding: second seeds of norm, photo+conv and dr. Still to come: dr+photo, depth, RGB-D, norm+dr. The depth runs were relaunched after a shape bug.

**E3b, basket/object centre estimators (own dev scenes, depth only; residual xy error after each estimator's clean-scene bias):**

| Target | Estimator | Bias | Clean (median) | Camera orbit (median / p90) |
|---|---|---|---|---|
| Basket | median of points (current) | 7.1 cm | 0.0 cm | 0.5 / 2.7 cm |
| Basket | bird's-eye box centre | 0.1 cm | 0.0 cm | 0.1 / 0.1 cm |
| Basket | rim-band box centre | 0.1 cm | 0.0 cm | 0.1 / 0.1 cm |
| Object | median | 1.6 cm | 0.5 cm | 0.6 / 1.1 cm |
| Object | box centre | 0.1 cm | 0.1 cm | 0.1 / 0.2 cm |

- I3 is solved at its root. The median is pulled toward the camera-facing side, so it needs a calibration that only holds for one viewpoint. The box centre is unbiased and viewpoint-invariant, so no calibration is needed.
- Decision: use the box centre for both targets (G4). It is to be confirmed on the LIBERO-Plus dev split, since its camera changes and clutter are larger than our orbit.

**E3, verification on own dev scenes:** every scheme (G0–G3) picks the right basket 100% and the right object 98–100% under light, texture and both. Our perturbations do not reproduce the LIBERO-Plus basket failure, so verification is not its cause.

**I2 root cause (diagnosis on 5 LIBERO-Plus dev light scenes; not tuning):**
- The basket estimate is identical across the five lightings, so lighting does not cause it.
- In these layouts two small items stand in front of the robot's pedestal. In the depth proposals they merge with it into 25 cm-tall clusters, and the VLM picks one of those crops as "the basket". The real basket is proposed correctly.
- Fix, **grounding v6**: a robot self-filter (the robot's own pixels removed before proposing; on a real arm, the standard model-based self-filter) and box centres (E3b).
- On the failing scene, v6 proposals are clean and the basket proposal lands 0.1 cm from the truth.
- Queued:
  - v6 re-grounding of standard (calibration, eval, shuffled) and all LIBERO-Plus scenes;
  - E4a, the v5 skill with v6 grounding on the dev split, which isolates the grounding gain.

**E4a, grounding v6 with the v5 skill (LIBERO-Plus dev split, 505 tasks):**
- All: 70 → 75% (perfect grounding: 82).
- Camera 70 → 92, background 24 → 35, layout 73 → 76; language unchanged (90).
- Standard scenes: object hit 98.5 → 99%, shuffled 98%; raw basket error 7.0 → 0.1 cm. No regression.
- The remaining grounding gap (layout, language) is wrong-object identification by the VLM.

**E2, waves 2–4 (own dev set; clean / light / texture / both):**
- norm+dr: 96.0 / 96.5 / 97.5 / 98.5, the most robust. Clean is one episode under the gate.
- drphoto: 95.5 / – / 95.0 / 95.0.
- depth: 94.0 / – / 93.5 / 95.0.
- RGB-D: 97.5 / 67.0 / 83.0 / 73.5 (its RGB stream is unaugmented).
- Second seeds:
  - photoconv: 99.0 / 99.0 / 88.5 / 87.5;
  - norm: 99.0 / 96.5 / 79.5 / 78.0.
- Queued: second seeds of norm+dr and drphoto. The decision follows the two-seed rule.
