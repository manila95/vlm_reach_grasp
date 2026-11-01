"""Builds report/Contact_Segment_Study.ipynb: learning only the contact-rich phase vs whole trajectories (MimicGen first).

Reads runs/mimicgen/* (copied from /scratch by `python -m vrg.mg.report_assets`) and demo/mimicgen/*.
  python report/build_mg_report.py && cd report && jupyter nbconvert --to notebook --execute --inplace Contact_Segment_Study.ipynb
"""
import nbformat as nbf
from nbformat.v4 import new_markdown_cell as md, new_code_cell as code

C = []
C.append(md(r"""# Learning only the contact-rich phase vs whole trajectories
### Does a policy that learns only the final contact-rich manipulation, with a VLM + planned reach getting there, generalize better than a whole-trajectory policy trained on the same demos?

**Status (2026-10-31).** Phase 0 is complete: the environment, the datasets, contact-phase segmentation, and a DROID-style segmenter validated against ground truth. The training grid for the first seed has run, and the whole-trajectory results are final. The composed-method evaluations (our policy and controls C1/C2) have been **withheld**: the harness that executes them is still being debugged (§5). The plan with all design decisions is in `docs/PLAN_contact_segment_vs_whole.md`."""))
C.append(code(r"""import json, os, glob, math, collections
import numpy as np
from IPython.display import display, Markdown, Image, Video
ROOT = os.path.abspath('..'); RUNS = f'{ROOT}/runs/mimicgen'; DEMO = f'{ROOT}/demo/mimicgen'
def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),) * 2
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d; return c - h, c + h
def fmt(R):
    if not R: return '–'
    k, n = sum(r['success'] for r in R), len(R); lo, hi = wilson(k, n); return f"{100*k/n:.0f}% [{100*lo:.0f}–{100*hi:.0f}]"
def table(rows, header):
    out = '| ' + ' | '.join(header) + ' |\n|' + '---|' * len(header) + '\n'
    for r in rows: out += '| ' + ' | '.join(str(x) for x in r) + ' |\n'
    display(Markdown(out))
def load(p): return json.load(open(p)) if os.path.exists(p) else []
TASKS = ['square', 'threading', 'coffee', 'three_piece_assembly', 'mug_cleanup']"""))

C.append(md(r"""---
## 1. Question and positioning

**Hypotheses.**
* **H1:** a contact-segment policy with egocentric inputs, composed with VLM-guided reaching, generalizes better than a whole-trajectory policy trained on the same demos with the same network and budget.
* **H2:** in distribution they are on par.
* **H3:** the gain comes from modelling only the contact phase's variance, not merely from starting near the object. Control C1 (the whole-trajectory policy started at the same handoff) tests this.

**Closest prior work.**
* **HITL-TAMP** (Mandlekar et al., CoRL 2023): TAMP handles free space, and people teleoperate only contact-rich segments. Policies trained on those segments beat full-demo training.
* **SkillMimicGen** (Garrett et al., 2024): learned contact-rich skills, connected by motion planning.

Both assume known object models and poses, and both report mainly in-distribution success and data efficiency.

**What is new here:**
1. open-world VLM grounding from RGB-D + language instead of TAMP's object knowledge;
2. systematic generalization along standard shift axes;
3. segment-only vs whole-trajectory fine-tuning of pretrained VLAs, up to real DROID data;
4. the C1 control and egocentric inputs, which explain *why* any gain appears."""))

C.append(md(r"""---
## 2. Setup: MimicGen, a standard contact-rich benchmark with built-in generalization splits

| | |
|---|---|
| Tasks | Square (nut insertion), Threading (needle), Coffee (pod insertion + lid), Three-Piece Assembly, Mug Cleanup (drawer + mug) |
| Data | 1000 MimicGen demos per task and variant; 84 px external + wrist images; OSC_POSE delta actions at 20 Hz |
| Generalization | train on **D0**, test on **D0 / D1 / D2**, MimicGen's official reset distributions (D1/D2 place objects over much wider regions) |
| Protocol | MimicGen/robomimic success checks and horizons (400 steps; 500 for Three-Piece Assembly and Mug Cleanup); 100 seeded episodes per variant |
| Stack | robosuite at MimicGen's commit, **MuJoCo 2.3.2** (2.3.7 changes the arm dynamics relative to the data), robomimic d0b37cf; own env on /scratch |
| Network | the LIBERO study's diffusion policy (ResNet18-GN + spatial softmax, 1-D UNet, 16-step chunks, 8 executed), crop 76 px, 100k steps, batch 128, for every policy |

**Policies:**
| | Trained on | Inputs | At test time |
|---|---|---|---|
| **P1 whole-trajectory** | all frames | external + wrist, absolute proprio | runs from reset |
| **P2 contact-segment (ours)** | contact phases only, from 10 steps before first contact | wrist only, position relative to the handoff, subtask id | reach to each subtask's handoff, then P2 |
| **C1** | (= P1) | as P1 | placed at the same handoffs as P2 |
| **C2** | contact phases | external + wrist, absolute proprio | as P2 |"""))

C.append(md(r"""---
## 3. Contact-phase segmentation

Each task is a chain of MimicGen's own object-centric subtasks; for example, Square is grasp the nut, then insert it on the peg.
* Every demo is replayed from its recorded MuJoCo states. Inside each subtask, the **contact phase** starts at the first physical contact between the gripper, or the object it holds, and the subtask's reference object.
* The learned segment starts 10 steps before that contact (the **handoff**) and ends with the subtask.
* Everything before the handoff is free-space motion, done at test time by the VLM-grounded reach.

Below: one demo per task. Blue frames are handled by the VLM and the reach; orange frames are the learned contact phases."""))
C.append(code(r"""for t in ['square', 'coffee', 'threading']:
    p = f'{DEMO}/contact_phases_{t}.png'
    if os.path.exists(p): display(Markdown(f'**{t}**')); display(Image(p, width=1100))"""))
C.append(code(r"""ST = load(f'{RUNS}/segment_stats.json')
rows = []
for name, st in ST.items():
    for s in st['subtasks']:
        rows.append([name, f"{s['ref']} ({s['signal'] or 'final'})", f"{s['contact_found']}/{st['n_demos']}", f"{s['contact_to_end_median']:.0f}", f"{s['subtask_len_median']:.0f}"])
    rows.append([name, '**share of frames in learned segments**', '', f"{100*st['segment_frame_share_pre10']:.0f}%", ''])
table(rows, ['Dataset', 'Subtask (reference object)', 'Contact found', 'Contact → end (median steps)', 'Whole subtask (median steps)'])"""))
C.append(md(r"""**A note on the data.** MimicGen's core datasets contain only successful demos. Replaying their stored states does not always reproduce the success check (MimicGen post-processed the datasets for robosuite 1.4), so we keep every demo in which all contacts were found. An early version filtered on the replay check and dropped 214 Mug Cleanup demos; those segment trainings were re-run. The first Square and Coffee segment trainings excluded 2 and 5 demos (< 0.5%)."""))

C.append(md(r"""---
## 4. Can contact be found without simulator contact data? (DROID-style segmentation)

DROID provides only end-effector poses and gripper commands. A heuristic segmenter (`vrg/mg/heuristic_segment.py`) estimates contact onsets from those signals alone:
* a close command marks a grasp;
* the slow-down after transporting a held object marks a placement or insertion;
* a slow-down before any grasp marks a push.

Here it is scored against the ground truth of §3."""))
C.append(code(r"""rows = []
for name, st in ST.items():
    for k, h in st.get('heuristic', {}).items():
        rows.append([name, k, f"{100*h['found']:.0f}%", f"{h['median_err']:+.0f}", f"{100*h['within10']:.0f}%"])
table(rows, ['Dataset', 'Contact', 'Detected', 'Median error (steps)', 'Within ±10 steps'])"""))
C.append(md(r"""**Reading.**
* **Grasps** are found reliably: within ±10 steps in 86–100% of demos, with a fixed bias of about 5–6 steps (the close command precedes finger contact), which can be subtracted. The exception is the mug (30%), where the fingers touch it about 19 steps before the close command.
* **Placements, insertions and pushes** are uneven: Square's insertion is caught within ±10 steps in 90% of demos, the Three-Piece Assembly inserts in about 50%, the drawer push in 60%, and Coffee's and Threading's insertions in only 12–16%, because their final approaches are long and slow. **The DROID tier needs a better contact detector** (learned from these sim labels, or VLM-based on the wrist video). These sim labels are the benchmark for building it."""))

C.append(md(r"""---
## 5. Results so far

### 5a. Whole-trajectory policies (P1): strong in distribution, near zero on wider placements"""))
C.append(code(r"""rows = []
for t in TASKS:
    rows.append([t] + [fmt(load(f'{RUNS}/{t}_whole_s0/eval_whole_d{v}/results.json')) for v in (0, 1, 2)])
table(rows, ['Task (P1, seed 0)', 'D0 (training distribution)', 'D1', 'D2'])
display(Markdown('Mug Cleanup has no D2 variant. Brackets: Wilson 95% CIs over 100 episodes.'))"""))
C.append(md(r"""**Reading.** Every whole-trajectory policy masters its training distribution (66–90%) and fails almost completely once objects are placed more widely (0–11%). This is the generalization gap the study targets, the same pattern as LIBERO-Object's layout shuffle (98.5% → 0%). Rule-selected clips (episode 0 of each variant):"""))
C.append(code(r"""for t in ['square', 'coffee', 'threading']:
    p = f'{DEMO}/whole_d0_vs_d1_{t}.mp4'
    if os.path.exists(p): display(Markdown(f'**{t}**: whole-trajectory policy on D0 and D1')); display(Video(p, embed=True, width=700, html_attributes='controls loop muted'))"""))

C.append(md(r"""### 5b. Composed evaluations (P2, C1, C2): withheld while the harness is debugged

In the composed method, a scripted reach moves to each subtask's handoff and the policy runs until MimicGen's subtask signal fires. Debugging it so far has found and fixed four bugs:
1. **Image orientation:** raw robosuite frames are upside down, while the datasets (recorded through robomimic) are flipped. Fixed, and verified pixel-exact against the dataset.
2. **End-effector frame:** MimicGen's eef frame and robosuite's `robot0_eef_quat` differ by a constant **90°**, so every handoff arrived rotated. Fixed: the reach servoes in MimicGen's frame.
3. **Transit height:** lifting above the start height put the far side of the table out of reach. Fixed: the transit height is set relative to the goal.
4. **Insertion handoff:** a hand-relative handoff varies by 6–7 cm across demos, because each demo grips the nut differently. Changed to a held-object-relative handoff (4.3 cm spread).

**Still open.** The C1 sanity check, the whole-trajectory policy started at its own handoff on D0, reaches only 13–30% on Square, against 90% from reset. Some reaches still end far from their goal. Until C1 approaches P1 in distribution, no composed number is reported. The first grid's composed numbers (near 0% for every policy) were invalid for reasons 1–2."""))

C.append(md(r"""---
## 6. Against the field: LIBERO-Plus (released VLA checkpoints vs our system)

**Benchmark.** LIBERO-Plus (Fei et al., 2025) perturbs the LIBERO suites along 7 dimensions: object layout, camera viewpoint, robot initial state, language, lighting, background texture and sensor noise, each at several difficulty levels. We use its **LIBERO-Object** suite (2,518 perturbed tasks) with the benchmark's own protocol: one trial per task from initial state 0, the benchmark's instruction string, and 280 steps after a 10-step settle.

**Methods.**
* **Released checkpoints, evaluated as is** (no training on our side):
  * π0.5-, π0- and π0-FAST-LIBERO via openpi, with a client identical to openpi's LIBERO example.
  * SmolVLA-LIBERO (`HuggingFaceVLA/smolvla_libero`, 0.45 B) and MolmoAct2-LIBERO (`allenai/MolmoAct2-LIBERO-LeRobot`) via LeRobot. A small server (`vrg/lplus/lerobot_server.py`) replays lerobot-eval's exact observation and processor chain. MolmoAct2 uses its recommended 50-step settle.
  * π0-FAST-LIBERO is **excluded**. On our GPUs (Blackwell, JAX 0.5.3) its server returns malformed action tokens ("Error decoding tokens"), and it scores 0/5 even on standard LIBERO-Object scenes (published ≈ 96%). π0 and π0.5 run correctly through the same client.
  * Check of the client on the 10 standard LIBERO-Object tasks (2 initial states each, `vrg/lplus/std_check.py`): SmolVLA 17/20 (published ≈ 90%), MolmoAct2 20/20.
* **Ours:** the LIBERO-Object v5 system (VLM handoff + egocentric local grasp + scripted transport), with grounding recomputed on each perturbed scene and the standard-scene calibration (strict zero-shot)."""))
C.append(code(r"""LP = f'{ROOT}/runs/liberoplus'
LPM = [('π0.5-LIBERO (openpi)', 'pi05_libero'), ('π0-LIBERO (openpi)', 'pi0_libero'), ('SmolVLA-LIBERO (LeRobot)', 'smolvla_libero'), ('MolmoAct2-LIBERO (LeRobot)', 'molmoact2_libero'), ('Ours (VLM + reach + local grasp)', 'ours')]
CATS = ['Objects Layout', 'Camera Viewpoints', 'Robot Initial States', 'Language Instructions', 'Light Conditions', 'Background Textures', 'Sensor Noise']
rows = []
for lab, name in LPM:
    R = load(f'{LP}/{name}/results.json')
    if not R: rows.append([lab, 'pending'] + [''] * len(CATS)); continue
    by = collections.defaultdict(list)
    for r in R: by[r.get('category')].append(r)
    rows.append([lab, f"{fmt(R)} (n={len(R)})"] + [f"{100*sum(r['success'] for r in by[c])/len(by[c]):.0f}%" if by[c] else '–' for c in CATS])
table(rows, ['Method', 'All perturbations'] + CATS)
display(Markdown('Published LIBERO-Plus totals over all four suites, for reference: OpenVLA-OFT 69.6, π0 53.6, π0-FAST 61.6, OpenVLA 15.6.'))"""))
C.append(code(r"""rows = []
for lab, name in LPM:
    R = load(f'{LP}/{name}/results.json')
    if not R: continue
    by = collections.defaultdict(list)
    for r in R: by[r.get('difficulty')].append(r)
    rows.append([lab] + [f"{100*sum(r['success'] for r in by[d])/len(by[d]):.0f}% (n={len(by[d])})" if by[d] else '–' for d in (1, 2, 3, 4, 5)])
if rows: table(rows, ['Method', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'])"""))
C.append(md(r"""**Where our system fails.** Each failure is split by stage: grasp, or place (the scripted transport and release), next to the grounding quality on the same scenes. The target hit rate is whether the VLM picked the right object. The basket error is the distance between the grounded and true basket centres in the table plane. On standard scenes it is ≈ 7 cm, a fixed offset that the large basket absorbs."""))
C.append(code(r"""R = load(f'{LP}/ours/results.json'); G = {g['tid']: g for g in load(f'{LP}/ours/grounding.json') or []}
rows = []
for c in CATS:
    rs = [r for r in R if r.get('category') == c]
    if not rs: continue
    be = [G[r['tid']]['basket_err'] for r in rs if G.get(r['tid'], {}).get('basket_err') is not None]
    rows.append([c, len(rs), f"{100*np.mean([r['success'] for r in rs]):.0f}%", sum(r.get('fail') == 'grasp' for r in rs), sum(r.get('fail') == 'place' for r in rs),
                 f"{100*np.mean([G.get(r['tid'], {}).get('obj_hit', False) for r in rs]):.0f}%", f"{100*np.median(be):.1f} cm" if be else '–', f"{100*np.mean(np.array(be) > 0.12):.0f}%" if be else '–'])
table(rows, ['Category', 'n', 'Success', 'Grasp fails', 'Place fails', 'Target hit', 'Basket err (median)', 'Basket err > 12 cm'])"""))
C.append(md(r"""**Reading (ours, final over all 2,518 tasks: 69.9%; π0.5-LIBERO 89.1%).**
* **Where the modular design wins:**
  * **Robot initial states:** 98% vs π0.5's 69%. The scripted reach makes the start pose irrelevant: the same mechanism behind the layout-shuffle result.
  * **Sensor noise:** 99% vs 95%.
* **Where it loses is appearance, not geometry:**
  * **Light conditions:** 1% vs 99%. The VLM finds the target in every scene, but 282/297 episodes fail at the grasp.
  * **Background textures:** 19% vs 98%, with 115 grasp failures.
  * The cause: the learned local skill sees only the wrist camera, was trained on 50 demos per task without colour or lighting augmentation, and does not transfer to a relit or retextured table.
  * The VLAs are pretrained on large, visually diverse data.
* **The basket grounding also degrades under the same perturbations:** median error 23–25 cm vs ≈ 7 cm elsewhere. So the depth-cluster proposals need to be robust to appearance too.
* **Camera viewpoints:** 66% vs 87%. The target is found in 84% of scenes, and most failures are at the place step.
* **For the contact-segment study:** the local skill must be visually robust, not only spatially local. That means photometric augmentation or a pretrained visual encoder in the segment policy. This is now a planned ablation, and it is not tuned on these test tasks."""))

C.append(md(r"""---
## 7. Next
1. Fix the transfer reach (video step-through; MimicGen-style interpolated transit). Validate with C1 ≈ P1 on D0, then re-run the composed grid, plus seeds 1–2, the data-efficiency (10 / 100 / 1000 demos) and handoff-length sweeps, and handoff noise.
2. Re-run the two Three-Piece Assembly segment trainings (they ran out of GPU memory).
3. Against the field: released LIBERO VLA checkpoints vs our system on LIBERO-Plus (2,518 perturbed LIBERO-Object episodes); environment in progress.
4. DROID tier: a contact detector trained on these sim labels, then segment vs whole fine-tuning of a small VLA, evaluated in PolaRiS / sim-evals scenes, then the real Franka."""))

nb = nbf.v4.new_notebook(); nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "report/Contact_Segment_Study.ipynb"); print("wrote report/Contact_Segment_Study.ipynb with", len(C), "cells")
