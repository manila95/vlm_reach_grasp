"""Section 6 of the report: RoboCasa365 and Colosseum (imported by build_report.py).

Reads runs/eval/rc_*/, runs/eval/col_*/, runs/eval/{rc,col}_grounding.json, runs/eval/colosseum_paper_tables.json and
demo/bench/*.mp4, all written by `python -m vrg.bench.collect_results`.
"""
from nbformat.v4 import new_markdown_cell as md, new_code_cell as code


def cells():
    c = []
    c.append(md(r"""---
## 6. Beyond LIBERO: RoboCasa365 and Colosseum

LIBERO-Object is a single tabletop with ten upright objects. Two harder simulators test whether the decomposition carries over:

* **RoboCasa365** (MuJoCo, mobile manipulator PandaOmron in full kitchens). The 5 official pick-and-place tasks of the *Atomic-Seen* set: counter → cabinet, counter → stove, drawer → counter, sink → counter, toaster → counter. All policies train on the same **100 demos per task** from the *pretrain* split. They are evaluated on 50 episodes per task in two settings:
  * **pretrain kitchens**: the layouts and object categories seen in training;
  * **target kitchens**: unseen kitchens and unseen object instances (the benchmark's OOD split).
* **Colosseum** (CoppeliaSim/RLBench). 12 of the 20 tasks, each under 9 test sets: no perturbation, and one perturbation each of manipulated-object colour, manipulated-object size, light colour, table colour, table texture, distractors, background texture and camera pose. All policies train on the 100 unperturbed demos per task. They are evaluated on 10 of the 25 saved test episodes per (task, perturbation), 1,080 episodes per method. Actions are keyframes executed by RLBench's motion planner. As in the benchmark's own evaluation, an episode ends at the first infeasible keyframe (`InvalidActionError`).

**How each method was adapted**, with one shared diffusion-policy architecture and training budget:

| | RoboCasa | Colosseum |
|---|---|---|
| **Ours** | The VLM grounds the object and the receptacle from the robot's cameras. A scripted controller drives the arm (and base) to a handoff pose. A local **pick** skill (wrist camera + egocentric state) grasps and lifts. A scripted transport is followed by a local **place** skill. | The VLM grounds the first-contact point. The planner goes to the handoff. An egocentric keyframe skill (wrist camera, goal vector in the gripper frame) does the rest. |
| **End-to-end** | 3 cameras + state + CLIP-embedded instruction → actions, full demos | front + wrist cameras + gripper pose + instruction → next keyframes |
| **SayCan-style** | The VLM picks between language skills ("pick up the X", "place it in Y"), each a language-conditioned policy | not run: the Colosseum tasks are single-skill, so it reduces to end-to-end |
| **HAMSTER-style** | VLM-grounded 2D path drawn onto the left camera, path-conditioned policy | same, on the front camera |

**Protocol notes.**
* Ours was developed on separate RoboCasa episodes (pretrain eps 50–59). The RoboCasa episodes used for testing are eps 0–49.
* A few test episodes were touched while debugging, before the dev set was set up:
  * RoboCasa: cabinet/stove eps 0–3 and drawer/sink/toaster eps 0–1.
  * Colosseum, no-perturbation test set: move_hanger and close_box eps 0–2, wine rack eps 0–1, basketball ep 0.

  That is under 1.5% of the evaluation episodes, and no hyper-parameter was chosen on them.
* One design choice was made on the dev set: a pick skill trained on longer slices (handoff 40 steps before the grasp instead of 20). It tied the original at 11/50, so the original was kept.
* Every training run is a single seed."""))

    c.append(code(r"""# ---- benchmark helpers ----------------------------------------------------------------------
import collections
BM = {'ours': ('#2a78d6', 'Ours: VLM handoff + local skills'), 'e2e': ('#eb6834', 'End-to-end diffusion'),
      'saycan': ('#1baf7a', 'SayCan-style'), 'hamster': ('#eda100', 'HAMSTER-style')}
def rows(name):
    r = load(name); return r or []
def rate_rows(R):
    k, n = sum(bool(x['success']) for x in R), len(R); lo, hi = wilson(k, n) if n else (np.nan, np.nan)
    return dict(sr=k / n if n else np.nan, lo=lo, hi=hi, n=n, k=k)
def fmt_rows(R):
    s = rate_rows(R)
    return 'pending' if s['n'] == 0 else f"{100*s['sr']:.1f}% [{100*s['lo']:.0f}–{100*s['hi']:.0f}] (n={s['n']})"
RC_T = ['PickPlaceCounterToCabinet', 'PickPlaceCounterToStove', 'PickPlaceDrawerToCounter', 'PickPlaceSinkToCounter', 'PickPlaceToasterToCounter']
RC_SHORT = dict(zip(RC_T, ['counter → cabinet', 'counter → stove', 'drawer → counter', 'sink → counter', 'toaster → counter']))
print('loaded:', {f'rc_{m}_{s}': len(rows(f'rc_{m}_{s}')) for m in BM for s in ('pretrain', 'target')}, {f'col_{m}': len(rows(f'col_{m}')) for m in ('ours', 'e2e', 'hamster')})"""))

    # ---------------------------------------------------------------- RoboCasa
    c.append(md(r"""### 6.1 RoboCasa365: success in seen and unseen kitchens

Blue = pretrain kitchens (seen layouts and object categories), orange = target kitchens (unseen kitchens and objects); thin bars are 95% Wilson CIs. 250 episodes per dot."""))
    c.append(code(r"""fig, ax = plt.subplots(figsize=(10, 3.2))
ML = ['ours', 'e2e', 'saycan', 'hamster']
for i, m in enumerate(ML[::-1]):
    a, b = rate_rows(rows(f'rc_{m}_pretrain')), rate_rows(rows(f'rc_{m}_target'))
    if a['n'] and b['n']: ax.plot([a['sr'], b['sr']], [i, i], color='#c3c2b7', lw=2, zorder=1)
    for r, col, dy in ((a, '#2a78d6', 0.12), (b, '#eb6834', -0.12)):
        if not r['n']: continue
        ax.errorbar(r['sr'], i + dy, xerr=[[r['sr'] - r['lo']], [r['hi'] - r['sr']]], fmt='none', ecolor=col, elinewidth=1, alpha=0.6)
        ax.scatter(r['sr'], i + dy, s=70, color=col, edgecolors='white', linewidths=2, zorder=3)
        ax.text(r['sr'], i + dy + 0.22 * np.sign(dy), f"{100*r['sr']:.0f}", ha='center', va='center', fontsize=8, color=INK)
ax.set_yticks(range(len(ML))); ax.set_yticklabels([BM[m][1] for m in ML[::-1]]); ax.set_ylim(-0.6, len(ML) - 0.4)
ax.set_xlim(-0.02, 0.6); ax.set_xticks([0, .1, .2, .3, .4, .5]); ax.set_xticklabels(['0', '10%', '20%', '30%', '40%', '50%']); ax.grid(axis='y', visible=False)
ax.scatter([], [], s=60, color='#2a78d6', label='pretrain kitchens'); ax.scatter([], [], s=60, color='#eb6834', label='target kitchens (unseen)')
ax.legend(loc='lower center', bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=9); ax.set_xlabel('success rate (5 pick-and-place tasks × 50 episodes)')
plt.show()
table([[BM[m][1], fmt_rows(rows(f'rc_{m}_pretrain')), fmt_rows(rows(f'rc_{m}_target'))] for m in ML], ['Method', 'Pretrain kitchens', 'Target kitchens (OOD)'])"""))
    c.append(code(r"""# Per task
hdr = ['Task'] + [f'{BM[m][1].split(":")[0]} ({s[:3]})' for m in ML for s in ('pretrain', 'target')]
out = []
for t in RC_T:
    row = [RC_SHORT[t]]
    for m in ML:
        for s in ('pretrain', 'target'):
            R = [x for x in rows(f'rc_{m}_{s}') if x['task'] == t]
            row.append(f"{sum(x['success'] for x in R)}/{len(R)}" if R else '–')
    out.append(row)
table(out, hdr)"""))
    c.append(md(r"""**Reading RoboCasa.**
* **Everyone is far from solving it at 500 demos.** The best method succeeds in under a fifth of episodes. The published diffusion-policy baseline with 60× more data averages 15.7% on the 18 Atomic-Seen tasks, so these are realistic numbers for a small-data diffusion policy in RoboCasa.
* **In seen kitchens ours is the best method, but not by a margin that excludes HAMSTER-style.** Ours (17%) beats end-to-end (10%) and SayCan-style (7%) with non-overlapping or barely overlapping CIs; HAMSTER-style (16%) is within noise.
* **In unseen kitchens the advantage does not survive.** Ours (11%), HAMSTER-style (12%) and end-to-end (10%) are statistically tied. This is not the LIBERO-shuffle story: there, the baselines failed because they went to memorized *locations*. RoboCasa kitchens vary layout in both splits, so no baseline can rely on memorized positions even in the pretrain split. The target split instead changes kitchen styles and object instances, which hurts every learned visual component, ours included.
* **The VLM part transfers; the learned grasp does not fully.** The pick point hits the right object in 93% (pretrain) and 90% (target) of episodes, but ours loses most episodes at the *grasp* (§6.2).
* **The decomposition pays off where the search is hardest.** Counter → stove (find the object, then a pan on a cluttered stove top) is 15/50 and 10/50 for ours and at most 2/50 for any baseline. Sink and drawer → counter, where the object sits inside a fixture the wrist camera sees poorly, are where ours is weakest."""))
    c.append(md(r"""**Published reference points.** These are not like-for-like: the published policies train on far more data (Human300: 100 demos × 300 tasks, or a GR00T N1.5 foundation model plus target-split fine-tuning). Ours and the baselines here see only 500 demos.

| Policy (source) | Training data | Eval split | Avg. over these 5 tasks |
|---|---|---|---|
| Diffusion Policy (RoboCasa365 paper, multi-task) | Human300 | pretrain | Atomic-Seen avg. 15.7% (18 tasks; per-task not published) |
| GR00T N1.5, public checkpoint (arXiv 2606.21572, Table 3b) | foundation model + Human300 | pretrain | 64.8% (76.0 / 69.6 / 11.6 / 88.8 / 78.0) |
| GR00T N1.5, pretrain + 100% target data (paper Table 11) | foundation model + target demos | target | 65.2% (63 / 70 / 50 / 70 / 73; 30 trials) |

Read at the scale of a 500-demo budget, the relevant comparison is with the in-house baselines, which share data, architecture and steps."""))
    c.append(md(r"""### 6.2 RoboCasa: where ours breaks

For every failed episode of ours, the rollout records the **first phase that broke**:
* **grounding**: the VLM returned no pick point;
* **reach**: the handoff pose was not reached within 5 cm;
* **grasp**: the object was not held after the pick skill;
* **transport**: the object was dropped before the place skill;
* **place**: the object was released but is not in the receptacle, or the robot did not retreat.

Grasp checks use privileged sim state and are analysis only. The VLM column is the share of episodes where the 3-D pick point lands on the right object (within 5 cm of it)."""))
    c.append(code(r"""GR = json.load(open(f'{EVAL}/rc_grounding.json')) if os.path.exists(f'{EVAL}/rc_grounding.json') else []
PH = ['grounding', 'reach', 'grasp', 'transport', 'place', 'timeout']
out = []
for s in ('pretrain', 'target'):
    R = rows(f'rc_ours_{s}')
    for t in RC_T + [None]:
        Rt = [x for x in R if t is None or x['task'] == t]
        g = [x for x in GR if x['split'] == s and (t is None or x['task'] == t)]
        if not Rt: continue
        cnt = collections.Counter(x['fail'] for x in Rt if not x['success'])
        out.append([s, RC_SHORT[t] if t else '**all**', f"{100*np.mean([x['hit'] for x in g]):.0f}%" if g else '–',
                    f"{sum(x['success'] for x in Rt)}/{len(Rt)}"] + [cnt.get(p, 0) for p in PH] + [sum(v for k, v in cnt.items() if k not in PH)])
table(out, ['split', 'task', 'VLM pick hit', 'success'] + [f'fail: {p}' for p in PH] + ['other'])"""))
    c.append(code(r"""# Stacked bars: first broken phase, ours, both splits
fig, axs = plt.subplots(1, 2, figsize=(11, 2.8), sharey=True)
PC = {'success': '#2a78d6', 'grounding': '#8a8984', 'reach': '#c3c2b7', 'grasp': '#eb6834', 'transport': '#eda100', 'place': '#1baf7a', 'other': '#52514e'}
for ax, s in zip(axs, ('pretrain', 'target')):
    R = rows(f'rc_ours_{s}')
    for j, t in enumerate(RC_T):
        Rt = [x for x in R if x['task'] == t]
        if not Rt: continue
        cnt = collections.Counter('success' if x['success'] else (x['fail'] if x['fail'] in PC else 'other') for x in Rt)
        left = 0
        for k, col in PC.items():
            v = cnt.get(k, 0) / len(Rt); ax.barh(j, v, left=left, color=col, height=0.7, label=k if j == 0 else None); left += v
    ax.set_title(f'{s} kitchens', fontsize=10); ax.set_xlim(0, 1); ax.set_yticks(range(len(RC_T))); ax.set_yticklabels([RC_SHORT[t] for t in RC_T])
    ax.grid(axis='y', visible=False); ax.invert_yaxis()
axs[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8); axs[0].set_xlabel('share of episodes'); axs[1].set_xlabel('share of episodes')
plt.tight_layout(); plt.show()"""))
    c.append(md(r"""**Diagnosis on the dev set (pretrain eps 50–59, not the evaluation episodes).**
* **The pick skill itself works.** Started from the demos' own handoff pose, it grasps 8/10 on both stove and cabinet scenes.
* **The handoff is the problem.** Started from our handoff (the VLM point plus the mean demo offset and orientation), the same skill grasps only 3–4/10.
* **Both parts of the handoff matter.** Swapping in each demo's own orientation and position brings it back to 7–8/10, and each part contributes.
* **Two other fixes did not help.** Per-category mean offsets did not. Neither did training the skill from further back (a wider basin of attraction), which tied at 11/50.

In RoboCasa the objects vary much more in shape and size than in LIBERO, so one mean approach pose per task is the bottleneck. The natural fix is the one listed in §7: let the VLM, or a small grasp-pose model, predict the approach orientation instead of using the demo mean."""))

    # ---------------------------------------------------------------- Colosseum
    c.append(md(r"""### 6.3 Colosseum: robustness to visual perturbations

Success per perturbation, averaged over the 12 tasks (10 episodes per task and perturbation; bars are 95% Wilson CIs over the pooled episodes)."""))
    c.append(code(r"""FN = {0: 'none', 2: 'object colour', 6: 'object size', 8: 'light colour', 9: 'table colour', 10: 'table texture',
      11: 'distractors', 12: 'background', 14: 'camera pose'}
CM = ['ours', 'e2e', 'hamster']
fig, ax = plt.subplots(figsize=(11, 3.4)); w = 0.27
for j, m in enumerate(CM):
    R = rows(f'col_{m}')
    st = [rate_rows([x for x in R if x['factor'] == f]) for f in FN]
    xs = np.arange(len(FN)) + (j - 1) * w; v = np.array([s['sr'] for s in st])
    ax.bar(xs, v, w * 0.92, color=BM[m][0], label=f"{BM[m][1]} ({fmt_rows(R).split(' (')[0]})")
    ax.errorbar(xs, v, yerr=[v - np.array([s['lo'] for s in st]), np.array([s['hi'] for s in st]) - v], fmt='none', ecolor=INK2, elinewidth=0.8)
ax.set_xticks(range(len(FN))); ax.set_xticklabels(list(FN.values()), rotation=20); ax.set_ylabel('success'); ax.set_ylim(0, 0.75)
ax.legend(fontsize=8, loc='upper right'); ax.grid(axis='x', visible=False); plt.show()
table([[BM[m][1], fmt_rows(rows(f'col_{m}')), fmt_rows([x for x in rows(f'col_{m}') if x['factor'] == 0]),
        fmt_rows([x for x in rows(f'col_{m}') if x['factor'] != 0])] for m in CM],
      ['Method', 'All 9 test sets', 'No perturbation', '8 perturbations'])"""))
    c.append(md(r"""**Per task, next to grounding quality.** *VLM found* is the share of episodes where the VLM returned a first-contact point. *Handoff error* is the median distance between the VLM handoff and the demonstrations' first contact keyframe. Published RVT and PerAct numbers (Colosseum paper, no perturbation, 25 episodes, models trained on all tasks) are shown for scale."""))
    c.append(code(r"""CG = json.load(open(f'{EVAL}/col_grounding.json')) if os.path.exists(f'{EVAL}/col_grounding.json') else []
PT = json.load(open(f'{EVAL}/colosseum_paper_tables.json'))
out = []
for t in sorted({x['task'] for x in rows('col_e2e') + rows('col_ours')}):
    g = [x for x in CG if x['task'] == t]; e = [x['err'] for x in g if x['err'] is not None]
    r = [f"{sum(x['success'] for x in R)}/{len(R)}" if R else '–' for R in ([x for x in rows(f'col_{m}') if x['task'] == t] for m in CM)]
    out.append([t, f"{100*len(e)/len(g):.0f}%" if g else '–', f"{100*np.median(e):.0f} cm" if e else '–', *r,
                PT['RVT'][1].get(t, ['–'])[0] + '%', PT['PerAct'][1].get(t, ['–'])[0] + '%'])
table(out, ['task', 'VLM found', 'handoff error (median)', 'Ours', 'End-to-end', 'HAMSTER-style', 'RVT (paper, no pert.)', 'PerAct (paper, no pert.)'])
R = rows('col_ours'); f = [x for x in R if not x['success']]
cnt = collections.Counter('no grounding' if x['error'] == 'grounding' else 'planner: handoff infeasible' if x['error'] == 'InvalidActionError' and x['keyframes'] <= 1
                          else 'planner: later keyframe infeasible' if x['error'] == 'InvalidActionError' else 'ran out of keyframes' for x in f)
print('ours, failed episodes by cause:', dict(cnt.most_common()))"""))
    c.append(md(r"""**Reading Colosseum.**
* **End-to-end is clearly ahead overall** (about 25% vs 9%), and ahead without perturbation (50% vs 15%).
* **Ours is only as good as its handoff point.** Ours beats end-to-end only on close_box (handoff ≈ 9 cm off, but pushing a lid shut tolerates that) and ties on move_hanger (handoff ≈ 2 cm off). On most other tasks the handoff is 8–50 cm from the demos' first contact. turn_oven_on is never grounded: the knob is too small for depth proposals. From there the egocentric skill starts outside anything it saw in training, its keyframes drift, and the planner rejects them; that is the largest failure cause.
* **Basketball shows that grounding is necessary but not sufficient.** When the ball is found, the handoff is accurate (≈1 cm), yet ours scores 0/90: the ball is found in only 40% of episodes, and the keyframe skill fails the throw from the handoff even when it is.
* **Colosseum tests a different kind of target.** On LIBERO the hard part was *which object, where*, and the VLM solved it. On Colosseum's articulated and tool-use tasks the contact point is a *part* (a lid edge, a drawer handle, a spatula handle, a knob). Our zero-shot point-and-verify grounding was built for whole objects, and part-level grounding is where it fails."""))
    c.append(code(r"""# 6.4 Perturbation robustness where the handoff works (tasks with median handoff error < 10 cm and the VLM finding the point in >= 50% of episodes)
good = []
for t in sorted({x['task'] for x in CG}):
    g = [x for x in CG if x['task'] == t]; e = [x['err'] for x in g if x['err'] is not None]
    if e and len(e) >= 0.5 * len(g) and np.median(e) < 0.10: good.append(t)
display(Markdown('### 6.4 Perturbation robustness on the tasks where grounding works\n\nTasks: ' + ', '.join(good)))
out = []
for m in CM:
    R = [x for x in rows(f'col_{m}') if x['task'] in good]
    if not R: continue
    a, b = rate_rows([x for x in R if x['factor'] == 0]), rate_rows([x for x in R if x['factor'] != 0])
    out.append([BM[m][1], fmt_rows([x for x in R if x['factor'] == 0]), fmt_rows([x for x in R if x['factor'] != 0]),
                f"{100*(b['sr'] - a['sr']):+.0f} pts" if a['n'] and b['n'] else '–'])
table(out, ['Method', 'No perturbation', '8 perturbations pooled', 'change'])"""))

    c.append(md(r"""**Robustness where the handoff works.** On these tasks, without perturbation, end-to-end is about twice as successful as ours. Under the 8 perturbations, end-to-end loses about 70% of its success, ours about half. End-to-end is still slightly ahead (≈26% vs ≈21%, overlapping CIs). The direction matches the LIBERO result: the egocentric skill sees only the wrist camera after the handoff, so table, background, lighting and camera changes reach it only through the wrist view. But with 4 tasks and 40 unperturbed episodes per method, this is suggestive, not conclusive.

*Note on HAMSTER-style episodes:* close_laptop_lid with the object-size perturbation occasionally fails to place the scene at reset. This crashed one HAMSTER-style worker. The rollout now retries the reset (and would skip an episode that never places, which did not happen), and that worker was resumed. All three methods have the same 1,070 episodes (move_hanger has no object-size test set)."""))

    # ---------------------------------------------------------------- videos
    c.append(md(r"""### 6.5 RoboCasa side by side

The same scene (evaluation episode 0) for each task, 2× speed. Top row: ours | end-to-end; bottom row: SayCan-style | HAMSTER-style. Each panel shows the left robot camera and the wrist camera, with the outcome in the banner. Episode 0 is fixed by rule, not chosen by outcome."""))
    c.append(code(r"""for s in ('pretrain', 'target'):
    for t in RC_T:
        show(f'{ROOT}/demo/bench/rc_{s}_{t[9:]}.mp4', w=760, caption=f'**{RC_SHORT[t]}, {s} kitchens**')"""))
    return c
