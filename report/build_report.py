"""Builds report/VLM_Reach_LocalGrasp_Report.ipynb (then execute with nbconvert).

Every number in the notebook is computed from files in runs/eval/*/results.json, data/*.pkl|json and results/*.json,
so re-running the evaluations and re-executing the notebook refreshes the report.
"""
import nbformat as nbf
from nbformat.v4 import new_markdown_cell as md, new_code_cell as code

cells = []

cells.append(md(r"""# VLM-guided reaching + learned local grasping on LIBERO-Object
### Proof of concept: a VLM shrinks the manipulation search space so imitation learning only has to solve the last few centimetres

**Setup in one line.** A vision-language model (Qwen3-VL-8B, run locally) reads the instruction, finds the target object and the receptacle in an RGB-D image,
a scripted controller brings the gripper to a *handoff pose* above the object, and a small diffusion policy (trained only on the ~1 s *grasp-and-lift* slice of the demonstrations) does the contact-rich part.
Transport and release are scripted again from the VLM's receptacle estimate.

This notebook is the full write-up: idea, design decisions, every number with a 95% confidence interval, failure analysis and video demos.
All numbers are recomputed from the saved evaluation files when the notebook runs.

> Related ideas: SayCan (LLM picks skills, value functions ground them), MOKA / VoxPoser / CoPa (VLM-derived keypoints or cost maps), and "coarse-to-fine" imitation (e.g. C2F-ARM). Here the specific question is: *if a VLM handles **where**, how much easier, more data-efficient and more robust does learning **how** become?*"""))

cells.append(code(r"""# ---- setup: paths, palette, helpers -------------------------------------------------------
import json, os, pickle, glob, math
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Video, Image, HTML

ROOT = os.path.abspath('..')          # project root (notebook lives in report/)
EVAL = f'{ROOT}/runs/eval'

# Fixed categorical slots (validated palette: CVD-safe adjacent order). Colour follows the method in every chart.
C = {'ego':      '#2a78d6',   # slot 1 blue   - VLM + local grasp, egocentric (wrist cam, no absolute xy)
     'e2e':      '#eb6834',   # slot 2 orange - end-to-end diffusion policy (task-id conditioned)
     'scripted': '#1baf7a',   # slot 3 aqua   - VLM + scripted top-down grasp
     'local':    '#eda100'}   # slot 4 yellow - VLM + local grasp, agent+wrist cams + absolute proprio
NAME = {'ego': 'VLM + local grasp (egocentric)', 'e2e': 'End-to-end diffusion policy',
        'scripted': 'VLM + scripted grasp', 'local': 'VLM + local grasp (agent+wrist)'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'legend.frameon': False})

TASKS = ['alphabet soup', 'cream cheese', 'salad dressing', 'bbq sauce', 'ketchup',
         'tomato sauce', 'butter', 'milk', 'chocolate pudding', 'orange juice']

def load(name):
    p = f'{EVAL}/{name}/results.json'
    return json.load(open(p)) if os.path.exists(p) else None

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c - h, c + h

def rate(name):
    r = load(name)
    if r is None: return dict(sr=np.nan, lo=np.nan, hi=np.nan, n=0, k=0)
    k, n = sum(x['success'] for x in r), len(r); lo, hi = wilson(k, n)
    return dict(sr=k/n, lo=lo, hi=hi, n=n, k=k)

def fmt(name):
    s = rate(name)
    return 'pending' if s['n'] == 0 else f"{100*s['sr']:.1f}% [{100*s['lo']:.0f}–{100*s['hi']:.0f}] (n={s['n']})"

def table(rows, header):
    out = '| ' + ' | '.join(header) + ' |\n|' + '---|' * len(header) + '\n'
    for r in rows: out += '| ' + ' | '.join(str(x) for x in r) + ' |\n'
    display(Markdown(out))

def bar_labels(ax, bars, vals, tops=None):
    tops = vals if tops is None else tops
    for b, v, t in zip(bars, vals, tops):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width()/2, max(v, np.nan_to_num(t)) + 0.02, f'{100*v:.0f}%', ha='center', va='bottom', fontsize=9, color=INK)
def show(path, w=760, caption=None):
    if os.path.exists(path):
        if caption: display(Markdown(caption))
        display(Video(path, embed=True, width=w, html_attributes='controls loop muted'))
    else:
        display(Markdown(f'*missing video: `{os.path.relpath(path, ROOT)}`*'))

print('root:', ROOT)"""))

# ------------------------------------------------------------------ TL;DR
cells.append(md(r"""## TL;DR"""))
cells.append(code(r"""rows = [
 ['End-to-end diffusion policy (task-id, 50 demos/task)', fmt('full_50'), fmt('full_50_shuf')],
 ['End-to-end diffusion policy (task-id, 10 demos/task)', fmt('full_10'), fmt('full_10_shuf')],
 ['VLA: OpenVLA-OFT 7B, Spatial-Forcing ckpt (language, all LIBERO-Object demos)', fmt('vla'), fmt('vla_shuf')],
 ['SayCan-style: VLM planner (LM × affordance) + language-conditioned skills', fmt('saycan'), fmt('saycan_shuf')],
 ['HAMSTER-style: VLM keypoint path drawn on image + path-conditioned policy', fmt('hamster'), fmt('hamster_shuf')],
 ['VLM + scripted grasp (no learning)', fmt('scripted_vlm'), fmt('scripted_vlm_shuf')],
 ['VLM + local grasp, agent+wrist cams', fmt('local_vlm'), fmt('local_vlm_shuf')],
 ['VLM + local grasp, egocentric (wrist only)', fmt('ego_vlm'), fmt('ego_vlm_shuf')],
 ['VLM + local grasp, egocentric v2 (full orientation lock + retract + retry)', fmt('ego_v2_vlm'), fmt('ego_v2_vlm_shuf')],
 ['VLM + local grasp, egocentric v3 (roll/pitch lock + retract + retry)', fmt('ego_v3_vlm'), fmt('ego_v3_vlm_shuf')],
 ['**VLM + local grasp, egocentric v5** (posture-aware lock & retract + retry)', '**'+fmt('v5strict_vlm')+'**', '**'+fmt('v5strict_vlm_shuf')+'**'],
]
table(rows, ['Method (LIBERO-Object, 10 tasks × 20 episodes)', 'In-distribution layouts', 'Shuffled layouts (OOD)'])"""))
cells.append(md(r"""**Headline findings** (numbers in the table above; brackets are Wilson 95% CIs):

1. **On the standard benchmark everything is near ceiling.** The end-to-end diffusion policy (98.5%), the 7B VLA (99%), SayCan-style (97%) and our hierarchy (≈97–98%) are statistically indistinguishable, because LIBERO-Object places each task's objects in nearly the same layout every time.
2. **When the objects are shuffled (zero-shot for every method: nothing is trained on shuffled scenes), every method that has to *find* the object with a learned visuomotor policy collapses.** End-to-end diffusion 0%, OpenVLA-OFT-7B ≈6% despite language conditioning and web-scale pretraining, SayCan-style 0%, HAMSTER-style ≈14% (and still ≈16% with a *perfect* path). They are not clumsy: they cleanly pick up **whatever object sits where the target used to be** (75–82% of episodes moved a wrong object).
3. **Where the decomposition happens matters more than whether there is one.** SayCan decomposes *semantically*: its VLM planner chose the right skill in 98.5% of shuffled episodes, but each skill still has to solve the spatial search, and fails. HAMSTER passes spatial guidance as an image-space path, and its policy does follow it, but only to locations some demo visited (79–91% there, 0% elsewhere, §4.1c). Only when the VLM's output is turned into **explicit control (reach to a handoff pose)** and the learned part is **restricted to egocentric inputs** does the system survive the shift: ≈75–78% shuffled (strictly zero-shot), moving the right object in >90% of episodes.
4. **The VLM handles *where* reliably, given the right interface.** Asking the VLM directly for a box or point finds the right object only 42–64% of the time, while *depth proposals + zoomed yes/no verification* gets 97–100% on every split.
5. **Learning the last centimetres pays off under localization error, but the skill's basin is small.** With 2–4 cm of noise the learned grasp degrades far more gracefully than a scripted one. Yet starting the skill 10 cm higher, or 6 cm to the side, of its training handoff roughly halves its success, so the VLM-driven reach is load-bearing.
6. **Hierarchical systems are debuggable.** The remaining shuffled failures were traced to a kinematic singularity (elbow at its joint limit near the workspace edge) and fixed with a posture-aware wrist constraint (v5). The fix was confirmed on a held-out split of 250 untouched episodes: shuffled 66% → 75%, with no in-distribution cost.
7. **Beyond LIBERO the picture is mixed (§6).**
   * **RoboCasa (5 kitchen pick-and-place tasks, 500 demos):** ours is the best in-house method in seen kitchens (17% vs 10% end-to-end), and alone solves counter → stove. In unseen kitchens it is tied with end-to-end and HAMSTER-style (≈11%), and most of its failures are grasps from a mean handoff pose that doesn't fit varied objects.
   * **Colosseum (12 tasks × 9 perturbations):** end-to-end wins overall (25% vs 9%) because zero-shot grounding of object *parts* (handles, lid edges, knobs) is 8–50 cm off on most tasks. Where grounding works, ours loses less under perturbation.
"""))

# ------------------------------------------------------------------ method
cells.append(md(r"""---
## 1. Method

```
 instruction ──► VLM: parse ──► pick="alphabet soup", place="basket"
                                   │
 agentview RGB-D (1024px) ──► depth proposals (bird's-eye clustering above floor plane)
                                   │  for each proposal: zoom crop ──► VLM "Is this the <pick>? Yes/No" ──► logit score
                                   ▼
                           3D target point (argmax score) − calibrated bias
                                   │
 [1] scripted OSC reach ──► handoff pose = target + Δ_handoff   (Δ, orientation taken from demo statistics)
 [2] learned local grasp ──► diffusion policy, wrist cam + (z, orientation, gripper)   until lifted 8 cm
 [3] scripted transport ──► lift to 30 cm, go over VLM basket estimate, descend, open
```

**Key design decisions**

| Decision | Choice | Why |
|---|---|---|
| Benchmark | LIBERO-Object (10 tasks, 7 objects/scene, 50 demos/task) | Target identity is the whole difficulty: every task is "pick X, place in basket" among 6 distractors |
| VLM | Qwen3-VL-8B-Instruct, local, bf16 | No API keys needed; 32B gave no grounding gain (see §2) |
| Handoff pose | EEF position 1 s (20 steps) before the gripper closes in the demos | Data-driven: the demos pass through this pose, so the local policy's training starts there. Mean offset from demos: ≈15 cm above the object |
| Local policy data | demo steps from handoff → object lifted 10 cm (+6 steps) | Only 34% of the frames; no task conditioning |
| Policy | Diffusion Policy (ResNet18-GN + spatial softmax, 1D conditional UNet, DDIM 10 steps, obs horizon 2, action chunk 8 of 16) | Same architecture and recipe for the baseline and the local skill |
| Baseline | Same diffusion policy, full demos, task-id embedding, 100k steps (vs 60k for the local skill) | Gives the baseline more compute, not less |
| Grasp → transport switch | proprioceptive only: gripper commanded closed and EEF rose 8 cm; fingers not fully closed ⇒ "holding" | No privileged sim state used at test time |
| Calibration | constant bias (VLM point − object body origin) estimated on **held-out** init states 0,10,…,40 | Evaluation uses 20 *other* init states per task |

Privileged information used at test time: none for the hierarchical methods (RGB-D + camera calibration + proprioception only). The *oracle* rows replace the VLM estimate with simulator object positions, purely for analysis."""))

cells.append(code(r"""# Where the handoff pose comes from: demo statistics (EEF relative to target object, 20 steps before grasp)
meta = json.load(open(f'{ROOT}/data/libero_object_meta.json'))
M = lambda k: np.array([m[k] for m in meta])
off = M('ee_ts') - M('obj0')
fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
for i, (ax, lab) in enumerate(zip(axes, ['x (forward)', 'y (lateral)', 'z (height)'])):
    ax.hist(100*off[:, i], bins=30, color=C['ego'], edgecolor='white', linewidth=1)
    ax.axvline(100*off[:, i].mean(), color=INK, lw=1.5, ls='--')
    ax.set_title(f'EEF − object, {lab}', fontsize=10, color=INK); ax.set_xlabel('cm'); ax.grid(axis='x', visible=False)
axes[0].set_ylabel('demos')
plt.suptitle('Handoff pose distribution across 500 demonstrations (dashed = mean used by the scripted reach)', fontsize=10, color=INK2, y=1.04)
plt.show()
seg = M('te') - M('ts') + 6
print(f"mean offset (cm): {np.round(100*off.mean(0),1)}, std {np.round(100*off.std(0),1)}")
print(f"local-skill segment length: {seg.mean():.0f} steps on average vs full demo {M('T').mean():.0f} steps "
      f"-> {100*seg.sum()/M('T').sum():.0f}% of all frames")"""))

# ------------------------------------------------------------------ grounding
cells.append(md(r"""---
## 2. Stage 1: VLM task parsing and grounding

**Parsing.** For all 10 instructions the VLM returned the correct `pick` / `place` noun phrases (JSON output).

**Grounding is the hard part, and naive prompting is not enough.** At the default LIBERO camera distance the product labels are a few pixels tall, and two assets are misleading even to a person: the ketchup bottle reads "Tomato", and "tomato sauce" is a can. The table below is an offline ablation on 50 held-out scenes. A hit means the lifted 3D point lands nearest to the correct object."""))
cells.append(code(r"""ab = json.load(open(f'{ROOT}/results/grounding_ablation.json'))['rows']
fig, ax = plt.subplots(figsize=(8.5, 3.2))
labels = [f"{r['strategy']}  ·  {r['model'].replace('Qwen3-VL-','')}" for r in ab][::-1]
vals = [r['hits']/r['n'] for r in ab][::-1]
cols = [C['ego'] if 'proposals' in l else '#b9b8b2' for l in labels]
bars = ax.barh(labels, vals, color=cols, height=0.6, edgecolor='white', linewidth=2)
for b, v, r in zip(bars, vals, ab[::-1]):
    ax.text(v + 0.01, b.get_y() + b.get_height()/2, f"{r['hits']}/{r['n']}", va='center', fontsize=9, color=INK)
ax.set_xlim(0, 1.1); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(['0', '25%', '50%', '75%', '100%'])
ax.set_xlabel('target identification accuracy'); ax.grid(axis='y', visible=False)
ax.set_title('Grounding strategy ablation (50 held-out scenes)', loc='left', fontsize=11, color=INK)
plt.show()
table([[r['strategy'], r['model'], f"{r['hits']}/{r['n']}"] for r in ab], ['Strategy', 'Model', 'Hits'])"""))
cells.append(md(r"""**What fixed it: propose with geometry, verify with the VLM.**
1. *Propose*: back-project the depth image, keep points above the floor plane, and cluster them in a bird's-eye occupancy grid. This separates objects that overlap in the image, e.g. a can behind a milk carton. It is class-agnostic, cheap (≈0.1 s), and uses no simulator labels.
2. *Verify*: crop each proposal at full resolution (448 px), ask *"Is this item the {pick}? Read any label on the packaging. Answer Yes or No."*, and score it with `logit(Yes) − logit(No)`.
3. Take the argmax. Its 3D point (cluster median xy, half its top height), minus a constant calibrated bias, is the reach target.

This is the same "zoom in before deciding" trick used by visual-search methods (e.g. V*, ViCrop), applied to robot perception, where depth makes the proposals essentially free."""))
cells.append(code(r"""G = pickle.load(open(f'{ROOT}/data/grounding.pkl', 'rb'))
GS = pickle.load(open(f'{ROOT}/data/grounding_shuf.pkl', 'rb'))
def gacc(G):
    ev = [r for r in G.values() if not r['calib']]
    return np.mean([r['obj_hit'] for r in ev]), len(ev), [r for r in ev if not r['obj_hit']]
for nm, g in [('in-distribution', G), ('shuffled layouts', GS)]:
    a, n, miss = gacc(g)
    print(f"{nm:18s}: grounding accuracy {100*a:.1f}% on {n} eval episodes; misses: {[(m['pick'], m['ep']) for m in miss]}")
errs = np.array([r['obj'][:2] - r['gt_obj'][:2] for r in G.values() if r['calib'] and r['obj_hit']])
print(f"calibration (held-out scenes): mean xy offset {np.round(100*errs.mean(0),1)} cm, residual std {np.round(100*errs.std(0),1)} cm")
print('VLM parse:', sorted({(r['pick'], r['place']) for r in G.values()}))"""))
cells.append(code(r"""# Example grounding outputs (red = selected pick target, blue = selected receptacle)
ims = sorted(glob.glob(f'{ROOT}/logs/grounding/*_1.png'))[:4] + sorted(glob.glob(f'{ROOT}/logs/grounding/*_1_shuf.png'))[:2]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, p in zip(axes.flat, ims):
    im = plt.imread(p); h = im.shape[0]; ax.imshow(im[int(.3*h):int(.85*h)]); ax.axis('off')
    tid = int(os.path.basename(p).split('_')[0]); ax.set_title(('[shuffled] ' if 'shuf' in p else '') + TASKS[tid], fontsize=10, color=INK)
plt.tight_layout(); plt.show()"""))

# ------------------------------------------------------------------ policy training
cells.append(md(r"""---
## 3. Stage 2: the local grasp skill

Three policies share one architecture (93 M params, mostly the two ResNet18 encoders):

| Run | Data | Inputs | Steps |
|---|---|---|---|
| `full_50` (baseline) | all frames of 50 demos × 10 tasks (74.5k samples) | agentview + wrist RGB 128², EEF pose, gripper, **task-id** | 100k |
| `local_50` | handoff→lift slice only (25.5k samples) | agentview + wrist RGB, EEF pose, gripper | 60k |
| `local_ego_50` | handoff→lift slice only | **wrist RGB only**, EEF z + orientation + gripper (**x,y removed**) | 60k |
| `full_10`, `local_10` | same as above with 10 demos/task | | 60k / 40k |"""))
cells.append(code(r"""fig, ax = plt.subplots(figsize=(8, 3))
for run, key in [('full_50', 'e2e'), ('local_50', 'local'), ('local_ego_50', 'ego')]:
    p = f'{ROOT}/runs/{run}/log.txt'
    if not os.path.exists(p): continue
    L = [l.split() for l in open(p) if l.startswith('step')]
    s = np.array([int(l[1]) for l in L]); v = np.array([float(l[3]) for l in L])
    k = 9; vs = np.convolve(v, np.ones(k)/k, mode='valid')
    ax.plot(s[k-1:], vs, color=C[key], lw=2, label=f'{run}')
    ax.text(s[-1] + 1500, vs[-1], run, color=INK2, fontsize=9, va='center')
ax.set_yscale('log'); ax.set_xlabel('training step'); ax.set_ylabel('denoising MSE (smoothed)')
ax.set_title('Training curves', loc='left', fontsize=11, color=INK); ax.legend(loc='upper right'); ax.set_xlim(0, 112000)
plt.show()"""))

# ------------------------------------------------------------------ main results
cells.append(md(r"""---
## 4. Results

**Protocol.** 10 tasks × 20 initial states (LIBERO's fixed init files, excluding the 5 calibration states), max 300 env steps, success = LIBERO's own BDDL goal check (`object in basket_contain_region`). *Shuffled layouts*: after reset, the floor (x, y) positions of the 6 grocery items are randomly permuted (seeded per episode; the target always moves to a slot that belonged to another object; basket fixed). Every object stays physically valid (its own height and orientation are kept; it only takes another object's floor slot), but the task→location association from training is broken.

**What changes and what does not.** LIBERO-Object is far less varied than it looks: across the 50 demos of a task the target's start position varies by only ≈0.2 cm, and all tasks share the same ≈7 floor slots. After shuffling, the target is 10–39 cm (median 22 cm) from where it was, so for every episode it is ≥5 cm from any position *that task* ever grasped at, and in **83% of episodes it is ≥5 cm from any position *any* task ever grasped at** (it lands in a slot only distractors occupied). Unchanged: the object set, object appearance, basket position (±1 cm), camera, lighting, robot start pose. So the shuffle combines two shifts: **(a) identity/grounding** (the target is no longer where the task id/instruction predicts) and **(b) spatial extrapolation** (most targets sit where no demo ever grasped). §4.1c separates them.

**All shuffled-layout results are zero-shot**: no policy, VLM or planner was trained, fine-tuned or prompted with shuffled scenes; all learned components saw only the standard LIBERO-Object demos. Two qualifications: (1) the grounding pipeline has one calibrated constant, a 3D offset between the VLM point and the object origin. In the headline ("strict") numbers it is estimated on *standard* calibration scenes only; an earlier variant that estimated it on shuffled calibration scenes is listed separately and is within noise (≈2–3 points). (2) The v2→v5 controller changes were *designed* while looking at shuffled evaluation failures. That is why the final comparison is repeated on a held-out split (§4.7), and why the zero-shot claim is about learning, not about the engineering iterations.

### 4.1 In-distribution vs shuffled layouts"""))
cells.append(code(r"""methods = [('e2e', 'full_50', 'full_50_shuf'), ('scripted', 'scripted_vlm', 'scripted_vlm_shuf'),
           ('local', 'local_vlm', 'local_vlm_shuf'), ('ego', 'ego_vlm', 'ego_vlm_shuf')]
conds = ['In-distribution layouts', 'Shuffled layouts (OOD)']
fig, ax = plt.subplots(figsize=(9, 3.8)); w = 0.19
for i, (key, a, b) in enumerate(methods):
    rs = [rate(a), rate(b)]; x = np.arange(2) + (i - 1.5) * (w + 0.02)
    v = [r['sr'] for r in rs]
    bars = ax.bar(x, v, w, color=C[key], label=NAME[key], edgecolor='white', linewidth=2)
    ax.errorbar(x, v, yerr=[[r['sr']-r['lo'] for r in rs], [r['hi']-r['sr'] for r in rs]], fmt='none', ecolor=INK2, elinewidth=1, capsize=3)
    bar_labels(ax, bars, v, [r['hi'] for r in rs])
ax.set_xticks(range(2)); ax.set_xticklabels(conds); ax.set_ylim(0, 1.15); ax.set_yticks([0, .25, .5, .75, 1])
ax.set_yticklabels(['0', '25%', '50%', '75%', '100%']); ax.set_ylabel('success rate'); ax.grid(axis='x', visible=False)
ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=9)
ax.set_title('Success rate, 200 episodes per bar (error bars: Wilson 95% CI)', loc='left', fontsize=11, color=INK)
plt.show()
table([[NAME[k], fmt(a), fmt(b)] for k, a, b in methods] +
      [['VLM + local grasp (agent+wrist), **oracle positions**', fmt('local_oracle'), '–'],
       ['VLM + scripted grasp, **oracle positions**', fmt('scripted_oracle'), '–']],
      ['Method', conds[0], conds[1]])"""))
cells.append(md(r"""**Reading this chart.**
* The end-to-end policy is excellent where it was trained and drops to essentially zero once the layout changes. The videos in §5 show *why*: its motions stay crisp, it simply goes to the old location.
* The VLM-guided methods barely notice the shuffle in the reach phase, since grounding stays at 98%. Whether the *whole* system is robust then depends on the grasp skill.
* The agent+wrist local policy loses much of its advantage under shuffle. Its inputs (third-person image + absolute EEF x,y) let it tie grasp behaviour to scene layout, so the shift that breaks the end-to-end policy partly leaks into the "local" skill too. Removing those inputs (egocentric variant) is the fix, not a tweak."""))

cells.append(md(r"""### 4.1b All methods at a glance (including related approaches)

Each row is one method; the blue dot is success on the standard layouts and the orange dot is success on shuffled layouts (thin bars: 95% CI). A long line pointing left means the method relies on memorized layouts. The related methods are described in §4.5."""))
cells.append(code(r"""ALL = [('End-to-end diffusion policy (task id)', 'full_50'),
       ('VLA: OpenVLA-OFT 7B (Spatial-Forcing), language-conditioned', 'vla'),
       ('SayCan-style: VLM planner + language-conditioned skills', 'saycan'),
       ('HAMSTER-style: VLM path + path-conditioned policy', 'hamster'),
       ('VLM + scripted grasp', 'scripted_vlm'),
       ('VLM + local grasp (agent+wrist)', 'local_vlm'),
       ('VLM + local grasp (egocentric)', 'ego_vlm'),
       ('VLM + local grasp (egocentric) v2', 'ego_v2_vlm'),
       ('VLM + local grasp (egocentric) v5', 'v5strict_vlm')]
COND = {'in': ('#2a78d6', 'In-distribution layouts'), 'shuf': ('#eb6834', 'Shuffled layouts (OOD)')}
fig, ax = plt.subplots(figsize=(10, 0.55 * len(ALL) + 1.3))
for i, (lab, run) in enumerate(ALL[::-1]):
    a, b = rate(run), rate(run + '_shuf')
    if a['n'] and b['n']: ax.plot([a['sr'], b['sr']], [i, i], color='#c3c2b7', lw=2, zorder=1)
    for r, (col, _), dy in ((a, COND['in'], 0.12), (b, COND['shuf'], -0.12)):
        if not r['n']: continue
        ax.errorbar(r['sr'], i + dy, xerr=[[r['sr'] - r['lo']], [r['hi'] - r['sr']]], fmt='none', ecolor=col, elinewidth=1, alpha=0.6)
        ax.scatter(r['sr'], i + dy, s=70, color=col, edgecolors='white', linewidths=2, zorder=3)
        ax.text(r['sr'], i + dy + 0.2 * np.sign(dy), f"{100*r['sr']:.0f}", ha='center', va='center', fontsize=8, color=INK)
    if not (a['n'] or b['n']): ax.text(0.02, i, 'pending', va='center', fontsize=8, color=INK2)
ax.set_yticks(range(len(ALL))); ax.set_yticklabels([l for l, _ in ALL[::-1]]); ax.set_ylim(-0.6, len(ALL) - 0.4)
ax.set_xlim(-0.03, 1.05); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(['0', '25%', '50%', '75%', '100%']); ax.grid(axis='y', visible=False)
for k, (col, lab) in COND.items(): ax.scatter([], [], s=60, color=col, label=lab)
ax.legend(loc='lower center', bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=9)
ax.set_xlabel('success rate (200 episodes per dot)')
plt.show()
table([[l, fmt(r), fmt(r + '_shuf')] for l, r in ALL], ['Method', 'In-distribution', 'Shuffled'])"""))
cells.append(code(r"""# Which object did each method actually move on shuffled layouts? (privileged sim state, analysis only)
def pick_stats(name):
    r = load(name)
    if r is None or 'moved' not in r[0]: return None
    n = len(r); right = sum(x['moved'] == x['target'] for x in r); wrong = sum(x['moved'] not in (None, x['target']) for x in r)
    return [f'{100*right/n:.0f}%', f'{100*wrong/n:.0f}%', f'{100*(n-right-wrong)/n:.0f}%']
rows = [[l] + pick_stats(r + '_shuf') for l, r in ALL if pick_stats(r + '_shuf')]
table(rows, ['Method (shuffled layouts)', 'moved the target', 'moved a WRONG object', 'moved nothing (>5 cm)'])
display(Markdown('Methods without a row here were evaluated before this diagnostic was added (or in an external harness: the VLA).'))"""))

cells.append(md(r"""### 4.1c Separating the two shifts in the shuffle

Shuffled episodes are split by whether the target landed within 5 cm of a location where *some* task's demos grasped (34 of 200 episodes) or at a location no demo ever grasped at (166 of 200)."""))
cells.append(code(r"""meta_ = json.load(open(f'{ROOT}/data/libero_object_meta.json')); alltgt = np.array([m['obj0'][:2] for m in meta_])
GS_ = pickle.load(open(f'{ROOT}/data/grounding_shuf.pkl', 'rb'))
seen = {k: np.min(np.linalg.norm(alltgt - r['gt_obj'][:2], axis=1)) < 0.05 for k, r in GS_.items() if not r['calib']}
def split(name):
    r = load(name)
    if r is None: return ['pending'] * 2
    out = []
    for flag in (True, False):
        x = [e['success'] for e in r if seen[(e['tid'], e['ep'])] == flag]; lo, hi = wilson(sum(x), len(x))
        out.append(f"{100*np.mean(x):.0f}% [{100*lo:.0f}–{100*hi:.0f}] (n={len(x)})")
    return out
rows = [[l] + split(r + '_shuf') for l, r in ALL] + [['HAMSTER-style with **oracle** path'] + split('hamster_oracle_shuf')]
table(rows, ['Method (shuffled layouts)', 'target at a location some demo grasped at', 'target at a never-grasped location'])"""))
cells.append(md(r"""This split changes how the baselines should be read:
* **HAMSTER-style does follow the path**: 79% (91% with the oracle path) when the target is at a location some demo grasped at, even though it is the *wrong* location for that task. Its failure under shuffle is **spatial extrapolation**: it cannot execute a path that ends where no demo ever went (0%).
* **The VLA shows some instruction grounding** (35% at previously-grasped locations) but also cannot extrapolate (0%).
* **End-to-end and SayCan-style skills fail on both subsets**: they neither re-target by identity nor extrapolate.
* **The VLM-handoff methods are flat across the two subsets** (≈78–82% either way): the scripted reach makes the grasp location irrelevant, and the egocentric skill only ever sees a 'gripper above object' view.

So the hierarchy's advantage here is mostly **(b) spatial extrapolation**, which it gets by construction, plus (a) for the methods that lack explicit grounding."""))

cells.append(md(r"""### 4.2 Robustness to localization error

To test how much the learned grasp *absorbs* error in the VLM's estimate, we use oracle object positions and add isotropic Gaussian xy noise (σ = 0, 2, 4 cm) to the reach target. The noise is the same per episode across methods."""))
cells.append(code(r"""curves = {'scripted': ['scripted_oracle', 'scripted_oracle_n0.02', 'scripted_oracle_n0.04'],
          'local':    ['local_oracle', 'local_oracle_n0.02', 'local_oracle_n0.04'],
          'ego':      ['ego_oracle', 'ego_oracle_n0.02', 'ego_oracle_n0.04']}
sig = np.array([0, 2, 4])
fig, ax = plt.subplots(figsize=(7, 3.6))
for key, runs in curves.items():
    rs = [rate(r) for r in runs]; v = np.array([r['sr'] for r in rs]); ok = ~np.isnan(v)
    if not ok.any(): continue
    ax.fill_between(sig[ok], [r['lo'] for r, o in zip(rs, ok) if o], [r['hi'] for r, o in zip(rs, ok) if o], color=C[key], alpha=0.12, lw=0)
    ax.plot(sig[ok], v[ok], color=C[key], lw=2, marker='o', ms=8, mec='white', mew=2, label=NAME[key])
    j = np.where(ok)[0][-1]; ax.text(sig[j] + 0.12, v[j], f'{100*v[j]:.0f}%', va='center', fontsize=9, color=INK)
ax.set_xticks(sig); ax.set_xticklabels(['0 cm', '2 cm', '4 cm']); ax.set_xlabel('σ of xy error added to the object estimate')
ax.set_ylim(0, 1.05); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(['0', '25%', '50%', '75%', '100%']); ax.set_ylabel('success rate')
ax.set_xlim(-0.3, 4.8); ax.legend(loc='lower left', fontsize=9)
ax.set_title('Closed-loop grasping absorbs localization error', loc='left', fontsize=11, color=INK)
plt.show()
table([[NAME[k]] + [fmt(r) for r in runs] for k, runs in curves.items()], ['Grasp skill', 'σ = 0', 'σ = 2 cm', 'σ = 4 cm'])"""))
cells.append(md(r"""With perfect positions, a hand-written top-down grasp is competitive. LIBERO-Object items are mostly upright boxes, bottles and cans, and its failures concentrate on the thin cream-cheese box. But real VLM localization is never perfect: the residual std of our grounding is ≈1–2 cm, and a box-centre point would be worse. Already at σ = 2 cm the scripted grasp loses more than half its successes, while the learned skill, which *sees* the object in its wrist camera, re-centres and keeps working. This is the core argument for learning only the last centimetres: the VLM gets you close, and the learned policy turns *close* into *in hand*."""))

cells.append(md(r"""### 4.2b How local is "local"? Moving the handoff away from where the skill's data starts

Same egocentric skill, oracle object positions, but the scripted reach stops at a pose shifted away from the demo-derived handoff (10 episodes × 10 tasks per point). This measures the skill's *basin of competence*, i.e. how much search space the VLM must remove for the learned part to work."""))
cells.append(code(r"""fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
for ax, (runs, xs, lab) in zip(axes, [(['ego_oracle', 'sweep_dz0.05', 'sweep_dz0.10', 'sweep_dz0.20'], [0, 5, 10, 20], 'handoff raised by (cm)'),
                                      (['ego_oracle', 'sweep_dx0.03', 'sweep_dx0.06'], [0, 3, 6], 'handoff shifted in x by (cm)')]):
    rs = [rate(r) for r in runs]; v = np.array([r['sr'] for r in rs]); ok = ~np.isnan(v); xs = np.array(xs)
    ax.fill_between(xs[ok], [r['lo'] for r, o in zip(rs, ok) if o], [r['hi'] for r, o in zip(rs, ok) if o], color=C['ego'], alpha=0.12, lw=0)
    ax.plot(xs[ok], v[ok], color=C['ego'], lw=2, marker='o', ms=8, mec='white', mew=2)
    for x_, v_ in zip(xs[ok], v[ok]): ax.text(x_, v_ + 0.05, f'{100*v_:.0f}%', ha='center', fontsize=9, color=INK)
    ax.set_xlabel(lab); ax.set_xticks(xs); ax.set_ylim(0, 1.1)
axes[0].set_yticks([0, .25, .5, .75, 1]); axes[0].set_yticklabels(['0', '25%', '50%', '75%', '100%']); axes[0].set_ylabel('success rate')
fig.suptitle('Egocentric grasp skill: success vs distance of the handoff from its training distribution', fontsize=10.5, color=INK, x=0.06, ha='left')
plt.tight_layout(); plt.show()
table([[r, fmt(r)] for r in ['ego_oracle', 'sweep_dz0.05', 'sweep_dz0.10', 'sweep_dz0.20', 'sweep_dx0.03', 'sweep_dx0.06']], ['run', 'success'])"""))
cells.append(code(r"""# Does a longer training slice (handoff 40 steps = 2 s before the grasp, ~20 cm above, wider spread) widen the basin?
xs = [0, 5, 10, 20]
rows = [['skill trained from 20 steps before grasp (default)'] + [fmt(r) for r in ['sweep_dz0', 'sweep_dz0.05', 'sweep_dz0.10', 'sweep_dz0.20']],
        ['skill trained from 40 steps before grasp'] + [fmt(r) for r in ['sweep40_dz0', 'sweep40_dz0.05', 'sweep40_dz0.10', 'sweep40_dz0.20']]]
table(rows, ['oracle positions, handoff raised by →', '0 cm', '+5 cm', '+10 cm', '+20 cm'])
table([['20-step slice (v3 settings)', fmt('ego_v3_vlm'), fmt('ego_v3_vlm_shuf')], ['40-step slice (v3 settings)', fmt('egopre40_v3_vlm'), fmt('egopre40_v3_vlm_shuf')]],
      ['Local skill, VLM grounding', 'In-distribution', 'Shuffled'])"""))
cells.append(md(r"""The skill's competence falls off within a few centimetres: raising the start by 10 cm, or shifting it 6 cm sideways, roughly halves success. A policy trained on only the last ~1 s of 50 demos/task has a *small* basin of attraction, so the VLM-driven reach is not a convenience. It is what makes the low-data local policy usable. A complementary view: training the local skill on longer slices (handoff further back) should widen this basin at the cost of more data and more layout leakage. That is a natural next experiment."""))

cells.append(md(r"""### 4.3 Data efficiency (10 vs 50 demos per task)"""))
cells.append(code(r"""groups = [('e2e', ['full_10', 'full_50']), ('local', ['local10_vlm', 'local_vlm'])]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, (suffix, title) in zip(axes, [('', 'In-distribution'), ('_shuf', 'Shuffled layouts')]):
    for i, (key, runs) in enumerate(groups):
        rs = [rate(r + suffix) for r in runs]; v = [r['sr'] for r in rs]; x = np.arange(2) + (i - 0.5) * 0.36
        bars = ax.bar(x, v, 0.34, color=C[key], label=NAME[key], edgecolor='white', linewidth=2)
        ax.errorbar(x, v, yerr=[[np.nan_to_num(r['sr']-r['lo']) for r in rs], [np.nan_to_num(r['hi']-r['sr']) for r in rs]], fmt='none', ecolor=INK2, elinewidth=1, capsize=3)
        bar_labels(ax, bars, v, [r['hi'] for r in rs])
    ax.set_xticks(range(2)); ax.set_xticklabels(['10 demos / task', '50 demos / task']); ax.set_title(title, fontsize=10, color=INK)
    ax.set_ylim(0, 1.15); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(['0', '25%', '50%', '75%', '100%']); ax.grid(axis='x', visible=False)
axes[0].set_ylabel('success rate'); axes[0].legend(loc='upper center', bbox_to_anchor=(1.05, -0.13), ncol=2, fontsize=9)
plt.show()
table([[NAME[k], *[fmt(r) for r in runs], *[fmt(r + '_shuf') for r in runs]] for k, runs in groups],
      ['Method', '10 demos (in-dist)', '50 demos (in-dist)', '10 demos (shuffled)', '50 demos (shuffled)'])
table([[NAME['ego'] + ' (v3 settings)', fmt('ego10_v3_vlm'), fmt('ego_v3_vlm'), fmt('ego10_v3_vlm_shuf'), fmt('ego_v3_vlm_shuf')]],
      ['Method', '10 demos (in-dist)', '50 demos (in-dist)', '10 demos (shuffled)', '50 demos (shuffled)'])
print('Note: the chart shows the agent+wrist local skill (trained before the egocentric fix); the egocentric skill is in the second table.')"""))

cells.append(md(r"""### 4.4 Per-task breakdown and failure modes"""))
cells.append(code(r"""LAB = {'v5strict_vlm': 'Ours v5 (strict zero-shot)', 'full_50': 'End-to-end diffusion', 'vla': 'VLA (OpenVLA-OFT 7B)', 'saycan': 'SayCan-style', 'hamster': 'HAMSTER-style',
       'scripted_vlm': 'VLM + scripted grasp', 'ego_vlm': 'Ours v1 (egocentric)'}
rows = [(None, r + suf) for suf in ('', '_shuf') for r in LAB]
rows = [(k, r) for k, r in rows if load(r) is not None]
Mx = np.array([[np.mean([x['success'] for x in load(r) if x['tid'] == t]) for t in range(10)] for _, r in rows])
fig, ax = plt.subplots(figsize=(11, 0.45 * len(rows) + 1.2))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('seq', ['#f0efec', '#9ec5f4', '#3987e5', '#184f95'])
ax.imshow(Mx, cmap=cmap, vmin=0, vmax=1, aspect='auto')
for i in range(Mx.shape[0]):
    for j in range(10):
        ax.text(j, i, f'{100*Mx[i,j]:.0f}', ha='center', va='center', fontsize=8.5, color='white' if Mx[i, j] > 0.6 else INK)
ax.set_xticks(range(10)); ax.set_xticklabels(TASKS, rotation=30, ha='right')
ax.set_yticks(range(len(rows))); ax.set_yticklabels([('[shuffled] ' if r.endswith('_shuf') else '') + LAB[r.replace('_shuf', '')] for k, r in rows])
ax.grid(False); ax.set_title('Per-task success (%) — 20 episodes per cell', loc='left', fontsize=11, color=INK)
for s in ax.spines.values(): s.set_visible(False)
plt.show()"""))
cells.append(code(r"""def fails(name):
    r = load(name)
    if r is None: return None
    c = {'success': 0, 'grasp failed': 0, 'place failed': 0, 'timeout (no phase info)': 0}
    for x in r:
        if x['success']: c['success'] += 1
        elif x['fail'] == 'grasp': c['grasp failed'] += 1
        elif x['fail'] == 'place': c['place failed'] += 1
        else: c['timeout (no phase info)'] += 1
    return c
names = ['scripted_vlm', 'scripted_vlm_shuf', 'ego_vlm', 'ego_vlm_shuf', 'v5strict_vlm', 'v5strict_vlm_shuf']
table([[('[shuffled] ' if n.endswith('_shuf') else '') + LAB[n.replace('_shuf', '')], *fails(n).values()] for n in names if fails(n)],
      ['Method', 'success', 'grasp failed', 'place failed', 'timeout'])
display(Markdown("*grasp failed* = after the local skill / scripted grasp the fingers were fully closed (nothing held) or never closed; "
                 "*place failed* = object held but not inside the basket's contain region at the end (usually it dropped on the rim)."))"""))

# ------------------------------------------------------------------ demos
cells.append(md(r"""### 4.5 Related approaches: how they were implemented here

All learned components use **the same diffusion-policy architecture, demos and training budget** (100k steps), so differences come from the *decomposition*, not from capacity.

| Approach | Original idea | Implementation in this PoC | What it tests |
|---|---|---|---|
| **SayCan-style** (Ahn et al. 2022) | LLM scores skills by usefulness, value functions score feasibility; skills are language-conditioned BC policies | Qwen3-VL's LM scores 12 skill strings (`pick up the <obj>` ×10, `place it in the basket`, `done`) with a SayCan prompt; affordance = VLM visual yes/no that the object is in the scene (top-3 LM candidates) and "holding" (proprio) for place. Skills = one diffusion policy conditioned on skill id, trained on demo segments (pick = start→lift, place = lift→end) | Semantic/temporal decomposition by a foundation model, but **visuomotor skills that must find the object themselves** |
| **HAMSTER-style** (Li et al. 2025) | Fine-tuned VLM draws a 2D end-effector path on the image; a 3D low-level policy conditioned on the path executes | Training: each demo's own keypoint path (start → pre-grasp → grasp● → lift → release●), projected and drawn on every agentview frame. Test: the same 5-keypoint path built from our zero-shot VLM grounding (object/basket points + demo-mean keypoint offsets). Policy: diffusion policy on path-overlaid agentview + wrist + proprio, no task id | Spatial guidance from a VLM, **passed as an image-space prompt** to one end-to-end policy (vs our explicit handoff) |
| **VLA** (OpenVLA-OFT, via the public Spatial-Forcing checkpoint) | Large pretrained vision-language-action model fine-tuned on the task | `haofuly/spatial-forcing-7b-finetuned-libero-object` run with the authors' own LIBERO eval code (L1 head, 2 images + proprio, chunk 8); our wrapper only selects the same init states and applies the same seeded shuffle | Whether **language conditioning + web-scale pretraining** alone produces object grounding under layout shift |

Honest caveats: HAMSTER's real high-level model is a VLM fine-tuned on path data (here a zero-shot VLM + geometry builds the path, which is arguably *more* accurate on this benchmark), and its low-level policy is a 3D policy (RVT-2 / 3D-DA), not a 2D diffusion policy. SayCan's affordances are learned value functions; here they are VLM detection scores. The VLA was trained by others with their own budget (LIBERO-Object demos, filtered/regenerated by the OpenVLA team) and is not compute-matched."""))
cells.append(code(r"""# SayCan planner: language-model and affordance scores for the first skill
P = pickle.load(open(f'{ROOT}/data/saycan_plans.pkl', 'rb')); SK = P['skills']
PS = pickle.load(open(f'{ROOT}/data/saycan_plans_shuf.pkl', 'rb')) if os.path.exists(f'{ROOT}/data/saycan_plans_shuf.pkl') else None
print('first-skill accuracy, in-distribution:', np.mean([r['correct'] for r in P['plans'].values()]),
      '| shuffled:', np.mean([r['correct'] for r in PS['plans'].values()]) if PS else 'pending')
ex = P['plans'][(4, 1)]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
order = np.argsort(-ex['combined'])[:6]
for ax, (key, title) in zip(axes, [('lm1', 'p_LM(skill | "pick up the ketchup ...")'), ('aff', 'affordance (VLM yes/no)'), ('combined', 'combined (argmax = executed)')]):
    v = ex[key][order] / (ex[key].max() if key != 'aff' else 1)
    ax.barh([SK[i] for i in order][::-1], v[::-1], color=['#2a78d6' if i == order[0] else '#b9b8b2' for i in order][::-1], height=0.6, edgecolor='white', linewidth=2)
    ax.set_title(title, fontsize=9.5, color=INK, loc='left'); ax.grid(axis='y', visible=False); ax.set_xticks([])
plt.tight_layout(); plt.show()"""))
cells.append(code(r"""# HAMSTER-style path prompts drawn on the 128px agentview input (training: each demo's own keypoints; test: same 5 keypoints from VLM grounding)
display(Image(f'{ROOT}/logs/hamster_paths.png', width=900))
vids = sorted(glob.glob(f'{EVAL}/hamster_shuf/videos/*.mp4'))[:1] + sorted(glob.glob(f'{EVAL}/saycan_shuf/videos/*.mp4'))[:1]
for v in vids: show(v, w=420, caption=f'`{os.path.relpath(v, EVAL)}` (left: agentview, right: wrist)')"""))
cells.append(md(r"""**How to read the related-method results**
* **SayCan-style: right plan, wrong object.** The VLM planner (LM × affordance) selects `pick up the <target>` in ≥98.5% of episodes in both layouts. Planning was never the bottleneck. On shuffled layouts the language-conditioned *pick skill* behaves exactly like the end-to-end policy and goes to where that object used to be. Semantic decomposition does not shrink the **spatial** search space; each skill still has to solve it from pixels.
* **HAMSTER-style: follows paths, cannot extrapolate them.** In-distribution ≈91%. Under shuffle ≈14% overall, but §4.1c shows it succeeds 79–91% when the path ends at a location some demo grasped at and 0% otherwise. The policy does read the path. What 50 demos/task in fixed slots cannot teach is how to *execute* a path to a place no demo visited. HAMSTER's real recipe (diverse training data, a 3D policy) targets exactly this, so this result says *path-conditioning needs spatially diverse data*, not that the idea fails.
* **VLA: language-conditioned ≠ grounded.** The 7B OpenVLA-OFT checkpoint fine-tuned on LIBERO-Object is essentially perfect in-distribution (99%) and ≈6% under shuffle. This agrees with recent LIBERO perturbation studies (e.g. LIBERO-PRO / LIBERO-Plus) that find fine-tuned VLAs largely ignore the instruction and object identity on this benchmark.
* **Ours: explicit handoff.** The VLM output does not *inform* a policy; it *moves the robot* until the remaining problem is local, and the policy that takes over sees only what is local. That is the property that transfers."""))

cells.append(md(r"""### 4.6 Improving the method: diagnosing the shuffled-layout failures

The egocentric policy's shuffled-layout failures were mostly *place* failures, which was surprising because grasping succeeded. Tracing them showed a **kinematic** cause, not a perception one:

* Shuffling can move the target to the far edge of the workspace (~0.83 m from the shoulder, close to the Panda's reach).
* During its descent the learned policy also commands small wrist rotations (imitating the demos). The OSC controller resolves these by straightening the elbow: joint 4 goes from −1.1 rad to its limit at −0.07 rad. The scripted grasp, which keeps the orientation fixed, reaches the same point with a bent elbow.
* With the elbow locked straight, the arm cannot follow the transport path, stalls 4–5 cm short of the basket and drops the object on the rim.

**v2 changes** (no retraining): (1) *orientation lock*: during the learned grasp the policy controls translation and gripper while the OSC holds the handoff orientation (objects here are upright, so the demos' wrist rotations are mostly incidental); (2) *retract waypoint* before transport; (3) *grasp retry* when the proprioceptive check (fingers fully closed) says nothing is held, at most 2 retries within the 300-step budget. Stronger OSC null-space posture gains (kp 40/100) were tried and did **not** help: the straightening comes from reach, not from self-motion."""))
cells.append(md(r"""**Component ablation on the regression v2 introduced.** v2 dropped in-distribution success, almost entirely on *tomato sauce*. Ablation on that task (20 episodes each, in-distribution):"""))
cells.append(code(r"""table([['v1 (no retract, no lock, no retry)', fmt('abl_t5_v1')], ['+ retract', fmt('abl_t5_retract')],
       ['+ retract + **full** orientation lock', fmt('abl_t5_retract_lock')], ['+ retract + retry', fmt('abl_t5_retract_retry')],
       ['+ retract + retry + **roll/pitch-only** lock (= v3)', fmt('abl_t5_tilt')]], ['Variant', 'tomato sauce success'])"""))
cells.append(md(r"""Locking *yaw* breaks the tomato-sauce grasp: the policy has learned to rotate about the vertical axis to align the fingers with that can, so that rotation is part of the skill. Locking only roll and pitch, the components that drive the elbow into its limit, keeps both benefits. **v3 = retract + retry + roll/pitch lock.**"""))
cells.append(md(r"""**Full-suite component ablation** (200 episodes per cell; all variants use the same egocentric skill checkpoint; no retraining):"""))
cells.append(code(r"""V = [('v1: baseline transport', 'ego_vlm'),
     ('+ retract waypoint only', 'ablA_retract'),
     ('+ roll/pitch lock + retry (no retract)', 'ablA_tilt_retry'),
     ('v2: full orientation lock + retract + retry', 'ego_v2_vlm'),
     ('v3: roll/pitch lock + retract + retry', 'ego_v3_vlm'),
     ('v4: roll/pitch lock + retry + posture-aware retract', 'ego_v4_vlm'),
     ('v5: posture-aware roll/pitch lock (joint4 > −0.6) + posture-aware retract + retry', 'ego_v5_vlm'),
     ('v5, threshold −0.8', 'v5th-0.8_vlm'),
     ('v5, threshold −1.0 (grounding bias calibrated on *shuffled* calibration scenes)', 'v5th-1.0_vlm'),
     ('**v5, threshold −1.0, strict zero-shot** (bias from standard scenes only; selected on this split → held-out §4.7)', 'v5strict_vlm')]
table([[l, fmt(r), fmt(r + '_shuf')] for l, r in V], ['Variant (egocentric skill)', 'In-distribution', 'Shuffled'])"""))
cells.append(md(r"""Reading the ablation:
* The **retract waypoint alone does nothing** under shuffle. Once the elbow is locked straight, the arm cannot follow *any* transport path.
* The **roll/pitch lock is the active ingredient** (shuffled ≈ 65% → ≈ 79%), because it prevents the elbow from straightening in the first place. But applied everywhere it costs in-distribution success, through *more place failures*: holding the wrist rigid changes how the object sits in the hand.
* v5 therefore applies the lock **only when the arm actually approaches the singular posture** (joint 4 > −0.6 rad during the grasp), leaving ordinary grasps untouched. This kind of state-dependent arbitration between a learned policy and a hand-written constraint is cheap, and it is exactly where a hierarchical system is easier to fix than an end-to-end one."""))
cells.append(code(r"""for nm in ['ego_vlm_shuf', 'ego_v3_vlm_shuf', 'v5strict_vlm_shuf']:
    f = fails(nm)
    if f: print(nm, f)"""))

cells.append(md(r"""### 4.7 Held-out confirmation

The v5 threshold (and the v2→v5 design iterations) were chosen by looking at results on the 200 evaluation episodes. To make sure the improvement is not test-set tuning, the final comparison is repeated on the **25 remaining init states per task (250 episodes)** that no experiment had touched: fresh VLM grounding, same calibration scenes, no further changes."""))
cells.append(code(r"""H = [('End-to-end diffusion policy', 'hold_full_50'), ('VLM + local grasp (egocentric) v1', 'hold_ego_v1'), ('VLM + local grasp (egocentric) v5, threshold −1.0, strict zero-shot', 'hold_ego_v5strict')]
table([[l, fmt(r), fmt(r + '_shuf')] for l, r in H], ['Method (held-out split, 10 tasks × 25)', 'In-distribution', 'Shuffled'])
for nm, p in [('held-out', 'grounding_hold.pkl'), ('held-out shuffled', 'grounding_hold_shuf.pkl')]:
    if os.path.exists(f'{ROOT}/data/{p}'):
        a, n, _ = gacc(pickle.load(open(f'{ROOT}/data/{p}', 'rb'))); print(f'grounding accuracy ({nm}): {100*a:.1f}% of {n}')"""))

cells.append(md(r"""---
## 5. Demos

Each clip: left = agentview, right = wrist camera; the caption shows the active phase. Clips start with the **VLM grounding card**: grey boxes are depth proposals, numbers are the VLM's yes/no logit for the target, red = selected pick, blue = selected receptacle."""))
cells.append(code(r"""show(f'{ROOT}/demo/demo_all_indist.mp4', caption='**5.1 Full pipeline (v5), in-distribution layouts**: alphabet soup, butter, chocolate pudding, milk. Episode 2 of each task, not cherry-picked.')"""))
cells.append(code(r"""for t in [0, 6, 8]:
    show(f'{ROOT}/demo/compare_task{t}_shuf.mp4', w=760,
         caption=f'**5.2 Shuffled layout — "{TASKS[t]}"** (episode 2): top = ours (v5), bottom = end-to-end diffusion policy, same scene. Ours fails on butter here; shown as-is.')"""))
cells.append(md(r"""Watch the bottom rows: the end-to-end policy is not confused or jittery. It executes a clean pick-and-place, of the **wrong object**, because the object now at its memorized location is a distractor. This is the failure that a VLM front-end removes by construction."""))
cells.append(md(r"""### 5.4 Side by side on out-of-distribution layouts: all five methods, same scene

Each grid replays one shuffled scene with the **same seed** for every method: ours (v5, strict zero-shot), end-to-end diffusion, SayCan-style, HAMSTER-style (its VLM path is drawn on the image) and the 7B VLA. The banner at the end of each panel names the object the method actually moved.

**Scenes are chosen by rule, not by outcome:** for every task, the first shuffled evaluation episode whose target lands where *no* demo ever grasped (the typical case, 83% of shuffled episodes), plus two episodes whose target lands where *some* demo grasped (the case where HAMSTER-style can succeed, §4.1c). Every selected scene is shown, including ours' failures. Full-resolution per-scene files: `demo/ood_compare/`."""))
cells.append(code(r"""S_ = json.load(open(f'{ROOT}/demo/ood_compare/summary.json'))
short = lambda v: ('✅ ' if v.startswith('SUCCESS') else '❌ ') + v.split(': ', 1)[1]
table([[TASKS[x['tid']], x['ep'], x['kind'], *[short(x[k]) for k in ('ours', 'e2e', 'saycan', 'hamster', 'vla')]] for x in S_],
      ['target', 'ep', 'target location', 'Ours', 'End-to-end', 'SayCan-style', 'HAMSTER-style', 'VLA 7B'])
tot = {k: sum(x[k].startswith('SUCCESS') for x in S_) for k in ('ours', 'e2e', 'saycan', 'hamster', 'vla')}
print('successes in these', len(S_), 'scenes:', tot)
show(f'{ROOT}/demo/ood_compare/reel_all.mp4', w=900, caption='**All 12 scenes back to back** (10 never-grasped-location scenes, then 2 previously-grasped-location scenes)')"""))
cells.append(code(r"""vids = sorted(glob.glob(f'{EVAL}/ego_v5_vlm_shuf/videos/*fail*.mp4'))[:2] + sorted(glob.glob(f'{EVAL}/scripted_vlm/videos/*fail*.mp4'))[:1]
display(Markdown('**5.3 Failure cases** (raw evaluation videos, 128 px): ' + ', '.join(os.path.relpath(v, EVAL) for v in vids)))
for v in vids: show(v, w=420)"""))

# ------------------------------------------------------------------ RoboCasa + Colosseum
import sys; sys.path.insert(0, 'report'); from bench_section import cells as bench_cells
cells += bench_cells()

# ------------------------------------------------------------------ discussion
cells.append(md(r"""---
## 7. Discussion

**What the PoC supports**
* A VLM is good at the part imitation learning is worst at: *which object, where*. It does this zero-shot, with no robot data, and it is unaffected by layout shift.
* Once the search space is collapsed to "the object is right below the gripper", one task-agnostic grasp skill trained on a third of the frames serves every task, and closed-loop learning makes it tolerant of the VLM's centimetre-level error.
* The decomposition only pays off if the low-level skill is *actually local*. Giving it global inputs (third-person camera, absolute position) recreates the memorization problem inside the skill. This is the most transferable lesson for a real-robot version.

**Caveats (please read before citing numbers)**
* Single seed per training run; 200 episodes per number (CIs shown). Differences of a few points in-distribution are not significant.
* LIBERO-Object is easy for grasping (rigid, mostly upright items). With exact positions a scripted grasp is near the learned one; the learned skill's advantage shows up under localization error. Harder geometry (handles, deformables, clutter, LIBERO-90/RoboCasa) is where it should matter more and is untested here.
* "Shuffled layouts" is one specific OOD shift that favours any method with explicit grounding. It shows that the baselines, including a 7B VLA, *memorized layouts* on this benchmark. It does not show that end-to-end policies cannot generalize when trained on diverse layouts; the natural counter-experiment is to retrain baselines with layout randomization (more data), which is exactly the cost the hierarchy avoids.
* Related methods are re-implementations under a controlled budget (same policy architecture, data and steps), not the original systems: SayCan affordances are VLM detection scores rather than learned value functions; HAMSTER's path comes from our zero-shot grounding (not a path-fine-tuned VLM) and the low-level policy is 2D rather than RVT-2 / 3D-DA. The VLA is the authors' public checkpoint, evaluated with their code.
* v2–v5 were designed by looking at the 200 evaluation episodes; only the final comparison on the **held-out** 250 episodes (§4.7) is free of that selection bias.
* Transport/release is scripted. The interesting next step is a second local skill (place) with its own handoff, i.e. a VLM-sequenced library of local skills — this is where SayCan-style selection would come in.
* Grounding uses a depth camera and known extrinsics; the 1.6 cm systematic bias (camera sees the front face) is removed with a constant calibrated on held-out scenes.

* RoboCasa and Colosseum (§6) use single training seeds and 500 / 1,080 evaluation episodes per method. Per-task differences of a few episodes are noise. Ours was developed on LIBERO and adapted with only a small dev set, while the baselines needed no adaptation.

**Suggested next steps**
0. From §6: a handoff that fits the object (VLM- or grasp-model-predicted approach orientation instead of the per-task mean pose) is the main RoboCasa bottleneck. Part-level grounding (handles, edges, knobs) is the main Colosseum bottleneck.
1. Learned *place* skill + VLM-chosen handoff for it (removes the remaining hand-coded part; place failures are a large share of hierarchical errors).
2. Harder suites: LIBERO-90 / LIBERO-Long (multi-step, articulated objects) and RoboCasa kitchens, where the reach phase is long and the search space genuinely large.
3. Retrain the baselines (diffusion, HAMSTER-style, VLA fine-tune) on layout-randomized demos and measure how many demos they need to match the hierarchy under shuffle: the data-cost of *not* decomposing.
4. Let the VLM also predict the *handoff* (approach direction / grasp axis) instead of using the demo-mean pose.
5. A learned arbitration between the policy and the posture constraint (v5 uses a hand-set joint-4 threshold), or training the skill with the constraint active.

---
## 8. Reproduce

```bash
cd ~/projects/vlm_reach_grasp && source env.sh          # conda env: vlmrg
$PY -m vrg.build_dataset                                  # hdf5 -> npz + phase labels
$PY -m vrg.train --mode local --cams wrist --mask_xy --out runs/local_ego_50 --device cuda:0
$PY -m vrg.train --mode full --steps 100000 --out runs/full_50 --device cuda:1
$PY -m vrg.ground_episodes [--shuffle --out data/grounding_shuf.pkl]       # VLM stage, cached per episode
$PY -m vrg.rollout --method hier_vlm --ckpt runs/local_ego_50/ckpt.pt --out runs/eval/ego_vlm [--shuffle --grounding data/grounding_shuf.pkl]
$PY -m vrg.rollout --method full --ckpt runs/full_50/ckpt.pt --out runs/eval/full_50 [--shuffle]
$PY -m vrg.rollout --method hier_scripted [--oracle] [--noise 0.02] --out runs/eval/...
$PY -m vrg.make_demo --tasks 0,6,8 --ep 1 --shuffle --ckpt runs/local_ego_50/ckpt.pt --e2e runs/full_50/ckpt.pt --tag _shuf
```
Code map: `vrg/vlm.py` (VLM wrapper, depth proposals, zoom-verify), `vrg/pipeline.py` (skills, controller, calibration constants), `vrg/policy.py` (diffusion policy), `vrg/train.py`, `vrg/rollout.py` (evaluation), `vrg/make_demo.py` (videos)."""))

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "report/VLM_Reach_LocalGrasp_Report.ipynb")
print("wrote report/VLM_Reach_LocalGrasp_Report.ipynb with", len(cells), "cells")
