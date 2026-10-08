#!/bin/bash
# After run_remaining_evals.sh: zero-noise ego oracle run, demo videos, then build + execute the report notebook
cd ~/projects/vlm_reach_grasp; source env.sh
until grep -q ALL_DONE logs/remaining_evals.log; do sleep 30; done
while pgrep -f "out runs/full_50 " >/dev/null; do sleep 30; done
$PY -m vrg.rollout --method hier_vlm --oracle --ckpt runs/local_ego_50/ckpt.pt --n_eps 20 --video 0 --out runs/eval/ego_oracle 2>&1 | grep -E "^==|Error"
$PY -m vrg.make_demo --tasks 0,6,8,7 --ep 1 --ckpt runs/local_ego_50/ckpt.pt --out demo --tag _indist 2>&1 | grep -E "success|Error"
$PY -m vrg.make_demo --tasks 0,6,8 --ep 1 --shuffle --ckpt runs/local_ego_50/ckpt.pt --e2e runs/full_50/ckpt.pt --out demo --tag _shuf 2>&1 | grep -E "success|Error"
python report/build_report.py
cd report && jupyter nbconvert --to notebook --execute --inplace VLM_Reach_LocalGrasp_Report.ipynb && jupyter nbconvert --to html VLM_Reach_LocalGrasp_Report.ipynb
echo REPORT_DONE
