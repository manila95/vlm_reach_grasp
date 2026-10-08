#!/bin/bash
# How local must the handoff be? Egocentric policy, oracle positions, handoff moved away from the demo-derived pose.
cd ~/projects/vlm_reach_grasp; source env.sh
until grep -q "^==" logs/ego_v2.log && [ $(grep -c "^==" logs/ego_v2.log) -ge 2 ]; do sleep 60; done
for dz in 0.05 0.10 0.20; do
  $PY -m vrg.rollout --method hier_vlm --oracle --handoff_dz $dz --ckpt runs/local_ego_50/ckpt.pt --n_eps 10 --video 0 --out runs/eval/sweep_dz$dz 2>&1 | grep -E "^==|Error"
done
for dx in 0.03 0.06; do
  $PY -m vrg.rollout --method hier_vlm --oracle --handoff_dx $dx --ckpt runs/local_ego_50/ckpt.pt --n_eps 10 --video 0 --out runs/eval/sweep_dx$dx 2>&1 | grep -E "^==|Error"
done
echo SWEEP_DONE
