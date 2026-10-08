#!/bin/bash
# Waits for SayCan/HAMSTER training (log reaches step 100000) and planning, then evaluates both, in-dist + shuffled.
cd ~/projects/vlm_reach_grasp; source env.sh
R() { $PY -m vrg.rollout "$@" 2>&1 | grep -E "^==|Error"; }
until grep -q "step 100000" logs/train_saycan_50.log && grep -q "step 100000" logs/train_hamster_50.log && [ -f data/saycan_plans_shuf.pkl ]; do sleep 60; done
sleep 90
R --method saycan --ckpt runs/saycan_50/ckpt.pt --n_eps 20 --video 1 --out runs/eval/saycan
R --method saycan --shuffle --ckpt runs/saycan_50/ckpt.pt --n_eps 20 --video 2 --out runs/eval/saycan_shuf
R --method hamster --ckpt runs/hamster_50/ckpt.pt --n_eps 20 --video 1 --out runs/eval/hamster
R --method hamster --shuffle --grounding data/grounding_shuf.pkl --ckpt runs/hamster_50/ckpt.pt --n_eps 20 --video 2 --out runs/eval/hamster_shuf
R --method hamster --oracle --shuffle --grounding data/grounding_shuf.pkl --ckpt runs/hamster_50/ckpt.pt --n_eps 20 --video 0 --out runs/eval/hamster_oracle_shuf
echo BASELINES_DONE
