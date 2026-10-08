#!/bin/bash
# Waits for training runs, then runs all remaining evaluations (results -> runs/eval/*/results.json)
cd ~/projects/vlm_reach_grasp; source env.sh
R() { $PY -m vrg.rollout "$@" 2>&1 | grep -E "^==|Error"; }
wait_for() { while pgrep -f "out runs/$1 " >/dev/null; do sleep 30; done; }
wait_for local_ego_50
R --method hier_vlm --ckpt runs/local_ego_50/ckpt.pt --n_eps 20 --video 2 --out runs/eval/ego_vlm
R --method hier_vlm --shuffle --grounding data/grounding_shuf.pkl --ckpt runs/local_ego_50/ckpt.pt --n_eps 20 --video 2 --out runs/eval/ego_vlm_shuf
R --method hier_vlm --oracle --noise 0.02 --ckpt runs/local_ego_50/ckpt.pt --n_eps 20 --video 0 --out runs/eval/ego_oracle_n0.02
R --method hier_vlm --oracle --noise 0.04 --ckpt runs/local_ego_50/ckpt.pt --n_eps 20 --video 0 --out runs/eval/ego_oracle_n0.04
R --method hier_vlm --shuffle --grounding data/grounding_shuf.pkl --ckpt runs/local_10/ckpt.pt --n_eps 20 --video 0 --out runs/eval/local10_vlm_shuf
wait_for full_10
R --method full --ckpt runs/full_10/ckpt.pt --n_eps 20 --video 0 --out runs/eval/full_10
R --method full --shuffle --ckpt runs/full_10/ckpt.pt --n_eps 20 --video 0 --out runs/eval/full_10_shuf
echo ALL_DONE
