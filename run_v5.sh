#!/bin/bash
cd ~/projects/vlm_reach_grasp; source env.sh
R() { echo -n "$1: "; shift; $PY -m vrg.rollout "$@" 2>&1 | grep -E "^==|Error"; }
C="--ckpt runs/local_ego_50/ckpt.pt --n_eps 20 --video 2"; S="--shuffle --grounding data/grounding_shuf.pkl"
R v5      --method hier_vlm $C --auto_retract --lock_ori auto --retries 2 --out runs/eval/ego_v5_vlm
R v5_shuf --method hier_vlm $C $S --auto_retract --lock_ori auto --retries 2 --out runs/eval/ego_v5_vlm_shuf
echo V5_DONE
