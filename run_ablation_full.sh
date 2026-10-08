#!/bin/bash
cd ~/projects/vlm_reach_grasp; source env.sh
R() { echo -n "$1: "; shift; $PY -m vrg.rollout "$@" 2>&1 | grep -E "^==|Error"; }
C="--ckpt runs/local_ego_50/ckpt.pt --n_eps 20 --video 0"; S="--shuffle --grounding data/grounding_shuf.pkl"
R retract        --method hier_vlm $C --out runs/eval/ablA_retract
R tilt_retry     --method hier_vlm $C --no_retract --lock_ori tilt --retries 2 --out runs/eval/ablA_tilt_retry
R v4             --method hier_vlm $C --auto_retract --lock_ori tilt --retries 2 --out runs/eval/ego_v4_vlm
R v4_shuf        --method hier_vlm $C $S --auto_retract --lock_ori tilt --retries 2 --out runs/eval/ego_v4_vlm_shuf
R tilt_retry_shuf --method hier_vlm $C $S --no_retract --lock_ori tilt --retries 2 --out runs/eval/ablA_tilt_retry_shuf
R retract_shuf   --method hier_vlm $C $S --out runs/eval/ablA_retract_shuf
echo ABL_DONE
