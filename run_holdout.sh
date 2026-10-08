#!/bin/bash
cd ~/projects/vlm_reach_grasp; source env.sh
$PY -m vrg.ground_episodes --holdout --out data/grounding_hold.pkl > logs/ground_hold.log 2>&1
$PY -m vrg.ground_episodes --holdout --shuffle --out data/grounding_hold_shuf.pkl > logs/ground_hold_shuf.log 2>&1
R() { echo -n "$1: "; shift; $PY -m vrg.rollout "$@" 2>&1 | grep -E "^==|Error"; }
C="--ckpt runs/local_ego_50/ckpt.pt --n_eps 25 --video 0 --holdout"
R v1_hold      --method hier_vlm $C --no_retract --grounding data/grounding_hold.pkl --out runs/eval/hold_ego_v1
R v1_hold_shuf --method hier_vlm $C --no_retract --shuffle --grounding data/grounding_hold_shuf.pkl --out runs/eval/hold_ego_v1_shuf
V5="--auto_retract --lock_ori auto --retries 2 --j4_thresh=-1.0"
R v5_hold      --method hier_vlm $C $V5 --grounding data/grounding_hold.pkl --out runs/eval/hold_ego_v5
R v5_hold_shuf --method hier_vlm $C $V5 --shuffle --grounding data/grounding_hold_shuf.pkl --out runs/eval/hold_ego_v5_shuf
R e2e_hold      --method full --ckpt runs/full_50/ckpt.pt --n_eps 25 --video 0 --holdout --out runs/eval/hold_full_50
R e2e_hold_shuf --method full --shuffle --ckpt runs/full_50/ckpt.pt --n_eps 25 --video 0 --holdout --out runs/eval/hold_full_50_shuf
echo HOLD_DONE
