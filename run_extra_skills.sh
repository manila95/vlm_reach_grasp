#!/bin/bash
# After SayCan/HAMSTER training: train longer-slice and 10-demo egocentric skills, then evaluate them.
cd ~/projects/vlm_reach_grasp; source env.sh
R() { $PY -m vrg.rollout "$@" 2>&1 | grep -E "^==|Error"; }
until grep -q "step 100000" logs/train_saycan_50.log && grep -q "step 100000" logs/train_hamster_50.log; do sleep 60; done
sleep 90
$PY -m vrg.train --mode local --cams wrist --mask_xy --pre 40 --steps 60000 --out runs/local_ego_pre40 --device cuda:0 > logs/train_local_ego_pre40.log 2>&1 &
$PY -m vrg.train --mode local --cams wrist --mask_xy --demos_per_task 10 --steps 40000 --out runs/local_ego_10 --device cuda:1 > logs/train_local_ego_10.log 2>&1 &
wait
V3="--lock_ori tilt --retries 2"
R --method hier_vlm $V3 --ckpt runs/local_ego_10/ckpt.pt --n_eps 20 --video 0 --out runs/eval/ego10_v3_vlm
R --method hier_vlm $V3 --shuffle --grounding data/grounding_shuf.pkl --ckpt runs/local_ego_10/ckpt.pt --n_eps 20 --video 0 --out runs/eval/ego10_v3_vlm_shuf
R --method hier_vlm $V3 --pre 40 --ckpt runs/local_ego_pre40/ckpt.pt --n_eps 20 --video 0 --out runs/eval/egopre40_v3_vlm
R --method hier_vlm $V3 --pre 40 --shuffle --grounding data/grounding_shuf.pkl --ckpt runs/local_ego_pre40/ckpt.pt --n_eps 20 --video 0 --out runs/eval/egopre40_v3_vlm_shuf
for dz in 0.05 0.10 0.20; do R --method hier_vlm --oracle --pre 40 --handoff_dz $dz --ckpt runs/local_ego_pre40/ckpt.pt --n_eps 10 --video 0 --out runs/eval/sweep40_dz$dz; done
R --method hier_vlm --oracle --pre 40 --ckpt runs/local_ego_pre40/ckpt.pt --n_eps 10 --video 0 --out runs/eval/sweep40_dz0
R --method hier_vlm --oracle --ckpt runs/local_ego_50/ckpt.pt --n_eps 10 --video 0 --out runs/eval/sweep_dz0
echo EXTRA_DONE
