#!/bin/bash
# Reactive grounding on eval (seeds 100-149, 257 eps). One chain per model (configs in sequence, resuming from
# results.json); the four chains run in parallel. VLM servers already up (2/4/8B on GPU0, 32B on GPU1).
# Budget: 4 clients x (3.4G RAM measured, 12G cap, ~4G GPU, 4.5 cores); clients split over both GPUs because four
# Isaac renderers next to three VLM servers saturated GPU0 (88% util).
P=$HOME/projects/vlm_reach_grasp; R=/scratch/kaustubh/vlm_reach_grasp/isaac; L=$P/tools/launch_job.sh; VPY=$HOME/anaconda3/envs/vlmrg/bin/python
declare -A PORT=([2b]=6102 [4b]=6104 [8b]=6108 [32b]=6132)
declare -A CG=([2b]=1 [4b]=1 [8b]=0 [32b]=0)
chain(){ s=$1
  best=$($VPY -c "import json;print(json.load(open('$R/ground_eval/best_mode_dev.json'))['qwen3vl-$s'])")
  for cfg in ens1_K2 ens3_K0 ens3_K2 ens3_K4; do
    ens=${cfg%_*}; K=${cfg#*_K}; ext=$best
    [ $ens = ens3 ] && ext=$($VPY -c "print('+'.join([m for m in ['point','marks','verify'] if m!='$best']+['$best']))")
    $L fr-rx-$s-$cfg 12 ${CG[$s]} 6 $R/logs/rx_${s}_$cfg.log "" -- $R/run.sh -m vrg.franka.run_isaac --seeds 100-149 --reactive port=${PORT[$s]},mode=$best,ext=$ext,K=$K --video 4 --out $R/runs/rx_${s}_$cfg
    sleep 30; until ! systemctl --user is-active -q vrg-fr-rx-$s-$cfg; do sleep 30; done
    echo "[$(date +%H:%M)] $s $cfg done"
  done
}
for s in 2b 4b 8b 32b; do chain $s & sleep 20; done
wait; touch $R/runs/rx_done; echo "RX DONE"
