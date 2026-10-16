#!/bin/bash
# Gated reactive configs (user suggestion 2026-10-16: stop trusting views once the approach gets too close).
# Each model's two configs start only after that model's queue_rx chain finished, so there is still one Isaac client
# per model at a time. Budget unchanged: client 3.4G RAM measured (12G cap), ~4G GPU, ~5 cores.
P=$HOME/projects/vlm_reach_grasp; R=/scratch/kaustubh/vlm_reach_grasp/isaac; L=$P/tools/launch_job.sh; VPY=$HOME/anaconda3/envs/vlmrg/bin/python
declare -A PORT=([2b]=6102 [4b]=6104 [8b]=6108 [32b]=6132)
declare -A CG=([2b]=1 [4b]=1 [8b]=0 [32b]=0)
chain(){ s=$1
  until grep -q "$s ens3_K4 done" $R/logs/queue_rx.log; do sleep 60; done
  best=$($VPY -c "import json;print(json.load(open('$R/ground_eval/best_mode_dev.json'))['qwen3vl-$s'])")
  ens3=$($VPY -c "print('+'.join([m for m in ['point','marks','verify'] if m!='$best']+['$best']))")
  for cfg in "gate_last:ext=$best,rule=last" "gate_vote:ext=$ens3,rule=vote"; do
    name=${cfg%%:*}; opts=${cfg#*:}
    $L fr-rx-$s-$name 12 ${CG[$s]} 6 $R/logs/rx_${s}_$name.log "" -- $R/run.sh -m vrg.franka.run_isaac --seeds 100-149 --reactive port=${PORT[$s]},mode=$best,K=6,gate=area5,$opts --video 4 --out $R/runs/rx_${s}_$name
    sleep 30; until ! systemctl --user is-active -q vrg-fr-rx-$s-$name; do sleep 30; done
    echo "[$(date +%H:%M)] $s $name done"
  done
}
for s in 2b 4b 8b 32b; do chain $s & sleep 5; done
wait; touch $R/runs/gate_done; echo "GATE DONE"
