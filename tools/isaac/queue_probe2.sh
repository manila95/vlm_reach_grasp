#!/bin/bash
# Dev probes with view-quality features (K=6 checkpoints), one model at a time (the eval chains already use ~26 cores).
# Budget: one Isaac client (3.4G RAM measured, 12G cap, ~4G GPU1, ~5 cores); VLM servers already up.
P=$HOME/projects/vlm_reach_grasp; R=/scratch/kaustubh/vlm_reach_grasp/isaac; L=$P/tools/launch_job.sh; VPY=$HOME/anaconda3/envs/vlmrg/bin/python
declare -A PORT=([2b]=6102 [4b]=6104 [8b]=6108 [32b]=6132)
for s in 2b 32b 4b 8b; do
  best=$($VPY -c "import json;print(json.load(open('$R/ground_eval/best_mode_dev.json'))['qwen3vl-$s'])")
  $L fr-probe2-$s 12 1 6 $R/logs/probe2_$s.log "" -- $R/run.sh -m vrg.franka.run_isaac --seeds 0-9 --reactive port=${PORT[$s]},mode=$best,probe=1,K=6 --out $R/runs/dev_probe2_$s
  sleep 30; until ! systemctl --user is-active -q vrg-fr-probe2-$s; do sleep 20; done; echo "[$(date +%H:%M)] probe2 $s done"
done
echo "PROBE2 DONE"
