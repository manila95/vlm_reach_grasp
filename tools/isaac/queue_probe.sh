#!/bin/bash
# Reactive-grounding design data (dev seeds 0-9): VLM servers for all four sizes + one probe client per model.
# Budget: servers 2B 4G / 4B 8G / 8B 16G GPU0, 32B 62G GPU1 (host <3G each, measured); clients 3.4G RAM, ~4G GPU, 4.5 cores.
P=$HOME/projects/vlm_reach_grasp; R=/scratch/kaustubh/vlm_reach_grasp/isaac; L=$P/tools/launch_job.sh; VPY=$HOME/anaconda3/envs/vlmrg/bin/python
declare -A PORT=([qwen3vl-2b]=6102 [qwen3vl-4b]=6104 [qwen3vl-8b]=6108 [qwen3vl-32b]=6132)
declare -A GPU=([qwen3vl-2b]=0 [qwen3vl-4b]=0 [qwen3vl-8b]=0 [qwen3vl-32b]=1)
declare -A GGB=([qwen3vl-2b]=6 [qwen3vl-4b]=10 [qwen3vl-8b]=18 [qwen3vl-32b]=64)
for m in qwen3vl-2b qwen3vl-4b qwen3vl-8b qwen3vl-32b; do
  s=${m#qwen3vl-}; systemctl --user is-active -q vrg-fr-vlm$s || \
    $L fr-vlm$s 20 ${GPU[$m]} ${GGB[$m]} $R/logs/vlm_server_$s.log "serving|Error" -- $VPY -u -m vrg.franka.vlm_server --model $m --port ${PORT[$m]}
done
for m in qwen3vl-2b qwen3vl-4b qwen3vl-8b qwen3vl-32b; do
  s=${m#qwen3vl-}; mode=$($VPY -c "import json;print(json.load(open('$R/ground_eval/best_mode_dev.json'))['$m'])")
  $L fr-probe$s 12 0 6 $R/logs/probe_$s.log "" -- $R/run.sh -m vrg.franka.run_isaac --seeds 0-9 --reactive port=${PORT[$m]},mode=$mode,probe=1,K=4 --out $R/runs/dev_probe_$s
  sleep 20
done
sleep 60; until [ $(systemctl --user list-units 'vrg-fr-probe*' --plain --no-legend | wc -l) -eq 0 ]; do sleep 30; done
echo "PROBE DONE"
