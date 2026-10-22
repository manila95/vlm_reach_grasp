#!/bin/bash
# Timing breakdown (seeds 100-109): oracle config b, 4B one-shot, 4B ens3 K=2. One Isaac client at a time + one 4B server.
# Budget: server 8G GPU0 (<3G RAM); client 3.4G RAM measured (12G cap), ~4G GPU, ~5 cores.
P=$HOME/projects/vlm_reach_grasp; R=/scratch/kaustubh/vlm_reach_grasp/isaac; L=$P/tools/launch_job.sh; VPY=$HOME/anaconda3/envs/vlmrg/bin/python
$L fr-vlm4b 20 0 10 $R/logs/vlm_server_4b.log "serving|Error" -- $VPY -u -m vrg.franka.vlm_server --model qwen3vl-4b --port 6104
run(){ name=$1; shift; rm -rf $R/runs/$name; $L fr-$name 12 0 6 $R/logs/$name.log "" -- $R/run.sh -m vrg.franka.run_isaac --seeds 100-109 --timing --controller cartesian --out $R/runs/$name "$@"
       sleep 20; until ! systemctl --user is-active -q vrg-fr-$name; do sleep 15; done; }
run time_oracle --grounding oracle --refine
run time_4b_oneshot --reactive port=6104,mode=point,ext=point,K=0
run time_4b_ens3K2 --reactive port=6104,mode=point,ext=marks+verify+point,K=2
systemctl --user stop vrg-fr-vlm4b; echo "TIMING DONE"
