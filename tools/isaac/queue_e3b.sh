#!/bin/bash
# Re-run of the 32B part of E2/E3 on ONE GPU (the two-GPU split produced NaN generations; outputs kept in bad_32b_split/).
# Budget: 32B bf16 ~66G on GPU 0, <=80G host RAM while loading; then one Isaac run (3.4G RAM measured, 12G cap, ~4G GPU).
P=$HOME/projects/vlm_reach_grasp; R=/scratch/kaustubh/vlm_reach_grasp/isaac; L=$P/tools/launch_job.sh; VPY=$HOME/anaconda3/envs/vlmrg/bin/python
waitu(){ until [ $(systemctl --user list-units "$1" --plain --no-legend | wc -l) -eq 0 ]; do sleep 60; done; }
waitu 'vrg-t32a*'
$L fr-g32b 80 0 70 $R/logs/ground_32b.log "" -- /bin/bash -c "$VPY -u -m vrg.franka.ground_offline --scenes $R/scenes_dev --models qwen3vl-32b --out $R/ground_dev --device cuda:0 && $VPY -u -m vrg.franka.ground_offline --scenes $R/scenes_eval --models qwen3vl-32b --out $R/ground_eval --device cuda:0"
sleep 60; waitu 'vrg-fr-g32b*'
$VPY - <<PY
import json; s = json.load(open("$R/ground_dev/summary.json")); best = {}
for k, v in s.items():
    m, mode = k.rsplit("_", 1)
    if m not in best or v["acc"] > s[f"{m}_{best[m]}"]["acc"]: best[m] = mode
json.dump(best, open("$R/ground_eval/best_mode_dev.json", "w"), indent=1); print(best)
PY
mode=$($VPY -c "import json;print(json.load(open('$R/ground_eval/best_mode_dev.json'))['qwen3vl-32b'])")
$L fr-e3-qwen3vl-32b 12 1 6 $R/logs/e3_qwen3vl-32b.log "" -- $R/run.sh -m vrg.franka.run_isaac --seeds 100-149 --grounding file:$R/ground_eval/qwen3vl-32b_$mode.json --controller cartesian --refine --video 6 --out $R/runs/e3_qwen3vl-32b
sleep 60; waitu 'vrg-fr-e3-*'; touch $R/runs/e3_done; echo "[$(date +%H:%M)] E3b done"
