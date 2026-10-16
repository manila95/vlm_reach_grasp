#!/bin/bash
# E2/E3: 32B grounding sweep (split over both GPUs), pick each model's mode on DEV scenes, then closed-loop runs with VLM choices.
# Budget (2026-10-15): 32B bf16 ~66G GPU over 2 GPUs (~33G each), loads through <= 80G host RAM; Isaac runs 3.4G RAM measured
# (12G cap), ~4G GPU, ~4.5 cores, MAXJ=3.
P=$HOME/projects/vlm_reach_grasp; R=/scratch/kaustubh/vlm_reach_grasp/isaac; L=$P/tools/launch_job.sh; VPY=$HOME/anaconda3/envs/vlmrg/bin/python
waitu(){ until [ $(systemctl --user list-units "$1" --plain --no-legend | wc -l) -eq 0 ]; do sleep 60; done; }
until [ -f $R/runs/e1_done ]; do sleep 60; done; waitu 'vrg-fr-e1*'; waitu 'vrg-fr-geval-small*'; waitu 'vrg-fr-gdev4b*'
$L fr-g32 80 -1 0 $R/logs/ground_32b.log "" -- /bin/bash -c "$VPY -u -m vrg.franka.ground_offline --scenes $R/scenes_dev --models qwen3vl-32b --out $R/ground_dev --device auto && $VPY -u -m vrg.franka.ground_offline --scenes $R/scenes_eval --models qwen3vl-32b --out $R/ground_eval --device auto"
best(){ $VPY - <<PY
import json; s = json.load(open("$R/ground_dev/summary.json")); best = {}
for k, v in s.items():
    m, mode = k.rsplit("_", 1)
    if m not in best or v["acc"] > s[f"{m}_{best[m]}"]["acc"]: best[m] = mode
json.dump(best, open("$R/ground_eval/best_mode_dev.json", "w"), indent=1); print(best)
PY
}
best
for m in qwen3vl-2b qwen3vl-4b qwen3vl-8b; do
  until [ $(systemctl --user list-units 'vrg-fr-e3-*' --plain --no-legend | wc -l) -lt 3 ]; do sleep 30; done
  mode=$($VPY -c "import json;print(json.load(open('$R/ground_eval/best_mode_dev.json'))['$m'])")
  $L fr-e3-$m 12 $((RANDOM % 2)) 6 $R/logs/e3_$m.log "" -- $R/run.sh -m vrg.franka.run_isaac --seeds 100-149 --grounding file:$R/ground_eval/${m}_$mode.json --controller cartesian --refine --video 6 --out $R/runs/e3_$m
  sleep 30
done
waitu 'vrg-fr-g32*'; best
mode=$($VPY -c "import json;print(json.load(open('$R/ground_eval/best_mode_dev.json'))['qwen3vl-32b'])")
$L fr-e3-qwen3vl-32b 12 1 6 $R/logs/e3_qwen3vl-32b.log "" -- $R/run.sh -m vrg.franka.run_isaac --seeds 100-149 --grounding file:$R/ground_eval/qwen3vl-32b_$mode.json --controller cartesian --refine --video 6 --out $R/runs/e3_qwen3vl-32b
sleep 60; waitu 'vrg-fr-e3-*'; touch $R/runs/e3_done; echo "[$(date +%H:%M)] E3 done"
