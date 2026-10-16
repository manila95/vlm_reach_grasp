#!/bin/bash
# E1: grasp / controller / perception ablations with oracle grounding, eval scenes 100-149 (dev = 0-9).
# Budget (2026-10-15): one Isaac run (CPU physics + RTX render) <= 12G RAM cap, ~4G GPU, ~4.5 cores; MAXJ=4 -> ~48G cap, ~16G GPU, ~18 cores.
P=$HOME/projects/vlm_reach_grasp; R=/scratch/kaustubh/vlm_reach_grasp/isaac; L=$P/tools/launch_job.sh; MAXJ=4; S=100-149
slots(){ until [ $(systemctl --user list-units 'vrg-fr-e1-*' --plain --no-legend | wc -l) -lt $MAXJ ]; do sleep 30; done; }
run(){ name=$1; shift; slots; $L fr-e1-$name 12 $((RANDOM % 2)) 6 $R/logs/e1_$name.log "" -- $R/run.sh -m vrg.franka.run_isaac --seeds $S --grounding oracle --video 6 --out $R/runs/e1_$name "$@"; sleep 30; }
run b_refine       --controller cartesian --refine --save_scenes $R/scenes_eval
run a_extonly      --controller cartesian
run c_fixedyaw     --controller cartesian --refine --grasp fixed
run d_joint        --controller joint --refine
run e_native       --controller native --refine
run f_gtseg        --controller cartesian --refine --perception gt
until [ $(systemctl --user list-units 'vrg-fr-e1-*' --plain --no-legend | wc -l) -eq 0 ]; do sleep 60; done
touch $R/runs/e1_done; echo "[$(date +%H:%M)] E1 done"
