#!/bin/bash
# Wrapper for Isaac jobs under tools/launch_job.sh: sets the headless-rendering env and runs the module. run_isaac exits
# with code 1 when its per-episode watchdog fires (a hung render); it resumes from results.json, so retry a few times.
source /scratch/kaustubh/vlm_reach_grasp/isaac/env.sh
cd $HOME/projects/vlm_reach_grasp
for attempt in 1 2 3 4 5; do
  $ISAACPY -u "$@" && exit 0
  echo "[run.sh] attempt $attempt exited with $?; restarting (resumes from results.json)"
done
exit 1
