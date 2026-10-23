#!/bin/bash
# pi0.5-DROID policy server (openpi, JAX) on port 8000; checkpoint cached in $OPENPI_DATA_HOME
source /scratch/kaustubh/vlm_reach_grasp/src/openpi_env.sh; cd $S/src/openpi
exec .venv/bin/python -u scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
