#!/bin/bash
# pi0.5-DROID joint-position (PolaRiS sim co-trained) server on port 8001; checkpoint cached in $OPENPI_DATA_HOME
source /scratch/kaustubh/vlm_reach_grasp/src/openpi_env.sh; cd $S/src/openpi
exec .venv/bin/python -u scripts/serve_policy.py --port 8001 policy:checkpoint --policy.config=pi05_droid_jointpos_polaris --policy.dir=gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris
