#!/bin/bash
# Isaac Sim 5.1 (pip) + Isaac Lab v2.3.x in a prefix env on /scratch. Blackwell GPUs need Isaac Sim >= 5.0.
set -ex
S=/scratch/kaustubh/vlm_reach_grasp; E=$S/envs/isaaclab
export PIP_CACHE_DIR=$S/cache/pip TMPDIR=$S/cache/tmp; mkdir -p $TMPDIR
[ -x $E/bin/python ] || ~/anaconda3/bin/conda create -y -p $E python=3.11
$E/bin/pip install -U pip
$E/bin/pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
$E/bin/pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
[ -d $S/src/IsaacLab ] || git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git $S/src/IsaacLab
cd $S/src/IsaacLab
for p in isaaclab isaaclab_assets isaaclab_tasks; do $E/bin/pip install -e source/$p; done
echo INSTALL_DONE
