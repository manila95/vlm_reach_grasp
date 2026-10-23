#!/bin/bash
set -e; source /scratch/kaustubh/vlm_reach_grasp/src/openpi_env.sh
[ -x $S/envs/uv/uv ] || (curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$S/envs/uv INSTALLER_NO_MODIFY_PATH=1 sh)
cd $S/src/openpi && git submodule update --init --recursive --depth 1 && uv sync && uv pip install -e .
echo "INSTALL DONE"
