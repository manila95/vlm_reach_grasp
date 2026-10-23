# openpi (pi0.5) environment, all on /scratch (home disk is nearly full)
export S=/scratch/kaustubh/vlm_reach_grasp
export PATH=$S/envs/uv:$PATH UV_CACHE_DIR=$S/cache/uv UV_PYTHON_INSTALL_DIR=$S/envs/uv/python OPENPI_DATA_HOME=$S/cache/openpi
export XDG_CACHE_HOME=$S/cache HF_HOME=$S/cache/hf GIT_LFS_SKIP_SMUDGE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30   # shared GPUs: JAX would otherwise grab 75%
