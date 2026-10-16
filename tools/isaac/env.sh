# Source before running anything Isaac (vrg.franka.*). Headless rendering needs NVIDIA's EGL vendor: conda's activate.d
# sets __EGL_VENDOR_LIBRARY_DIRS to a Mesa-only dir, and then NVIDIA's Vulkan ICD finds no GPU ("No device could be created").
unset __EGL_VENDOR_LIBRARY_DIRS DISPLAY
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACPY=/scratch/kaustubh/vlm_reach_grasp/envs/isaaclab/bin/python
export PYTHONPATH=$HOME/projects/vlm_reach_grasp
export HF_HOME=/scratch/kaustubh/vlm_reach_grasp/cache/hf
