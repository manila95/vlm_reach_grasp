#!/bin/bash
# Launch a long job as its own systemd user unit, only once the machine can hold it.
#
#   tools/launch_job.sh NAME PEAK_GB GPU GPU_GB LOG READY_REGEX -- CMD...
#     NAME         unit becomes vrg-NAME (systemctl --user status vrg-NAME)
#     PEAK_GB      peak host RAM incl. data loading (see the table in CLAUDE.md); becomes the unit's MemoryMax
#     GPU          cuda index, or -1 for none;  GPU_GB: GPU memory it needs
#     LOG          stdout/stderr are appended here
#     READY_REGEX  returns only once LOG matches this (e.g. "samples"), so the next launch sees the job's real
#                  footprint and data loads never overlap. "" returns immediately.
#
# Why: on 2026-10-10 overlapping data loads exhausted host RAM and the OOM kill took down every job plus the
# Claude session, because they shared one tmux cgroup. Each unit here is its own cgroup with a hard MemoryMax,
# so a job that outgrows its budget is killed alone.
set -u
NAME=$1 PEAK=$2 GPU=$3 GPU_GB=$4 LOG=$5 READY=$6; shift 6; [ "$1" = "--" ] && shift
HEADROOM_GB=${HEADROOM_GB:-24}   # kept free for other users, page cache and estimation error
READY_MIN=${READY_MIN:-60}       # give up waiting for READY after this long (exit 2); the RAM gate still guards later launches
P=$HOME/projects/vlm_reach_grasp; S=/scratch/kaustubh/vlm_reach_grasp
log(){ echo "[$(date +%H:%M)] launch $NAME: $*"; }

if systemctl --user is-active -q vrg-$NAME; then log "already running"; exit 1; fi
while :; do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  gfree=999; [ "$GPU" -ge 0 ] && gfree=$(( $(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits) / 1024 ))
  [ $((avail - PEAK)) -ge $HEADROOM_GB ] && [ $gfree -ge $GPU_GB ] && break
  log "waiting: RAM avail ${avail}G (need ${PEAK}+${HEADROOM_GB}), GPU$GPU free ${gfree}G (need ${GPU_GB})"; sleep 120
done
log "RAM avail ${avail}G, GPU free ${gfree}G -> starting (MemoryMax ${PEAK}G)"
echo "--- $(date '+%F %T') launched as vrg-$NAME" >> "$LOG"; L0=$(wc -l < "$LOG")
systemd-run --user --unit=vrg-$NAME --collect --working-directory=$P -p MemoryMax=${PEAK}G -p MemorySwapMax=4G \
  -E PYTHONPATH=$P -E MUJOCO_GL=egl -E __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  -E HF_HOME=$S/cache/hf $([ "$GPU" -ge 0 ] && echo "-E CUDA_VISIBLE_DEVICES=$GPU") \
  /bin/bash -c "$(printf '%q ' "$@") >> $(printf '%q' "$LOG") 2>&1" || exit 1
[ -z "$READY" ] && exit 0
T0=$(date +%s)
until tail -n +$((L0 + 1)) "$LOG" | grep -q -E "$READY"; do
  systemctl --user is-active -q vrg-$NAME || { log "unit exited before ready; see $LOG"; exit 1; }
  [ $(( $(date +%s) - T0 )) -gt $((READY_MIN * 60)) ] && { log "WARNING: not ready after ${READY_MIN} min, moving on (check it)"; exit 2; }
  sleep 20
done
log "ready"
