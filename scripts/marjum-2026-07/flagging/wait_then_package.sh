#!/bin/bash
# Waits for the B16 full-campaign fit (PID given as $1) to exit, then runs
# the DPSS-outlier packaging step. Detached, logs its own wait + handoff.
set -u
B16_PID="$1"
LOG="/tmp/rfi-b16-linear-wt/marjum-2026-07/flags/b16/_run_logs/wait_then_package.log"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }

log "waiting for B16 fit PID $B16_PID to exit"
while kill -0 "$B16_PID" 2>/dev/null; do
    sleep 30
done
log "B16 fit PID $B16_PID has exited -- checking summary for success before packaging"

SUMMARY=/tmp/rfi-b16-linear-wt/marjum-2026-07/flags/b16/_run_logs/full_campaign_summary.json
if [ ! -f "$SUMMARY" ]; then
    log "ERROR: $SUMMARY not found -- B16 job likely crashed before writing it. NOT running packaging. Manual review needed."
    exit 1
fi
log "summary found, proceeding to packaging"

cd /tmp/rfi-b16-linear-wt/marjum-2026-07/flagging
MARJUM_DATA_ROOT=/mnt/data02/eigsep/marjum-2026-07 \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/aparsons/.local/share/mamba/envs/arp/bin/python3 package_dpss_into_mask.py \
    >> "$LOG" 2>&1
log "packaging script exited with code $?"
