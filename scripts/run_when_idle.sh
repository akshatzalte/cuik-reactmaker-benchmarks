#!/usr/bin/env bash
# Wait for the machine to go idle, then run the full benchmark suite.
#
# Timings here are contention-sensitive in a way that does not cancel out: under
# load the Python path degrades more than the C++ path, so a busy machine
# *inflates* the measured speedup (observed 2026-11-05: 9.1x at load 87 vs 8.3x
# idle). Published numbers must come from an idle host, so this script refuses to
# start until the host looks idle and records the load it saw throughout.
#
# Usage:
#   bash scripts/run_when_idle.sh [GPU_ID]
#   MAX_LOAD=6 MAX_GPU_MEM_MIB=500 CHECKS=3 bash scripts/run_when_idle.sh 0
#
# Activate the environment first (see README requirements), then run it detached
# so it survives a disconnect:
#   conda activate <env with chemprop >= 2.3.1>
#   nohup bash scripts/run_when_idle.sh 0 > logs/idle_runner.out 2>&1 &
#
set -u -o pipefail

# Fail now rather than after a multi-hour wait: the suite and the provenance
# block both need chemprop and cuik-molmaker importable.
if ! python -c "import chemprop, cuik_molmaker" 2>/dev/null; then
    echo "error: chemprop / cuik_molmaker not importable — activate an environment" >&2
    echo "       with chemprop >= 2.3.1 before starting (see README requirements)." >&2
    exit 2
fi

GPU=${1:-0}
MAX_LOAD=${MAX_LOAD:-8}            # 1-minute load average must be below this
MAX_GPU_MEM_MIB=${MAX_GPU_MEM_MIB:-1000}   # other processes' memory on the target GPU
CHECKS=${CHECKS:-3}                # consecutive idle checks required
INTERVAL=${INTERVAL:-300}          # seconds between checks
MAX_WAIT_HOURS=${MAX_WAIT_HOURS:-72}

cd "$(dirname "$0")/.."
mkdir -p logs results/raw
STAMP=$(date +%Y%m%d-%H%M)
LOG=logs/benchmark_run_${STAMP}.log
PROV=results/run_provenance_${STAMP}.txt

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

load1()    { awk '{print $1}' /proc/loadavg; }
gpu_mem()  { nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits; }
is_idle() {
    local l m
    l=$(load1); m=$(gpu_mem)
    awk -v l="$l" -v ml="$MAX_LOAD" -v m="$m" -v mm="$MAX_GPU_MEM_MIB" \
        'BEGIN { exit !(l < ml && m < mm) }'
}

log "waiting for idle: load1 < $MAX_LOAD and GPU $GPU memory < ${MAX_GPU_MEM_MIB} MiB, ${CHECKS}x ${INTERVAL}s apart"
log "once idle the full suite runs: featurization (~20 min CPU), training 1k-300k x 3 seeds (~4 h GPU), inference (~30 min GPU), figures"

deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
streak=0
while [ "$streak" -lt "$CHECKS" ]; do
    if [ "$(date +%s)" -gt "$deadline" ]; then
        log "gave up after ${MAX_WAIT_HOURS}h without an idle window — nothing was run"
        exit 1
    fi
    if is_idle; then
        streak=$((streak + 1))
        log "idle check $streak/$CHECKS passed (load $(load1), GPU $GPU $(gpu_mem) MiB)"
    else
        [ "$streak" -gt 0 ] && log "idle streak broken (load $(load1), GPU $GPU $(gpu_mem) MiB)"
        streak=0
    fi
    [ "$streak" -lt "$CHECKS" ] && sleep "$INTERVAL"
done

log "host is idle — starting benchmark suite on GPU $GPU"

# Provenance: record the machine and sample load for the whole run, so the
# resulting numbers can be defended as idle-host measurements.
{
    echo "=== Benchmark run provenance ==="
    echo "Started: $(date -Is)"
    echo "Host: $(hostname)"
    echo "Gate: load1 < $MAX_LOAD, GPU $GPU mem < ${MAX_GPU_MEM_MIB} MiB, ${CHECKS} consecutive checks"
    echo
    echo "--- CPU ---"; lscpu | grep -E "^Model name|^CPU\(s\)|^Thread|^Core"
    echo "--- Memory ---"; free -h | head -2
    echo "--- GPU ---"; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    echo "--- Software ---"
    python -c "import chemprop, cuik_molmaker, rdkit, torch; print('chemprop', chemprop.__version__); print('rdkit', rdkit.__version__); print('torch', torch.__version__); print('cuik_molmaker', getattr(cuik_molmaker,'__version__','n/a'))"
    pip list 2>/dev/null | grep -iE "^chemprop|^cuik|^rdkit|^torch "
    echo
    echo "--- Load samples during run (every 60 s) ---"
} > "$PROV"

( while true; do echo "$(date '+%F %T') load=$(cat /proc/loadavg | cut -d' ' -f1-3) gpu${GPU}=$(gpu_mem)MiB" >> "$PROV"; sleep 60; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

bash scripts/experiments.sh "$GPU" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}

kill $SAMPLER 2>/dev/null
{
    echo
    echo "Finished: $(date -Is)   exit status: $STATUS"
    echo "Peak 1-min load observed during run: $(grep -o 'load=[0-9.]*' "$PROV" | cut -d= -f2 | sort -g | tail -1)"
} >> "$PROV"

log "benchmark suite finished with status $STATUS; provenance in $PROV"
exit "$STATUS"
