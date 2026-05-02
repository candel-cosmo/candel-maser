#!/bin/bash -l
# Submit maser disk runs for one or more galaxies. Supports NSS (nested
# sampling, mode2 only), NUTS, and DE (differential-evolution MAP, mode2
# only). Cluster (arc or glamdring) is picked up from `machine` in
# local_config.toml via _submit_lib.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

SAMPLER=""
MODE=""
QUEUE=""
F_GRID=""
NUM_CHAINS=1
GALAXY=""
INIT_METHOD=""
GPUTYPE=""
GPU_MEM=""
TIME=""
MEM=7
CPUS=""
DRY=false
_WATCH_RETRIES=""
_WATCH_POLL=""
RESUME=false
NO_ECC=false
NO_QUAD_WARP=false

ALL_GALS="CGCG074-064 NGC4258 NGC5765b NGC6264 NGC6323 UGC3789"

usage() {
    cat <<EOF
Usage: $0 -q QUEUE --sampler nss|nuts|de --galaxy GAL[,GAL,...] [options]

Required:
  -q, --queue QUEUE      Queue/partition (glamdring: gpulong|cmbgpu|optgpu;
                         arc: short|medium|long)
  --sampler nss|nuts|de  Inference method.
                         de = differential-evolution MAP (mode2 only).
  --galaxy GAL[,GAL,...] Galaxy/galaxies to submit (comma-separated).
                         Choices: $ALL_GALS

Options:
  --mode MODE            Sampling mode: mode1, mode2
                         (default: runner picks — mode2 for NSS;
                          ignored for DE which forces mode2)
  --f-grid F             Grid scaling factor (default: 1.0; nss/nuts only)
  --num-chains N         NUTS vectorised chains (default: $NUM_CHAINS)
  --init-method METHOD   NUTS init method: config | median | sample
                         (default: runner picks from config)
  --cpus N               CPU cores (default: 4 with --gpu)
  --gputype TYPE         GPU type (default: any; e.g. h100, l40s)
  --gpu-mem GB           Min GPU VRAM in GB (arc only; queries sinfo)
  --time T               Wall time. Bare integer = hours (arc only).
                         (default on arc: short=12, medium=48, long=required;
                          ignored on glamdring)
  --mem GB               Memory in GB (default: $MEM)
  --no-ecc               Disable eccentricity model
  --no-quadratic-warp    Disable quadratic disk warp
  --dry                  Print submit command without submitting (default: off)
  --resume               Resume from latest checkpoint (nss/de; ignored for NUTS)
  --max-retries N        Watch and resubmit up to N times on timeout
  --poll S               Seconds between squeue polls (default: 120)
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sampler) SAMPLER="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        -q|--queue) QUEUE="$2"; shift 2 ;;
        --f-grid) F_GRID="$2"; shift 2 ;;
        --num-chains) NUM_CHAINS="$2"; shift 2 ;;
        --init-method) INIT_METHOD="$2"; shift 2 ;;
        --galaxy) GALAXY="$2"; shift 2 ;;
        --gputype) GPUTYPE="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --mem) MEM="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --no-ecc) NO_ECC=true; shift ;;
        --no-quadratic-warp) NO_QUAD_WARP=true; shift ;;
        --dry) DRY=true; shift ;;
        --resume) RESUME=true; shift ;;
        --max-retries) _WATCH_RETRIES="$2"; shift 2 ;;
        --poll) _WATCH_POLL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$SAMPLER" != "nss" && "$SAMPLER" != "nuts" && "$SAMPLER" != "de" ]]; then
    echo "[ERROR] --sampler is required (nss|nuts|de)"; exit 1
fi
JOB_TAG=""
$NO_ECC && JOB_TAG="${JOB_TAG}_noecc"
$NO_QUAD_WARP && JOB_TAG="${JOB_TAG}_noqw"

# If --max-retries is set, delegate to the watcher and exit. One watcher
# per galaxy so each retries independently as soon as its own job ends.
if [[ -n "$_WATCH_RETRIES" ]]; then
    if [[ -z "$GALAXY" ]]; then
        echo "[ERROR] --galaxy is required"; exit 1
    fi
    _watcher="$ROOT/scripts/megamaser/watch_and_resubmit.sh"
    if [[ "$SAMPLER" == "de" ]]; then
        _marker="MAP init"
    else
        _marker="saved samples to"
    fi
    _wargs=(--marker "$_marker" --max-retries "$_WATCH_RETRIES")
    [[ -n "$_WATCH_POLL" ]] && _wargs+=(--poll "$_WATCH_POLL")
    [[ "$SAMPLER" == "nuts" ]] && _wargs+=(--no-resume)
    # Per-galaxy submit command (--galaxy is appended in the loop below).
    _cmd=(bash "$0" --sampler "$SAMPLER" -q "$QUEUE" --mem "$MEM")
    [[ -n "$MODE" ]]        && _cmd+=(--mode "$MODE")
    [[ -n "$F_GRID" ]]      && _cmd+=(--f-grid "$F_GRID")
    [[ -n "$INIT_METHOD" ]] && _cmd+=(--init-method "$INIT_METHOD")
    [[ -n "$GPUTYPE" ]]     && _cmd+=(--gputype "$GPUTYPE")
    [[ -n "$GPU_MEM" ]]     && _cmd+=(--gpu-mem "$GPU_MEM")
    [[ -n "$TIME" ]]        && _cmd+=(--time "$TIME")
    [[ -n "$CPUS" ]]        && _cmd+=(--cpus "$CPUS")
    [[ "$NUM_CHAINS" != "1" ]] && _cmd+=(--num-chains "$NUM_CHAINS")
    $NO_ECC && _cmd+=(--no-ecc)
    $NO_QUAD_WARP && _cmd+=(--no-quadratic-warp)
    $DRY && _cmd+=(--dry)
    $RESUME && _cmd+=(--resume)
    _logdir="$ROOT/scripts/megamaser/logs"
    mkdir -p "$_logdir"
    _ts=$(date +%H%M%S)
    _gals="${GALAXY//,/ }"
    for _gal in $_gals; do
        _sname="watcher_${SAMPLER}_${_gal}${JOB_TAG}_${_ts}"
        launch_detached "$_sname" "$_logdir/${_sname}.log" \
            bash "$_watcher" "${_wargs[@]}" -- "${_cmd[@]}" --galaxy "$_gal"
    done
    exit 0
fi

if [[ "$SAMPLER" == "nss" && -n "$MODE" && "$MODE" != "mode2" ]]; then
    echo "Error: NSS only supports mode2."; exit 1
fi
if [[ "$SAMPLER" == "de" ]]; then
    [[ -n "$F_GRID" ]] && { echo "Error: --f-grid not applicable with --sampler de"; exit 1; }
    [[ "$NUM_CHAINS" != "1" ]] && { echo "Error: --num-chains not applicable with --sampler de"; exit 1; }
    [[ -n "$INIT_METHOD" ]] && { echo "Error: --init-method not applicable with --sampler de"; exit 1; }
    [[ -n "$MODE" && "$MODE" != "mode2" ]] && { echo "Error: DE only supports mode2."; exit 1; }
fi
if [[ -z "$GALAXY" ]]; then
    echo "[ERROR] --galaxy is required. Choices: $ALL_GALS"; exit 1
fi
GALAXY="${GALAXY//,/ }"
for _g in $GALAXY; do
    echo "$ALL_GALS" | grep -qw "$_g" || { echo "Error: unknown galaxy '$_g'. Choices: $ALL_GALS"; exit 1; }
done
if [[ -z "$QUEUE" ]]; then
    echo "[ERROR] -q QUEUE is required (cluster=$CANDEL_CLUSTER)"; exit 1
fi

if [[ "$SAMPLER" == "de" ]]; then
    RUNNER="$ROOT/scripts/megamaser/run_de_map.py"
    JOB_PREFIX="de_map"
    # DE relies on async CUDA allocation; the other samplers do not need
    # these and currently rely on JAX auto-detect. JAX_PLATFORMS=cuda is
    # the correct value (not 'gpu') for JAX 0.8.x.
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export TF_GPU_ALLOCATOR=cuda_malloc_async
    export JAX_PLATFORMS=cuda
else
    RUNNER="$ROOT/scripts/megamaser/run_maser_disk.py"
    JOB_PREFIX="maser"
fi

EXTRA_ARGS=""
[[ -n "$MODE" && "$SAMPLER" != "de" ]]        && EXTRA_ARGS="$EXTRA_ARGS --mode $MODE"
[[ -n "$F_GRID" ]]      && EXTRA_ARGS="$EXTRA_ARGS --f-grid $F_GRID"
[[ -n "$INIT_METHOD" ]] && EXTRA_ARGS="$EXTRA_ARGS --init-method $INIT_METHOD"
$NO_ECC && EXTRA_ARGS="$EXTRA_ARGS --no-ecc"
$NO_QUAD_WARP && EXTRA_ARGS="$EXTRA_ARGS --no-quadratic-warp"
$RESUME && EXTRA_ARGS="$EXTRA_ARGS --resume"

dry_flag=()
[[ "$DRY" == true ]] && dry_flag=(--dry)

for GAL in $GALAXY; do
    echo "Submitting $GAL ($SAMPLER) -> $CANDEL_CLUSTER:$QUEUE"
    case "$SAMPLER" in
        nss)  pycmd="$CANDEL_PYTHON -u $RUNNER $GAL --sampler nss $EXTRA_ARGS" ;;
        nuts) pycmd="$CANDEL_PYTHON -u $RUNNER $GAL --sampler nuts --num-chains $NUM_CHAINS $EXTRA_ARGS" ;;
        de)   pycmd="$CANDEL_PYTHON -u $RUNNER $GAL $EXTRA_ARGS" ;;
    esac
    extra_flags=()
    [[ -n "$CPUS" ]]    && extra_flags+=(--cpus "$CPUS")
    [[ -n "$GPUTYPE" ]] && extra_flags+=(--gputype "$GPUTYPE")
    [[ -n "$GPU_MEM" ]] && extra_flags+=(--gpu-mem "$GPU_MEM")
    [[ -n "$TIME" ]]    && extra_flags+=(--time "$TIME")
    submit_job --gpu --queue "$QUEUE" --mem "$MEM" --name "${JOB_PREFIX}_${GAL}${JOB_TAG}" \
        --logdir "$ROOT/scripts/megamaser/logs" \
        "${extra_flags[@]}" \
        "${dry_flag[@]}" \
        -- $pycmd
done
