#!/bin/bash -l
# Submit BlackJAX Gibbs sampler or DE MAP megamaser jobs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

QUEUE=""
GALAXY=""
SAMPLER="gibbs"
MEM=16
CPUS=""
GPUTYPE=""
GPU_MEM=""
TIME=""
DRY=false
LOCAL=false

EXTRA_ARGS=()
ALL_GALS="CGCG074-064 NGC4258 NGC5765b NGC6264 NGC6323 UGC3789"

usage() {
    cat <<EOF
Usage: $0 (--local | -q QUEUE) --galaxy GAL[,GAL,...] [--sampler gibbs|de] [options]

Required:
  --local                Run in the current terminal instead of submitting
                         to a batch backend.
  -q, --queue QUEUE      Queue/partition for batch submission
                         (glamdring: gpulong|cmbgpu|optgpu;
                         arc: short|medium|long).
  --galaxy GAL[,GAL,...] Galaxy/galaxies to submit.
                         Choices: $ALL_GALS
  --sampler gibbs|de     Job type (default: gibbs).

Gibbs options passed to run_maser.py:
  --num-warmup N
  --num-samples N
  --n-inner N
  --seed N
  --spot-batch N
  --n-sys N
  --n-red N
  --n-blue N
  --target-accept-nuts X
  --target-accept-r X
  --initial-step-size X
  --max-tree-depth N
  --diagonal-mass
  --no-progress
  --no-jit-steps
  --no-ecc | --add-ecc
  --no-quadratic-warp | --add-quadratic-warp
  --f64

DE options passed to run_maser.py --sampler de:
  --resume
  --checkpoint-interval-minutes M
  --log2-N N
  --pop-size N
  --max-generations N
  --patience N
  --eval-chunk N

Cluster options:
  --cpus N
  --gputype TYPE
  --gpu-mem GB
  --time T
  --mem GB              Memory in GB (default: $MEM)
  --dry                 Print submit command without submitting.
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -q|--queue) QUEUE="$2"; shift 2 ;;
        --sampler) SAMPLER="$2"; shift 2 ;;
        --galaxy) GALAXY="$2"; shift 2 ;;
        --mem) MEM="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --gputype) GPUTYPE="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --local) LOCAL=true; shift ;;
        --dry) DRY=true; shift ;;
        --f64) EXTRA_ARGS+=("--f64"); shift ;;
        --num-warmup|--num-samples|--n-inner|--seed|--spot-batch|--n-sys|--n-red|--n-blue|--target-accept-nuts|--target-accept-r|--initial-step-size|--max-tree-depth|--checkpoint-interval-minutes|--log2-N|--pop-size|--max-generations|--patience|--eval-chunk)
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --diagonal-mass|--no-progress|--no-jit-steps|--no-ecc|--add-ecc|--no-quadratic-warp|--add-quadratic-warp|--resume)
            EXTRA_ARGS+=("$1"); shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$GALAXY" ]]; then
    echo "[ERROR] --galaxy is required. Choices: $ALL_GALS"; exit 1
fi
if [[ "$SAMPLER" != "gibbs" && "$SAMPLER" != "de" ]]; then
    echo "[ERROR] --sampler must be gibbs or de"; exit 1
fi
if [[ "$LOCAL" == false && -z "$QUEUE" ]]; then
    echo "[ERROR] pass --local or -q QUEUE (cluster=$CANDEL_CLUSTER)"; exit 1
fi

GALAXY="${GALAXY//,/ }"
for gal in $GALAXY; do
    echo "$ALL_GALS" | grep -qw "$gal" || {
        echo "Error: unknown galaxy '$gal'. Choices: $ALL_GALS"; exit 1;
    }
done

RUNNER="$ROOT/scripts/megamaser/run_maser.py"
if [[ "$SAMPLER" == "de" ]]; then
    JOB_PREFIX="maser_de"
else
    JOB_PREFIX="maser_gibbs"
fi
dry_flag=()
[[ "$DRY" == true ]] && dry_flag=(--dry)

extra_flags=()
[[ -n "$CPUS" ]] && extra_flags+=(--cpus "$CPUS")
[[ -n "$GPUTYPE" ]] && extra_flags+=(--gputype "$GPUTYPE")
[[ -n "$GPU_MEM" ]] && extra_flags+=(--gpu-mem "$GPU_MEM")
[[ -n "$TIME" ]] && extra_flags+=(--time "$TIME")

for gal in $GALAXY; do
    if [[ "$LOCAL" == true ]]; then
        echo "Running $gal ($SAMPLER) locally"
        cmd=("$CANDEL_PYTHON" -u "$RUNNER" "$gal" --sampler "$SAMPLER" "${EXTRA_ARGS[@]}")
        if [[ "$DRY" == true ]]; then
            printf '[dry]'; printf ' %q' "${cmd[@]}"; printf '\n'
        else
            "${cmd[@]}"
        fi
        continue
    fi

    echo "Submitting $gal ($SAMPLER) -> $CANDEL_CLUSTER:$QUEUE"
    pycmd="$CANDEL_PYTHON -u $RUNNER $gal --sampler $SAMPLER ${EXTRA_ARGS[*]}"
    submit_args=(--gpu --queue "$QUEUE" --mem "$MEM"
                 --name "${JOB_PREFIX}_${gal}")
    if [[ ${#extra_flags[@]} -gt 0 ]]; then
        submit_args+=("${extra_flags[@]}")
    fi
    if [[ ${#dry_flag[@]} -gt 0 ]]; then
        submit_args+=("${dry_flag[@]}")
    fi
    submit_job "${submit_args[@]}" -- $pycmd
done
