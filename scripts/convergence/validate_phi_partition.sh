#!/bin/bash -l
# Submit the phi-integration validation suite on one Glamdring GPU node.

QUEUE="cmbgpu"
MEMORY=16
GPUS=1
LOCAL_RUN=false
CLEAN_CACHE=false
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash $0 [--local] [-q QUEUE] [--gpus N] [--mem GB] [ARGS...]"
            echo ""
            echo "Submits validate_phi_partition.py, or runs it directly with"
            echo "--local. Defaults: all galaxies and 4 Sobol points; pass"
            echo "--galaxies NAME to restrict it. Other args are forwarded."
            echo ""
            echo "--local runs directly on this machine with the full"
            echo "production reference grids and tolerances (expect on the"
            echo "order of an hour or more per galaxy on CPU)."
            echo "Pass --clean-cache to delete cached references and exit."
            echo "Repeat --scheme-setting METHOD.KEY=VALUE for numerical"
            echo "experiments, e.g. peak-partition.n_phi_partition_sys=257."
            echo "See scripts/megamaser/README.md for the full whitelist."
            exit 0
            ;;
        --local) LOCAL_RUN=true; shift ;;
        --clean-cache)
            CLEAN_CACHE=true
            PASS_ARGS+=("$1")
            shift
            ;;
        -q) QUEUE="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --mem) MEMORY="$2"; shift 2 ;;
        --) shift; PASS_ARGS+=("$@"); break ;;
        *) PASS_ARGS+=("$1"); shift ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=../../_submit_lib.sh
source "$ROOT_DIR/scripts/_submit_lib.sh"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ "$CLEAN_CACHE" == true ]]; then
    exec "$CANDEL_PYTHON" -u \
        "$ROOT_DIR/scripts/megamaser/convergence/validate_phi_partition.py" \
        "${PASS_ARGS[@]}"
fi

if [[ "$LOCAL_RUN" == true ]]; then
    if [[ "$CANDEL_CLUSTER" != "local" ]]; then
        echo "[ERROR] --local requires machine=local (machine=$CANDEL_CLUSTER)" >&2
        exit 1
    fi
    echo "Running phi-integration validation locally (production settings)"
    echo "Args: ${PASS_ARGS[*]:-(defaults)}"
    exec "$CANDEL_PYTHON" -u \
        "$ROOT_DIR/scripts/megamaser/convergence/validate_phi_partition.py" \
        --allow-cpu "${PASS_ARGS[@]}"
fi

if [[ "$CANDEL_CLUSTER" != "glamdring" ]]; then
    echo "[ERROR] This script is glamdring-only (machine=$CANDEL_CLUSTER)" >&2
    exit 1
fi

export JAX_PLATFORMS=cuda

echo "Submitting phi-integration validation -> $QUEUE ($GPUS GPU)"
echo "Args: ${PASS_ARGS[*]:-(defaults)}"
addqueue -q "$QUEUE" -s -m "$MEMORY" --gpus "$GPUS" \
    "$CANDEL_PYTHON" -u \
    "$ROOT_DIR/scripts/megamaser/convergence/validate_phi_partition.py" \
    --n-devices "$GPUS" "${PASS_ARGS[@]}"
