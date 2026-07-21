#!/bin/bash -l
# Submit the peak-partition validation suite on one Glamdring GPU node.

QUEUE="cmbgpu"
MEMORY=16
GPUS=1
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash $0 [-q QUEUE] [--gpus N] [--mem GB] [ARGS...]"
            echo ""
            echo "Submits validate_phi_partition.py; remaining arguments are"
            echo "forwarded unchanged. The Python command requires a GPU unless"
            echo "--allow-cpu is passed directly for local development."
            exit 0
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
if [[ "$CANDEL_CLUSTER" != "glamdring" ]]; then
    echo "[ERROR] This script is glamdring-only (machine=$CANDEL_CLUSTER)" >&2
    exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

echo "Submitting peak-partition validation -> $QUEUE ($GPUS GPU)"
echo "Args: ${PASS_ARGS[*]:-(defaults)}"
addqueue -q "$QUEUE" -s -m "$MEMORY" --gpus "$GPUS" \
    "$CANDEL_PYTHON" -u \
    "$ROOT_DIR/scripts/megamaser/convergence/validate_phi_partition.py" \
    --n-devices "$GPUS" "${PASS_ARGS[@]}"
