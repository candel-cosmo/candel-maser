#!/bin/bash -l
# Per-spot 1D r_ang posterior (phi marginalised, globals pinned to config
# init) for two representative galaxies. Diagnostic — no sampling.

QUEUE="gpulong"
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash $0 [-q QUEUE] [ARGS...]"
            echo ""
            echo "Computes p(r_ang | d_i, globals fixed to config init) for"
            echo "every spot, marginalising phi with the Mode 1 production"
            echo "grid. Produces one overlay plot per galaxy (rows: sys/red/"
            echo "blue; columns: galaxies) plus an npz of the raw curves."
            echo ""
            echo "Default galaxies: UGC3789, NGC6323. Pass --galaxies <names>"
            echo "to override, or --include-ngc4258 to append NGC4258 (heavy)."
            echo ""
            echo "  -q QUEUE      GPU queue (default: gpulong)"
            echo ""
            echo "Remaining args are passed to r_ang_posteriors.py. Common:"
            echo "  --f-grid F    scale phi grid sizes (default 1.0); 2 needs more GPU mem"
            echo "  --r-batch N   r-axis chunk size (default 64); lower on OOM (try 8 or 4)"
            echo "For Python help:"
            echo "  python scripts/megamaser/convergence/r_ang_posteriors.py --help"
            exit 0
            ;;
        -q) QUEUE="$2"; shift 2 ;;
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
PYTHON="$CANDEL_PYTHON"

echo "Submitting r_ang_posteriors -> $QUEUE"
echo "Args: ${PASS_ARGS[*]:-(defaults)}"

addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
    $PYTHON -u "$ROOT_DIR/scripts/megamaser/convergence/r_ang_posteriors.py" \
    "${PASS_ARGS[@]}"
