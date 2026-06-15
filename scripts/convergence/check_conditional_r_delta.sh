#!/bin/bash -l
# Per-spot conditional-r Δll diagnostic. Screens all spots via a cheap
# brute-force, then runs full-res on the worst N and plots their
# 1D r posteriors.

QUEUE="optgpu"
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash $0 [-q QUEUE] [ARGS...]"
            echo ""
            echo "Per-spot conditional-r convergence diagnostic. For each galaxy:"
            echo "  1. Screen all spots (production vs 5k×5k brute-force)."
            echo "  2. Full-res brute-force (50k×50k) on the worst N spots."
            echo "  3. Plot zoomed r posteriors (prod phi vs full 2pi phi)."
            echo ""
            echo "Default: all MCP galaxies. Override with --galaxies."
            echo ""
            echo "  -q QUEUE      GPU queue (default: optgpu)"
            echo ""
            echo "Common Python toggles (forwarded as-is):"
            echo "  --galaxies G [G ...]  subset to run"
            echo "      choices: CGCG074-064 NGC4258 NGC5765b NGC6264 NGC6323 UGC3789"
            echo "      (default: all configured galaxies)"
            echo "  --n-worst N           how many worst spots to plot (default: 12)"
            echo "  --spot-batch N        spot-axis chunk (default: 4)"
            echo ""
            echo "Examples:"
            echo "  bash $0 --galaxies NGC4258"
            echo "  bash $0 --galaxies NGC5765b UGC3789 --n-worst 6"
            echo "  bash $0                              # all MCP galaxies"
            echo ""
            echo "For full Python help:"
            echo "  python scripts/megamaser/convergence/check_conditional_r_delta.py --help"
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

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

echo "Submitting check_conditional_r_delta -> $QUEUE"
echo "JAX: XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda"
echo "Args: ${PASS_ARGS[*]:-(defaults)}"

addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
    $PYTHON -u "$ROOT_DIR/scripts/megamaser/convergence/check_conditional_r_delta.py" \
    "${PASS_ARGS[@]}"
