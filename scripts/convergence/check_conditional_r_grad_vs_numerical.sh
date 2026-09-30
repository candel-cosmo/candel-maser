#!/bin/bash -l
# Conditional-r AD gradient vs numerical FD validation.
# Submits to a GPU queue on glamdring.

QUEUE="gpulong"
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash $0 [-q QUEUE] [ARGS...]"
            echo ""
            echo "Validate conditional-r AD gradients against numerical FD."
            echo "Tests one spot at a time (low memory)."
            echo ""
            echo "  -q QUEUE   GPU queue (default: gpulong)"
            echo ""
            echo "Python args forwarded as-is:"
            echo "  --galaxy NAME           galaxy (default: NGC5765b)"
            echo "  --small-grids           use reduced grids (faster, for quick checks)"
            echo "  --rtol-fd TOL           AD vs FD tolerance (default: 1e-5)"
            echo "  --rtol-pipeline TOL     full vs separated tolerance (default: 1e-8)"
            echo "  --skip-full-pipeline    skip check B"
            echo ""
            echo "For full Python help:"
            echo "  python -m candel_maser.convergence.check_conditional_r_grad_vs_numerical -h"
            exit 0
            ;;
        -q) QUEUE="$2"; shift 2 ;;
        *) PASS_ARGS+=("$1"); shift ;;
    esac
done

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Core CANDEL checkout (_submit_lib.sh, local_config.toml); defaults
# to a sibling clone of candel-cosmo/CANDEL.
ROOT_DIR="${CANDEL_ROOT:-$(cd "$PKG_ROOT/../CANDEL" 2>/dev/null && pwd)}"
[[ -f "$ROOT_DIR/scripts/_submit_lib.sh" ]] || {
    echo "[ERROR] Set CANDEL_ROOT to the CANDEL core checkout." >&2; exit 1; }
# shellcheck source=../../../../scripts/_submit_lib.sh
source "$ROOT_DIR/scripts/_submit_lib.sh"
if [[ "$CANDEL_CLUSTER" != "glamdring" ]]; then
    echo "[ERROR] This script is glamdring-only (machine=$CANDEL_CLUSTER)" >&2
    exit 1
fi
PYTHON="$CANDEL_PYTHON"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

echo "Submitting check_conditional_r_grad_vs_numerical -> $QUEUE"
echo "JAX: XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda"
echo "Args: ${PASS_ARGS[*]:-(defaults)}"

addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
    $PYTHON -u -m candel_maser.convergence.check_conditional_r_grad_vs_numerical \
    "${PASS_ARGS[@]}"
