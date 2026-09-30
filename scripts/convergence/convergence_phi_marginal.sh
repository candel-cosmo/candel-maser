#!/bin/bash -l
# Fixed-r brute-force phi convergence test for all six MCP galaxies
# including NGC4258. Sweeps per-sub-range phi grid sizes against a
# high-resolution float64 reference AND compares configured per-type
# phi ranges against a single full-2π integration.

QUEUE="gpulong"
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash $0 [-q QUEUE] [ARGS...]"
            echo ""
            echo "Tests the sub-range phi marginaliser against a"
            echo "float64 full-2π reference. Per (galaxy, grid, scale) prints three"
            echo "narrow tables:"
            echo "  1. Δ logL per category (total, sys, red, blue)."
            echo "  2. rel. diff of AD ∇globals (max + worst param)."
            echo "  3. rel. diff of AD ∇r_ang per spot category (sys/red/blue)."
            echo "Then a fiducial driver diagnostic bumps n_hv_high and"
            echo "n_hv_low one at a time and reports which dimension limits"
            echo "the production grid."
            echo ""
            echo "Default: runs on all six galaxies (CGCG, NGC5765b, NGC6264,"
            echo "  NGC6323, UGC3789, NGC4258). Pass --galaxies <names> to"
            echo "  override."
            echo ""
            echo "  -q QUEUE      GPU queue (default: gpulong)"
            echo ""
            echo "Common Python toggles (forwarded as-is):"
            echo "  --dataset NAME          original_published, fiducial, or unpruned"
            echo "  --driver-factor N       multiplier for the fiducial driver"
            echo "                          diagnostic (default: 2)"
            echo "  --grid-factors F...     also test scaled production grids,"
            echo "                          e.g. 0.25 0.5 1 2"
            echo "  --no-grad               skip the AD-gradient checks"
            echo "  --grad-rtol-globals X   pass tolerance on rel. ∇globals diff"
            echo "  --grad-rtol-r X         pass tolerance on rel. ∇r_ang diff"
            echo ""
            echo "Remaining args are passed to convergence_phi_marginal.py. For Python help:"
            echo "  python -m candel_maser.convergence.convergence_phi_marginal --help"
            exit 0
            ;;
        -q) QUEUE="$2"; shift 2 ;;
        *) PASS_ARGS+=("$1"); shift ;;
    esac
done

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Core CANDEL checkout (_submit_lib.sh, local_config.toml); defaults
# to a sibling clone of candel-cosmo/CANDEL.
ROOT_DIR="${CANDEL_ROOT:-$(cd "$PKG_ROOT/../candel" 2>/dev/null && pwd)}"
[[ -f "$ROOT_DIR/scripts/_submit_lib.sh" ]] || {
    echo "[ERROR] Set CANDEL_ROOT to the CANDEL core checkout." >&2; exit 1; }
# shellcheck source=../../../../scripts/_submit_lib.sh
source "$ROOT_DIR/scripts/_submit_lib.sh"
if [[ "$CANDEL_CLUSTER" != "glamdring" ]]; then
    echo "[ERROR] This script is glamdring-only (machine=$CANDEL_CLUSTER)" >&2
    exit 1
fi
PYTHON="$CANDEL_PYTHON"

# Default: test all six galaxies, including NGC4258.
# Override by passing --galaxies ... on the command line.
DEFAULT_GALAXIES=(CGCG074-064 NGC5765b NGC6264 NGC6323 UGC3789 NGC4258)

if [[ ${#PASS_ARGS[@]} -eq 0 ]]; then
    SCRIPT_ARGS=(--galaxies "${DEFAULT_GALAXIES[@]}")
else
    SCRIPT_ARGS=("${PASS_ARGS[@]}")
fi

echo "Submitting convergence_phi_marginal -> $QUEUE"
echo "Galaxies/args: ${SCRIPT_ARGS[*]}"

addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
    $PYTHON -u -m candel_maser.convergence.convergence_phi_marginal \
    "${SCRIPT_ARGS[@]}"
