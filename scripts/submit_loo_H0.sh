#!/bin/bash -l
# Leave-one-out joint H0 with redshift selection + Manticore-Local reconstruction.
# One joint NUTS chain per dropped MCP galaxy (5 jobs); the dropped galaxy shows
# up in each output filename. Thin wrapper around submit_sweep_H0.sh so it stays
# in sync with the sweep.
#
#   ./scripts/megamaser/submit_loo_H0.sh -q gpulong --cpus 2
#
# All args (queue, --cpus, --dry, -y, --num-samples, -- extra ...) are forwarded.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/submit_sweep_H0.sh" \
    --leave-one-out --selection redshift --reconstruction ManticoreLocalCOLA "$@"
