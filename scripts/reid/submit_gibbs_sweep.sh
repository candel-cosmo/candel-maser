#!/bin/bash -l
# Submit the fit_disk_Reid_reflection variant comparison for a list of
# galaxies x both start points (config and pesce). Each (galaxy, init) is one
# submit_gibbs_comparison.sh run -- which itself fans out one job per variant
# plus a dependent collect job -- with its OWN --out-dir, so config and pesce
# never share chain directories.
#
# Usage:
#   submit_gibbs_sweep.sh -q QUEUE [options] [-- extra submit_gibbs args]
#
# Example (matches the single-run command, swept over both galaxies/inits):
#   submit_gibbs_sweep.sh -q berg --chains 12
set -euo pipefail

ROOT="${CANDEL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
SUBMIT="$ROOT/packages/candel-maser/scripts/reid/submit_gibbs_comparison.sh"
BASE_OUTPUT="$ROOT/results/Megamaser/reid_mcmc"

QUEUE=""
CHAINS=12
DATASET=""
# GALAXIES="NGC6323 NGC6264 NGC5765b UGC3789 CGCG074-064"
GALAXIES="NGC6323 NGC6264 UGC3789"
INITS="config pesce"
# INITS="pesce"
H0_RANGE=""
MATCH_PRIORS=""
DRY=false
EXTRA=()

usage() {
    cat <<EOF
Usage: bash $0 -q QUEUE [options] [-- extra args forwarded to submit_gibbs_comparison.sh]

Runs submit_gibbs_comparison.sh for every galaxy in --galaxies and every start
point in --inits. The whole sweep goes under one parent folder
  $BASE_OUTPUT/gibbs_sweep_<stamp>/
with one subdir per combination (<galaxy>_<init>), so config and pesce never
share chain directories and the sweep stays self-contained.

Required:
  -q, --queue QUEUE       Queue/partition

Options:
  --chains N              Chains per variant / CPUs per variant job (default: $CHAINS)
  --dataset NAME          Spot-table dataset (default: config [io].dataset)
  --galaxies "G1 G2 .."   Space-separated galaxy list (default: "$GALAXIES")
  --inits "config pesce"  Space-separated start points (default: "$INITS")
  --output-dir DIR        Output root (default: $BASE_OUTPUT)
  --H0-range LOW,HIGH     Forwarded to submit_gibbs_comparison.sh: restrict
                          the Ho sampling window (see its --help for the
                          +-10 km/s hard-bound caveat)
  --match-priors T|F|orig Forwarded to submit_gibbs_comparison.sh: error-floor
                          treatment for ALL variants (T=CANDEL Gaussian sampled;
                          F=template verbatim; orig=Reid/Kuo fixed floors, not
                          sampled -- Kuo 2015 reproduction) (default: submit's T)
  --dry                   Forward --dry: print the submit commands, submit nothing
  -h, --help

Each galaxy needs the selected dataset's control file (both inits use it for
priors/steps; pesce only overwrites its value column). The collect step also
needs the matching CANDEL posterior HDF5 for the overlay -- pass --candel via
-- ... if it is nonstandard.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -q|--queue) QUEUE="$2"; shift 2 ;;
        --chains|--cpus) CHAINS="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --galaxies) GALAXIES="$2"; shift 2 ;;
        --inits) INITS="$2"; shift 2 ;;
        --output-dir) BASE_OUTPUT="$2"; shift 2 ;;
        --H0-range) H0_RANGE="$2"; shift 2 ;;
        --match-priors) MATCH_PRIORS="$2"; shift 2 ;;
        --dry) DRY=true; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "[ERROR] unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

[[ -n "$QUEUE" ]] || { echo "[ERROR] -q QUEUE is required" >&2; exit 1; }
if [[ -n "$DATASET" && "$DATASET" != "original_published" && "$DATASET" != "fiducial" ]]; then
    echo "[ERROR] --dataset must be original_published or fiducial" >&2
    exit 1
fi

stamp="$(date +%Y%m%d_%H%M%S)"
# Self-document the batch folder by floor mode (matching the per-combo
# artifact / job-name tags), so an orig/F sweep is not confused with a
# default-T sweep -- the timestamp alone does not say which was submitted.
case "$MATCH_PRIORS" in
    orig) mp_tag="_origfloor" ;;
    F)    mp_tag="_flatfloor" ;;
    *)    mp_tag="" ;;
esac
sweep_dir="$BASE_OUTPUT/gibbs_sweep_${stamp}${mp_tag}"
echo "Sweep output root: $sweep_dir"
failures=()
for galaxy in $GALAXIES; do
    for init in $INITS; do
        out_dir="$sweep_dir/${galaxy}_${init}"
        cmd=("$SUBMIT" -q "$QUEUE" --chains "$CHAINS" --galaxy "$galaxy"
             --init "$init" --out-dir "$out_dir")
        [[ -n "$DATASET" ]] && cmd+=(--dataset "$DATASET")
        [[ -n "$H0_RANGE" ]] && cmd+=(--H0-range "$H0_RANGE")
        [[ -n "$MATCH_PRIORS" ]] && cmd+=(--match-priors "$MATCH_PRIORS")
        [[ ${#EXTRA[@]} -gt 0 ]] && cmd+=("${EXTRA[@]}")
        $DRY && cmd+=(--dry)
        echo "=================================================================="
        echo "$galaxy / $init -> $out_dir"
        echo "+ ${cmd[*]}"
        # Don't let one bad combination (e.g. a missing control template) abort
        # the rest of the sweep; collect failures and report at the end.
        if ! "${cmd[@]}"; then
            echo "[WARN] submit failed for $galaxy / $init" >&2
            failures+=("$galaxy/$init")
        fi
    done
done

echo
echo "Once all collect jobs finish, consolidate every combo's distance into"
echo "one sweep-level table (median, asymmetric 1sigma, R-hat) with:"
echo "  $ROOT/venv_candel/bin/python \\"
echo "    -m candel_maser.reid.aggregate_sweep_distances \\"
echo "    $sweep_dir"

if [[ ${#failures[@]} -gt 0 ]]; then
    echo "[ERROR] ${#failures[@]} combination(s) failed: ${failures[*]}" >&2
    exit 1
fi
echo "All submissions done."
