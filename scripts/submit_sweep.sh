#!/bin/bash -l
# Submit the standard megamaser MCMC sweep over the five MCP H0 galaxies:
#   {no quadratic warp, +quadratic warp} x {init config, init reid (Pesce)}.
# MCMC and the marginal-objective diagnostic are deliberately separate.
# Thin wrapper around submit.sh: each variant runs the GALAXIES list (below),
# so the 2x2 grid is 4 submit.sh calls -> 4 x len(GALAXIES) single-galaxy jobs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="$ROOT/submit.sh"

# Galaxies to sweep (space-separated). Default: the five MCP H0 galaxies
# (NGC4258 excluded). Edit to add/remove galaxies.
GALAXIES="CGCG074-064 NGC5765b NGC6264 NGC6323 UGC3789"

LOCAL=false
QUEUE=""
CPUS=""
MEM=""
NUM_WARMUP="10000"
NUM_SAMPLES="20000"
NUM_CHAINS="10"
GPU_MEM=""
DRY=false
YES=false
EVIDENCE=false
SKIP_DONE=false
EXTRA=()

usage() {
    cat <<EOF
Usage: $0 (--local | -q QUEUE) [--cpus N] [--mem GB] \\
          [--num-warmup N] [--num-samples N] [--num-chains N] [--evidence] \\
          [--skip-done] [--dry] [-y] [-- extra submit.sh args]

Runs the 2x2 megamaser MCMC sweep over the GALAXIES set at the top of this
script (default: $GALAXIES):

  quadratic warp : off, on        (--add-quadratic-warp)
  init strategy  : config, reid   (reid = published Pesce/Reid globals)

= 4 submit.sh calls, one single-galaxy job per galaxy each.
Default mode adds --compare-reid --sampler mcmc and does not run the
marginal-objective diagnostic. --evidence mode submits only those GPU
diagnostics for existing HDF5 chains; they are not rigorous absolute evidence.

  --local            Run locally (submit.sh --local), one job at a time.
  -q, --queue QUEUE  Submit to the cluster queue (glamdring CPU: redwood|berg|cmb;
                     glamdring GPU: gpulong|cmbgpu|optgpu;
                     arc: short|medium|long).
  --cpus N           Forwarded to submit.sh; glamdring MCMC uses -s -n N.
  --mem GB           Forwarded to submit.sh; omit for the backend-aware
                     default (7 GB per requested CPU).
  --num-warmup N     Forwarded to submit.sh (default 10000).
  --num-samples N    Forwarded to submit.sh (default 20000).
  --num-chains N     Forwarded to submit.sh (default 10). Chains run
                     concurrently up to the configured worker and allocated
                     CPU limits.
  --evidence         Submit marginal-objective diagnostics instead of MCMC.
  --skip-done        Forward submit.sh --skip-done; skip existing MCMC HDF5s.
  --gpu-mem GB       Forwarded to submit.sh --evidence.
  --dry              Forward --dry: print the runner commands without submitting.
  -y, --yes          Do not ask for confirmation before submitting.
  -- extra...        Everything after -- is forwarded verbatim to every submit.sh
                     call (e.g. --time 24, --max-retries 4; with --evidence,
                     --gputype rtx3090with24gb).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local) LOCAL=true; shift ;;
        -q|--queue) QUEUE="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --mem) MEM="$2"; shift 2 ;;
        --num-warmup) NUM_WARMUP="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --num-chains) NUM_CHAINS="$2"; shift 2 ;;
        --evidence) EVIDENCE=true; shift ;;
        --skip-done) SKIP_DONE=true; shift ;;
        --gpu-mem) GPU_MEM="$2"; shift 2 ;;
        --dry) DRY=true; shift ;;
        -y|--yes) YES=true; shift ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ "$LOCAL" == true && -n "$QUEUE" ]]; then
    echo "[ERROR] pass either --local or -q QUEUE, not both"; exit 1
fi
if [[ "$LOCAL" == false && -z "$QUEUE" ]]; then
    echo "[ERROR] pass --local or -q QUEUE"; exit 1
fi
if [[ "$EVIDENCE" == false && -n "$GPU_MEM" ]]; then
    echo "[ERROR] --gpu-mem is only valid with --evidence"; exit 1
fi
if [[ "$EVIDENCE" == true && "$SKIP_DONE" == true ]]; then
    echo "[ERROR] --skip-done is only valid for MCMC sweep mode"; exit 1
fi

if [[ "$LOCAL" == true ]]; then
    target=(--local)
else
    target=(-q "$QUEUE")
fi

# Args shared by every variant.
if [[ "$EVIDENCE" == true ]]; then
    common=(--galaxy "$GALAXIES" --evidence)
    [[ -n "$CPUS" ]] && common+=(--cpus "$CPUS")
    [[ -n "$MEM" ]] && common+=(--mem "$MEM")
    [[ -n "$GPU_MEM" ]] && common+=(--gpu-mem "$GPU_MEM")
else
    common=(--galaxy "$GALAXIES" --sampler mcmc --compare-reid)
    [[ -n "$CPUS" ]] && common+=(--cpus "$CPUS")
    [[ -n "$MEM" ]] && common+=(--mem "$MEM")
    [[ -n "$NUM_WARMUP" ]] && common+=(--num-warmup "$NUM_WARMUP")
    [[ -n "$NUM_SAMPLES" ]] && common+=(--num-samples "$NUM_SAMPLES")
    [[ -n "$NUM_CHAINS" ]] && common+=(--num-chains "$NUM_CHAINS")
    [[ "$SKIP_DONE" == true ]] && common+=(--skip-done)
fi
[[ "$DRY" == true ]] && common+=(--dry)
[[ ${#EXTRA[@]} -gt 0 ]] && common+=("${EXTRA[@]}")

gal_arr=($GALAXIES)
n_gal=${#gal_arr[@]}
mode="MCMC"
[[ "$EVIDENCE" == true ]] && mode="evidence"
echo "[sweep] target: ${target[*]} | $mode"
echo "[sweep] jobs: 2 warp x 2 init x $n_gal galaxies = $((4 * n_gal)) jobs"
echo "[sweep] galaxies: $GALAXIES"
echo "[sweep] will run:"
for warp in "" "--add-quadratic-warp"; do
    for init in config reid; do
        variant=(--init-strategy "$init")
        [[ -n "$warp" ]] && variant+=("$warp")
        printf '  bash %q' "$SUBMIT"
        printf ' %q' "${target[@]}" "${common[@]}" "${variant[@]}"
        printf '\n'
    done
done
if [[ "$DRY" == false && "$YES" == false ]]; then
    printf "[sweep] submit these jobs? [y/N] "
    read -r reply || reply=""
    case "$reply" in
        y|Y|yes|YES) ;;
        *) echo "[sweep] cancelled."; exit 1 ;;
    esac
fi

fail=0
for warp in "" "--add-quadratic-warp"; do
    for init in config reid; do
        variant=(--init-strategy "$init")
        [[ -n "$warp" ]] && variant+=("$warp")
        echo
        echo "[sweep] === warp=${warp:-off} init=$init ==="
        if ! bash "$SUBMIT" "${target[@]}" "${common[@]}" "${variant[@]}"; then
            echo "[sweep] WARNING: variant (warp=${warp:-off} init=$init) failed" >&2
            fail=1
        fi
    done
done

if [[ $fail -ne 0 ]]; then
    echo "[sweep] one or more variants failed" >&2
fi
exit $fail
