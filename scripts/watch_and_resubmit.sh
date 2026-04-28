#!/bin/bash -l
# Job watcher: submit maser jobs, poll until done, auto-resubmit incomplete
# galaxies with --resume. Supports DE MAP, NSS, and NUTS modes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

# ── defaults ────────────────────────────────────────────────────────────────
MAX_RETRIES=5
POLL=120

usage() {
    cat <<'EOF'
Usage: bash watch_and_resubmit.sh [watcher-opts] <de|nss|nuts> <submit-flags...>

Watcher options (before mode):
  --max-retries N   Max resubmit attempts per galaxy (default: 5)
  --poll S          Seconds between squeue polls (default: 120)
  -h, --help        Show this help

Modes:
  de    → run_de_map.sh   (galaxy names as positional args)
  nss   → submit_all.sh --sampler nss  (galaxy via --galaxy GAL)
  nuts  → submit_all.sh --sampler nuts (galaxy via --galaxy GAL)

Everything after the mode word is forwarded to the submit script.

Examples:
  bash watch_and_resubmit.sh de -q cmbgpu
  bash watch_and_resubmit.sh --max-retries 3 nss -q cmbgpu --galaxy NGC5765b
  bash watch_and_resubmit.sh nuts -q optgpu --num-chains 4
EOF
}

# ── parse watcher options ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-retries) MAX_RETRIES="$2"; shift 2 ;;
        --poll)        POLL="$2"; shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        *)             break ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "[watch] Error: mode required (de|nss|nuts)" >&2
    usage >&2
    exit 1
fi

MODE="$1"; shift
case "$MODE" in
    de|nss|nuts) ;;
    *) echo "[watch] Error: unknown mode '$MODE' (choose de|nss|nuts)" >&2; exit 1 ;;
esac

SUBMIT_ARGS=("$@")

# ── helpers ─────────────────────────────────────────────────────────────────

log_path_for_job() {
    # Return the log file path for a given JOBID.
    local jid="$1"
    case "$CANDEL_CLUSTER" in
        glamdring) echo "$ROOT/python-${jid}.out" ;;
        arc)       echo "$PWD/logs-${jid}.out" ;;
        *)         echo "unknown-${jid}.out" ;;
    esac
}

completion_marker() {
    case "$MODE" in
        de)       echo "MAP init" ;;
        nss|nuts) echo "saved samples to" ;;
    esac
}

all_config_galaxies() {
    # Galaxy names from config (used to identify positional args in DE mode).
    "$CANDEL_PYTHON" -c "
import tomli
with open('$ROOT/scripts/megamaser/config_maser.toml', 'rb') as f:
    cfg = tomli.load(f)
print(' '.join(cfg['model']['galaxies']))
"
}

strip_galaxy_args() {
    # Given the SUBMIT_ARGS array, return args with galaxy specifications
    # removed, so we can replace them with only the incomplete galaxies.
    # Result is printed space-separated; caller reads into array.
    local mode="$1"; shift
    local args=("$@")
    if [[ "$mode" == "de" ]]; then
        # DE: galaxy names are positional. Strip any arg that matches a
        # known galaxy name from config.
        local known
        known=$(all_config_galaxies)
        local out=()
        for a in "${args[@]}"; do
            if echo "$known" | grep -qw -- "$a"; then
                continue
            fi
            out+=("$a")
        done
        printf '%s\n' "${out[@]}"
    else
        # NSS/NUTS: strip --galaxy and its value.
        local out=()
        local skip_next=0
        for a in "${args[@]}"; do
            if (( skip_next )); then
                skip_next=0
                continue
            fi
            if [[ "$a" == "--galaxy" ]]; then
                skip_next=1
                continue
            fi
            out+=("$a")
        done
        if [[ ${#out[@]} -gt 0 ]]; then
            printf '%s\n' "${out[@]}"
        fi
    fi
}

submit_and_collect() {
    # Submit jobs for the given galaxies. Populates the global arrays:
    #   JOB_IDS, JOB_GALAXIES  (parallel arrays: jobid <-> galaxy)
    # Arguments: mode, base_args (array without galaxy specs), galaxies...
    local mode="$1"; shift

    # Split: base_args are up to "--", then galaxies follow.
    local base_args=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        base_args+=("$1"); shift
    done
    [[ "${1:-}" == "--" ]] && shift
    local galaxies=("$@")

    JOB_IDS=()
    JOB_GALAXIES=()

    if [[ "$mode" == "de" ]]; then
        # DE: single call with all galaxy names as positional args + --resume.
        local cmd=("$ROOT/scripts/megamaser/run_de_map.sh"
                    "${base_args[@]}" --resume "${galaxies[@]}")
        echo "[watch] Running: ${cmd[*]}"
        _out=$("${cmd[@]}" 2>&1) || true
        echo "$_out"
        # Parse output: one JOBID= per galaxy, interleaved with
        # "Submitting DE MAP: GAL" lines.
        _cur_gal=""
        while IFS= read -r line; do
            if [[ "$line" =~ ^Submitting\ DE\ MAP:\ ([^ ]+) ]]; then
                _cur_gal="${BASH_REMATCH[1]}"
            elif [[ "$line" =~ ^JOBID=([0-9]+)$ ]]; then
                JOB_IDS+=("${BASH_REMATCH[1]}")
                JOB_GALAXIES+=("$_cur_gal")
            fi
        done <<< "$_out"
    else
        # NSS/NUTS: one submit_all.sh call per galaxy.
        for gal in "${galaxies[@]}"; do
            local resume_flag=()
            # NUTS restarts from scratch (no checkpoint support in numpyro).
            [[ "$mode" == "nss" ]] && resume_flag=(--resume)
            local cmd=("$ROOT/scripts/megamaser/submit_all.sh"
                        --sampler "$mode" "${base_args[@]}"
                        --galaxy "$gal" "${resume_flag[@]}")
            echo "[watch] Running: ${cmd[*]}"
            _out=$("${cmd[@]}" 2>&1) || true
            echo "$_out"
            while IFS= read -r line; do
                if [[ "$line" =~ ^JOBID=([0-9]+)$ ]]; then
                    JOB_IDS+=("${BASH_REMATCH[1]}")
                    JOB_GALAXIES+=("$gal")
                fi
            done <<< "$_out"
        done
    fi
}

# ── initial submission (round 0) ───────────────────────────────────────────

# Determine which galaxies are being submitted.
INITIAL_GALAXIES=()
if [[ "$MODE" == "de" ]]; then
    # DE: galaxies are positional args. If none given, run_de_map.sh submits
    # all from config — we need to know which ones.
    _all_gals=$(all_config_galaxies)
    _found=0
    for a in "${SUBMIT_ARGS[@]}"; do
        if echo "$_all_gals" | grep -qw -- "$a"; then
            INITIAL_GALAXIES+=("$a")
            _found=1
        fi
    done
    if (( ! _found )); then
        read -ra INITIAL_GALAXIES <<< "$_all_gals"
    fi
else
    # NSS/NUTS: look for --galaxy in SUBMIT_ARGS.
    _skip=0
    for a in "${SUBMIT_ARGS[@]}"; do
        if (( _skip )); then
            INITIAL_GALAXIES+=("$a")
            _skip=0
            continue
        fi
        [[ "$a" == "--galaxy" ]] && _skip=1
    done
    if [[ ${#INITIAL_GALAXIES[@]} -eq 0 ]]; then
        read -ra INITIAL_GALAXIES <<< "CGCG074-064 NGC5765b NGC6264 NGC6323 UGC3789"
    fi
fi

echo "[watch] Mode: $MODE | Galaxies: ${INITIAL_GALAXIES[*]}"
echo "[watch] Max retries: $MAX_RETRIES | Poll interval: ${POLL}s"
echo "[watch] Cluster: $CANDEL_CLUSTER"
echo ""

# Strip galaxy specifications from SUBMIT_ARGS to get the "base" flags.
readarray -t BASE_ARGS < <(strip_galaxy_args "$MODE" "${SUBMIT_ARGS[@]}")

# Per-galaxy retry counter.
declare -A RETRIES
for gal in "${INITIAL_GALAXIES[@]}"; do
    RETRIES["$gal"]=0
done

# First submission uses original args (no --resume for round 0).
JOB_IDS=()
JOB_GALAXIES=()
echo "=== Round 0: initial submission ==="
if [[ "$MODE" == "de" ]]; then
    cmd=("$ROOT/scripts/megamaser/run_de_map.sh" "${SUBMIT_ARGS[@]}")
    echo "[watch] Running: ${cmd[*]}"
    _out=$("${cmd[@]}" 2>&1) || true
    echo "$_out"
    _cur_gal=""
    while IFS= read -r line; do
        if [[ "$line" =~ ^Submitting\ DE\ MAP:\ ([^ ]+) ]]; then
            _cur_gal="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^JOBID=([0-9]+)$ ]]; then
            JOB_IDS+=("${BASH_REMATCH[1]}")
            JOB_GALAXIES+=("$_cur_gal")
        fi
    done <<< "$_out"
else
    if [[ ${#INITIAL_GALAXIES[@]} -eq 1 ]]; then
        # Single galaxy specified via --galaxy: forward as-is.
        cmd=("$ROOT/scripts/megamaser/submit_all.sh" --sampler "$MODE" "${SUBMIT_ARGS[@]}")
        echo "[watch] Running: ${cmd[*]}"
        _out=$("${cmd[@]}" 2>&1) || true
        echo "$_out"
        while IFS= read -r line; do
            if [[ "$line" =~ ^Submitting\ ([^ ]+)\ \( ]]; then
                _cur_gal="${BASH_REMATCH[1]}"
            elif [[ "$line" =~ ^JOBID=([0-9]+)$ ]]; then
                JOB_IDS+=("${BASH_REMATCH[1]}")
                JOB_GALAXIES+=("${_cur_gal:-${INITIAL_GALAXIES[0]}}")
            fi
        done <<< "$_out"
    else
        # All galaxies: submit_all.sh submits all by default.
        cmd=("$ROOT/scripts/megamaser/submit_all.sh" --sampler "$MODE" "${SUBMIT_ARGS[@]}")
        echo "[watch] Running: ${cmd[*]}"
        _out=$("${cmd[@]}" 2>&1) || true
        echo "$_out"
        _cur_gal=""
        while IFS= read -r line; do
            if [[ "$line" =~ ^Submitting\ ([^ ]+)\ \( ]]; then
                _cur_gal="${BASH_REMATCH[1]}"
            elif [[ "$line" =~ ^JOBID=([0-9]+)$ ]]; then
                JOB_IDS+=("${BASH_REMATCH[1]}")
                JOB_GALAXIES+=("$_cur_gal")
            fi
        done <<< "$_out"
    fi
fi

if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
    echo "[watch] No jobs were submitted (dry run or error). Exiting."
    exit 0
fi

echo ""
echo "[watch] Submitted ${#JOB_IDS[@]} job(s):"
for i in "${!JOB_IDS[@]}"; do
    echo "  ${JOB_GALAXIES[$i]} -> JOBID=${JOB_IDS[$i]}"
done

# ── poll / resubmit loop ───────────────────────────────────────────────────
ROUND=0
MARKER=$(completion_marker)

while true; do
    echo ""
    echo "[watch] Polling (round $ROUND) — waiting for ${#JOB_IDS[@]} job(s)..."

    # Wait until all jobs in this round have left the queue.
    while true; do
        still_running=0
        for jid in "${JOB_IDS[@]}"; do
            if squeue -j "$jid" -h 2>/dev/null | grep -q "$jid"; then
                still_running=1
                break
            fi
        done
        if (( ! still_running )); then
            break
        fi
        echo "[watch] $(date '+%H:%M:%S') — jobs still running, sleeping ${POLL}s"
        sleep "$POLL"
    done

    echo "[watch] All jobs in round $ROUND have finished."

    # Check logs for completion marker.
    incomplete=()
    for i in "${!JOB_IDS[@]}"; do
        jid="${JOB_IDS[$i]}"
        gal="${JOB_GALAXIES[$i]}"
        logfile=$(log_path_for_job "$jid")
        if [[ -f "$logfile" ]] && grep -q "$MARKER" "$logfile"; then
            echo "[watch] $gal (job $jid): COMPLETE"
        else
            echo "[watch] $gal (job $jid): INCOMPLETE (marker '$MARKER' not found in $logfile)"
            incomplete+=("$gal")
        fi
    done

    if [[ ${#incomplete[@]} -eq 0 ]]; then
        echo ""
        echo "[watch] All galaxies completed successfully!"
        exit 0
    fi

    # Check retry limits.
    resubmit=()
    for gal in "${incomplete[@]}"; do
        count=${RETRIES["$gal"]}
        if (( count >= MAX_RETRIES )); then
            echo "[watch] $gal: exceeded max retries ($MAX_RETRIES). Giving up."
        else
            RETRIES["$gal"]=$(( count + 1 ))
            resubmit+=("$gal")
            echo "[watch] $gal: retry ${RETRIES[$gal]}/$MAX_RETRIES"
        fi
    done

    if [[ ${#resubmit[@]} -eq 0 ]]; then
        echo ""
        echo "[watch] No galaxies left to resubmit. Exiting."
        exit 1
    fi

    ROUND=$(( ROUND + 1 ))
    echo ""
    echo "=== Round $ROUND: resubmitting ${#resubmit[@]} galaxy(ies): ${resubmit[*]} ==="

    JOB_IDS=()
    JOB_GALAXIES=()
    submit_and_collect "$MODE" "${BASE_ARGS[@]}" -- "${resubmit[@]}"

    if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
        echo "[watch] Resubmission produced no jobs. Exiting."
        exit 1
    fi

    echo ""
    echo "[watch] Resubmitted ${#JOB_IDS[@]} job(s):"
    for i in "${!JOB_IDS[@]}"; do
        echo "  ${JOB_GALAXIES[$i]} -> JOBID=${JOB_IDS[$i]}"
    done
done
