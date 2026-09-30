#!/bin/bash -l
# Generic job watcher: run a submit command, capture JOBIDs, poll squeue
# until all jobs finish, check logs for a completion marker, and resubmit
# the same command with --resume if any job is incomplete.
#
# Works on glamdring (addqueue) and arc (sbatch).
#
# Usage:
#   bash watch_and_resubmit.sh [opts] -- <submit-command...>
#
# Examples:
#   bash watch_and_resubmit.sh --marker "MAP init" -- \
#       bash scripts/submit.sh --sampler de -q cmbgpu --galaxy NGC5765b
#
#   bash watch_and_resubmit.sh --marker "saved samples to" --no-resume -- \
#       bash scripts/submit.sh --sampler mcmc -q cmbgpu --galaxy NGC5765b
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Core CANDEL checkout (_submit_lib.sh, local_config.toml); defaults
# to a sibling clone of candel-cosmo/CANDEL.
ROOT="${CANDEL_ROOT:-$(cd "$PKG_ROOT/../candel" 2>/dev/null && pwd)}"
[[ -f "$ROOT/scripts/_submit_lib.sh" ]] || {
    echo "[ERROR] Set CANDEL_ROOT to the CANDEL core checkout." >&2; exit 1; }
# shellcheck source=../../../scripts/_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

MAX_RETRIES=5
POLL=120
MARKER=""
RESUME_FLAG="--resume"

usage() {
    cat <<'EOF'
Usage: bash watch_and_resubmit.sh [options] -- <submit-command...>

Options:
  --marker STRING     Completion marker to grep for in job logs (REQUIRED)
  --max-retries N     Max resubmit rounds (default: 5)
  --poll S            Seconds between squeue polls (default: 120)
  --resume-flag FLAG  Flag appended on resubmit (default: --resume)
  --no-resume         Don't append any resume flag on resubmit
  -h, --help          Show this help

The submit command must produce JOBID=<id> lines on stdout (provided
by _submit_lib.sh's submit_job function).

On resubmit, the same command is re-run with the resume flag appended unless
--no-resume was passed.
For megamaser submit.sh commands with multiple galaxies, retry rounds
resubmit only the galaxies whose latest jobs missed the marker.

Examples:
  # DE MAP (one galaxy, auto-resume on timeout)
  bash watch_and_resubmit.sh --marker "MAP init" -- \
      bash scripts/submit.sh --sampler de -q cmbgpu --galaxy NGC5765b

  # DE MAP (multiple galaxies)
  bash watch_and_resubmit.sh --marker "MAP init" -- \
      bash scripts/submit.sh --sampler de -q cmbgpu --galaxy NGC5765b,NGC6264

  # DE MAP (all MCP H0 galaxies; excludes NGC4258)
  bash watch_and_resubmit.sh --marker "MAP init" -- \
      bash scripts/submit.sh --sampler de -q cmbgpu --galaxy all

  # MCMC sampler (one galaxy; no resume flag exists)
  bash watch_and_resubmit.sh --marker "saved samples to" --no-resume -- \
      bash scripts/submit.sh --sampler mcmc -q cmbgpu --galaxy NGC5765b

  # Custom poll and retries
  bash watch_and_resubmit.sh --marker "MAP init" --max-retries 10 --poll 60 -- \
      bash scripts/submit.sh --sampler de -q cmbgpu --galaxy NGC5765b
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --marker)       MARKER="$2"; shift 2 ;;
        --max-retries)  MAX_RETRIES="$2"; shift 2 ;;
        --poll)         POLL="$2"; shift 2 ;;
        --resume-flag)  RESUME_FLAG="$2"; shift 2 ;;
        --no-resume)    RESUME_FLAG=""; shift ;;
        -h|--help)      usage; exit 0 ;;
        --)             shift; break ;;
        *)              echo "[watch] Unknown option: $1 (use -- before the command)" >&2
                        exit 1 ;;
    esac
done

if [[ -z "$MARKER" ]]; then
    echo "[watch] Error: --marker is required" >&2; exit 1
fi
if [[ $# -eq 0 ]]; then
    echo "[watch] Error: no command given after --" >&2; exit 1
fi

CMD=("$@")

# ── helpers ────────────────────────────────────────────────────────────────

log_path_for_job() {
    local jid="$1" template="${2:-}"
    if [[ -n "$template" ]]; then
        template="${template//<jobid>/$jid}"
        template="${template//%j/$jid}"
        echo "$template"
        return
    fi
    # Default submit_job logs are "logs-<jobid>-<jobname>.out" in $PWD.
    local f
    f=$(ls "$PWD"/logs-"${jid}"-*.out 2>/dev/null | head -1)
    echo "${f:-$PWD/logs-${jid}-<name>.out}"
}

run_and_capture_jobids() {
    # Run the command, tee output to the terminal, collect JOBID= lines.
    local cmd=("$@")
    local current_gal="" current_log=""
    job_ids=()
    job_gals=()
    job_logs=()
    echo "[watch] Running: ${cmd[*]}"
    _out=$("${cmd[@]}" 2>&1) || true
    echo "$_out"
    while IFS= read -r line; do
        if [[ "$line" =~ ^(Submitting|Running)[[:space:]]+([^[:space:]]+)[[:space:]]+\( ]]; then
            current_gal="${BASH_REMATCH[2]}"
        fi
        if [[ "$line" =~ ^\[submit_job\][[:space:]]+log[[:space:]]*:[[:space:]]*(.*)$ ]]; then
            current_log="${BASH_REMATCH[1]}"
        fi
        if [[ "$line" =~ ^JOBID=([0-9]+)$ ]]; then
            job_ids+=("${BASH_REMATCH[1]}")
            job_gals+=("$current_gal")
            job_logs+=("$current_log")
            current_gal=""
            current_log=""
        fi
    done <<< "$_out"
}

wait_for_jobs() {
    local jids=("$@")
    [[ ${#jids[@]} -eq 0 ]] && return 0
    echo "[watch] Waiting for ${#jids[@]} job(s): ${jids[*]}"
    while true; do
        still_running=0
        for jid in "${jids[@]}"; do
            if squeue -j "$jid" -h 2>/dev/null | grep -q "$jid"; then
                still_running=$((still_running + 1))
            fi
        done
        [[ $still_running -eq 0 ]] && break
        echo "[watch] $(date '+%H:%M:%S') — $still_running job(s) still running"
        sleep "$POLL"
    done
    echo "[watch] All jobs finished."
}

check_jobs() {
    # Check completion for a list of job IDs. Returns 0 if all complete,
    # 1 if any incomplete. Prints status for each.
    local jids=("${job_ids[@]}")
    local any_incomplete=0
    incomplete_gals=()
    for i in "${!jids[@]}"; do
        local jid="${jids[$i]}"
        local gal="${job_gals[$i]:-}"
        logfile=$(log_path_for_job "$jid" "${job_logs[$i]:-}")
        if [[ -f "$logfile" ]] && grep -q "$MARKER" "$logfile"; then
            echo "[watch] Job $jid: COMPLETE"
        else
            echo "[watch] Job $jid: INCOMPLETE (log: ${logfile})"
            [[ -n "$gal" ]] && incomplete_gals+=("$gal")
            any_incomplete=1
        fi
    done
    return $any_incomplete
}

replace_galaxy_arg() {
    local galaxies="$1"; shift
    local cmd=("$@")
    local replaced=0
    resubmit_cmd=()
    for ((i = 0; i < ${#cmd[@]}; i++)); do
        case "${cmd[$i]}" in
            --galaxy)
                resubmit_cmd+=("--galaxy" "$galaxies")
                i=$((i + 1))
                replaced=1
                ;;
            --galaxy=*)
                resubmit_cmd+=("--galaxy=$galaxies")
                replaced=1
                ;;
            *)
                resubmit_cmd+=("${cmd[$i]}")
                ;;
        esac
    done
    if [[ $replaced -eq 0 ]]; then
        resubmit_cmd=("${cmd[@]}")
    fi
}

# ── main loop ──────────────────────────────────────────────────────────────

echo "[watch] Marker: '$MARKER'"
echo "[watch] Max retries: $MAX_RETRIES | Poll: ${POLL}s"
echo "[watch] Resume flag: ${RESUME_FLAG:-(none)}"
echo "[watch] Cluster: $CANDEL_CLUSTER"
echo "[watch] Host: $(hostname -f 2>/dev/null || hostname)"
echo "[watch] PWD: $PWD"
echo "[watch] Command: ${CMD[*]}"
echo ""

resubmit_cmd=("${CMD[@]}")

for attempt in $(seq 0 "$MAX_RETRIES"); do
    export CANDEL_WATCH_ROUND="$attempt"

    echo "========================================"
    echo "[watch] Round $attempt/$MAX_RETRIES ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "========================================"

    run_and_capture_jobids "${resubmit_cmd[@]}"

    if [[ ${#job_ids[@]} -eq 0 ]]; then
        echo "[watch] No jobs submitted (dry run or error). Exiting."
        exit 0
    fi

    wait_for_jobs "${job_ids[@]}"

    if check_jobs; then
        echo ""
        echo "[watch] All jobs completed successfully!"
        exit 0
    fi

    if [[ $attempt -eq $MAX_RETRIES ]]; then
        echo "[watch] Max retries ($MAX_RETRIES) reached. Exiting."
        exit 1
    fi

    echo ""
    echo "[watch] Resubmitting (round $((attempt + 1))/$MAX_RETRIES)..."

    # Build resubmit command: original command, narrowed to incomplete
    # galaxies when submit.sh output provided a jobid -> galaxy mapping.
    if [[ ${#incomplete_gals[@]} -gt 0 ]]; then
        gal_csv=$(IFS=,; echo "${incomplete_gals[*]}")
        echo "[watch] Retrying galaxies: $gal_csv"
        replace_galaxy_arg "$gal_csv" "${CMD[@]}"
    else
        resubmit_cmd=("${CMD[@]}")
    fi
    if [[ -n "$RESUME_FLAG" ]]; then
        resubmit_cmd+=("$RESUME_FLAG")
    fi
done
