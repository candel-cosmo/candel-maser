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
#       bash scripts/megamaser/submit.sh --sampler de -q cmbgpu --galaxy NGC5765b
#
#   bash watch_and_resubmit.sh --marker "saved samples to" -- \
#       bash scripts/megamaser/submit.sh --sampler nss -q cmbgpu --galaxy NGC5765b
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
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

On resubmit, the same command is re-run with the resume flag appended.

Examples:
  # DE MAP (one galaxy, auto-resume on timeout)
  bash watch_and_resubmit.sh --marker "MAP init" -- \
      bash scripts/megamaser/submit.sh --sampler de -q cmbgpu --galaxy NGC5765b

  # DE MAP (multiple galaxies)
  bash watch_and_resubmit.sh --marker "MAP init" -- \
      bash scripts/megamaser/submit.sh --sampler de -q cmbgpu --galaxy NGC5765b,NGC6264

  # NSS (one galaxy)
  bash watch_and_resubmit.sh --marker "saved samples to" -- \
      bash scripts/megamaser/submit.sh --sampler nss -q cmbgpu --galaxy NGC5765b

  # NUTS (no resume support)
  bash watch_and_resubmit.sh --marker "saved samples to" --no-resume -- \
      bash scripts/megamaser/submit.sh --sampler nuts -q cmbgpu --galaxy NGC5765b

  # Custom poll and retries
  bash watch_and_resubmit.sh --marker "MAP init" --max-retries 10 --poll 60 -- \
      bash scripts/megamaser/submit.sh --sampler de -q cmbgpu --galaxy NGC5765b
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
    # submit_job writes logs as "logs-<jobid>-<jobname>.out" in $PWD on
    # both clusters. The watcher only knows the jobid, so glob to find it.
    local jid="$1"
    local f
    f=$(ls "$PWD"/logs-"${jid}"-*.out 2>/dev/null | head -1)
    echo "${f:-$PWD/logs-${jid}-<name>.out}"
}

run_and_capture_jobids() {
    # Run the command, tee output to the terminal, collect JOBID= lines.
    local -n _jids=$1; shift
    local cmd=("$@")
    echo "[watch] Running: ${cmd[*]}"
    _out=$("${cmd[@]}" 2>&1) || true
    echo "$_out"
    while IFS= read -r line; do
        if [[ "$line" =~ ^JOBID=([0-9]+)$ ]]; then
            _jids+=("${BASH_REMATCH[1]}")
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
    local jids=("$@")
    local any_incomplete=0
    for jid in "${jids[@]}"; do
        logfile=$(log_path_for_job "$jid")
        if [[ -f "$logfile" ]] && grep -q "$MARKER" "$logfile"; then
            echo "[watch] Job $jid: COMPLETE"
        else
            echo "[watch] Job $jid: INCOMPLETE (log: ${logfile})"
            any_incomplete=1
        fi
    done
    return $any_incomplete
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

    job_ids=()
    run_and_capture_jobids job_ids "${resubmit_cmd[@]}"

    if [[ ${#job_ids[@]} -eq 0 ]]; then
        echo "[watch] No jobs submitted (dry run or error). Exiting."
        exit 0
    fi

    wait_for_jobs "${job_ids[@]}"

    if check_jobs "${job_ids[@]}"; then
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

    # Build resubmit command: original command + resume flag (if set)
    resubmit_cmd=("${CMD[@]}")
    if [[ -n "$RESUME_FLAG" ]]; then
        resubmit_cmd+=("$RESUME_FLAG")
    fi
done
