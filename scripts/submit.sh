#!/bin/bash -l
# Submit megamaser sampler or MAP jobs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIG_ARGS=("$@")
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

QUEUE=""
GALAXY=""
SAMPLER="mcmc"
MEM=""            # unset -> default memory request (computed post-parse)
CPUS=""
GPUTYPE=""
GPU_MEM=""
GPU_COUNT=""
TIME=""
DRY=false
LOCAL=false
SKIP_DONE=false
MAX_RETRIES=""
WATCH_POLL=""
INFER_H0=false
EVIDENCE=false
SAMPLER_EXPLICIT=false
COMPUTE_EVIDENCE=false
INIT_STRATEGY=""
SPOT_BATCH=""
PHI_INTEGRATION=""
PEAK_CANDIDATES_PER_WAVE=""
ADD_ECC=false
ADD_QW=false
MATCH_REID=false
FIX_FLOORS_PESCE=false
LOO_DROPPED=""

ALWAYS_ARGS=()
SINGLE_ARGS=()
VARIANT_ARGS=()       # --add-ecc/--add-quadratic-warp: valid for single + joint
MCMC_JOINT_ARGS=()
MCMC_ARGS=()
DE_ARGS=()
JOINT_ARGS=()
PASSTHRU_ARGS=()
SUBMIT_GALS="CGCG074-064 NGC5765b NGC6264 NGC6323 UGC3789"
ALL_GALS="$SUBMIT_GALS NGC4258"
GLAMDRING_GPU_QUEUES="gpulong cmbgpu optgpu"

usage() {
    cat <<EOF
Usage: $0 (--local | -q QUEUE) --galaxy GAL[,GAL,...]|all \\
          [--sampler mcmc|de | --infer-H0 | --evidence] [options] \\
          [-- runner args]

Required:
  --local                Run in the current terminal instead of submitting
                         to a batch backend.  Local runs are also tee'd to
                         <root_output>/<gal>/logs/<gal>_<sampler><variant>_<stamp>.log
                         (root_output from config_maser.toml [io], e.g.
                         results/Megamaser; full transcript incl. Reid/Pesce).
  -q, --queue QUEUE      Queue/partition for batch submission
                         (glamdring CPU: redwood|berg|cmb;
                         glamdring GPU: gpulong|cmbgpu|optgpu;
                         arc: short|medium|long).
  --galaxy GAL[,GAL,...]|all
                         Galaxy/galaxies to submit.
                         Choices: $ALL_GALS
                         all expands to: $SUBMIT_GALS
                         NGC4258 is only submitted if requested explicitly for
                         single-galaxy jobs.
  --sampler mcmc|de
                         Job type (default: mcmc). de runs the 2D-marginal
                         L-SHADE MAP (global search to seed mcmc). Ignored with
                         --infer-H0
                         (always a joint NUTS chain).
  --infer-H0             Run ONE toy joint H0 chain over all requested galaxies
                         with a shared H0 (run_joint_H0.py), instead of one
                         single-galaxy job each.  This samples H0 and one D_c
                         per galaxy using KDE distance likelihoods from the
                         saved single-galaxy MCMC chains.  --galaxy takes the
                         comma list (or all = the five MCP H0 galaxies; NGC4258
                         rejected).
  --evidence             Submit the single-galaxy harmonic evidence job for an
                         existing MCMC HDF5 chain. The chain path is resolved
                         from --galaxy, --init-strategy, and model flags.
                         This is the GPU path for total evidence.

Joint H0 options (with --infer-H0), passed to run_joint_H0.py:
  --selection none|distance|redshift
                         Default: redshift (selection ON by default).
  --distance-prior distance|volume
                         Default flips with --selection: distance when
                         selection=none, else volume.  distance is rejected
                         when selection is active (volume prior required).
  --reconstruction none|Carrick2015|ManticoreLocalCOLA
  --field-config PATH
  --Vext                 Sample external bulk flow Vext (off by default).
  --overwrite-vlos-cache
  (plus --num-warmup/--num-samples)

Common options passed to run_maser.py:
  --init-strategy median|config|reid
                         median/config apply to all samplers. reid is MCMC and
                         evidence only; DE never initialises from Pesce/Reid
                         (but reports its exact reference logP).
  --spot-batch N         Maser spots evaluated together per pass (DE, mcmc and
                         --evidence). Default: all at once (auto-shrunk only if
                         one candidate's spots overflow VRAM); lower to cut
                         memory at the cost of sequential spot passes.
  --add-ecc              Also valid with --infer-H0 (applies to all galaxies;
                         selects matching distance files).
  --add-quadratic-warp   Also valid with --infer-H0 (applies to all galaxies;
                         selects matching distance files).
  --f64                  Emergency/debug precision override.
  --fix-floors-pesce     Hold the five error floors fixed at the published
                         Pesce/Reid values. de: dropped from the DE search;
                         mcmc: dropped from the sampled sites.

MCMC/joint quick overrides passed to the Python runner:
  --num-warmup N
  --num-samples N
  --seed N               Random seed (all samplers; default: config
                         inference/seed).
  --num-chains N         Run N chains sequentially in one job. With the
                         default config/reid init they all start from the
                         same point with independent per-chain seeds and
                         drift apart; --init-strategy median instead gives
                         overdispersed starts.
  --output PATH
  --max-tree-depth N     NUTS max tree depth (default: config inference).
  --target-accept-theta F
                         NUTS global target acceptance (default: config
                         inference; mcmc + joint only).

Experimental MCMC options passed to run_maser.py --sampler mcmc:
  --compare-reid        After sampling, run the expensive Pesce/Reid/config/
                         MCMC fixed-global comparison. Disabled by default.
  --match-reid          Diagnostic: use Reid fit_disk physical constants.
                         Reid scatter always uses the same D_A.
  --compare-reid-2x     With --compare-reid, also run the 2x-denser-grid
                         logZ check. Disabled by default.
  --compute-evidence    Not supported through submit.sh; run a separate
                         --evidence submission after the chain finishes.
  --save-latents        Save per-spot r_ang/phi samples in the HDF5 output.
                         Disabled by default.

DE optimiser options passed to run_maser.py --sampler de:
  DE always uses L-SHADE and a data-ridge + Sobol initial population. The
                         Pesce/Reid point is never seeded; its exact all-spot
                         unnormalised log posterior density is reported.
                         Population reduction follows DE fitness evaluations,
                         independently of the generation ceiling.
  --resume               Resume from the DE checkpoint if present.
  --fix-globals          Skip the DE search; score logP and the conditional
                         r_ang MAP at the config [init] globals.
  --fix-globals-pesce    Like --fix-globals but at the published Pesce/Reid
                         globals, scoring the data-only marginal.
  --fix-floors-pesce     Run the full DE but hold the five error floors at the
                         published Pesce/Reid values (all other globals free).
  --phi-integration fixed-grid|peak-partition
                         Phi integration for the 2D marginal. Default:
                         fixed-grid. peak-partition numerically locates and
                         refines peaks in two independent systemic half-planes
                         and one half-plane for each high-velocity group.
  --peak-candidates-per-wave 1|2|4|8
                         Concurrent candidates per GPU for peak-partition.
                         Default: 8; try 2 or 4 when calibrating throughput.
Cluster options:
  MCMC jobs submit as CPU-only jobs. DE and --evidence request GPU.
  Joint H0 follows the selected node/queue: GPU queues request GPU; CPU queues
  force JAX_PLATFORMS=cpu.
  --cpus N              CPU cores. CPU jobs: total cores. GPU jobs: cores per
                        GPU (default 2 per GPU; --cpus 3 --gpu-count 8
                        requests 24 cores). glamdring submits as -s -n TOTAL
                        (shared slice), not the whole-node 1xN form.
  --gputype TYPE
  --gpu-mem GB           GPU VRAM request; also passed to DE/evidence
                         autobatching where relevant.
  --gpu-count N          GPUs per DE job on one node (-> --gres=gpu:N on ARC);
                         the runner uses adaptive weighted round-robin across
                         these local devices.
  --time T
  --mem GB              Memory in GB per CPU (default: 7) for every submitted
                         job type.
  --dry                 Print submit command without submitting.
  --skip-done           Skip MCMC jobs whose expected HDF5 output exists.
  -h, --help

Advanced runner options:
  Put options after -- to pass them directly to the selected Python runner.
  Example:
    $0 -q cmbgpu --galaxy all --infer-H0 -- --field-indices 0 1 2

Retries:
  --max-retries N       Launch this submit command through the detached
                         retry watcher.
  --poll S              Retry watcher poll interval in seconds (default: 120).
  DE marker:   "MAP init"
  MCMC marker: "saved samples to"
  Example:
    $0 -q cmbgpu --galaxy all --sampler de --max-retries 4

Galaxy aliases:
  5765b -> NGC5765b
  6264  -> NGC6264
  6323  -> NGC6323
  3789  -> UGC3789
  4258  -> NGC4258
EOF
}

strip_watcher_args() {
    local out=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --max-retries|--poll)
                shift 2 ;;
            *)
                out+=("$1"); shift ;;
        esac
    done
    printf '%s\0' "${out[@]}"
}

normalise_galaxy() {
    case "$1" in
        5765b|NGC5765B) echo "NGC5765b" ;;
        6264) echo "NGC6264" ;;
        6323) echo "NGC6323" ;;
        3789) echo "UGC3789" ;;
        4258) echo "NGC4258" ;;
        CGCG|CGCG074064|CGCG074-064) echo "CGCG074-064" ;;
        *) echo "$1" ;;
    esac
}

normalise_galaxy_csv() {
    local input="${1//,/ }"
    local out=()
    local gal
    for gal in $input; do
        if [[ "$gal" == "all" ]]; then
            out+=("all")
        else
            out+=("$(normalise_galaxy "$gal")")
        fi
    done
    local IFS=,
    echo "${out[*]}"
}

config_value() {
    local section="$1" key="$2"
    awk -F'=' -v section="[$section]" -v key="$key" '
        /^\[/ { current = $1$2 }
        current == section && $0 ~ "^[[:space:]]*"key"[[:space:]]*=" {
            line = $2
            sub(/[[:space:]]*(#.*)?$/, "", line)
            gsub(/^[[:space:]]*["'\'']|["'\''][[:space:]]*$/, "", line)
            print line
            exit
        }
    ' "$ROOT/scripts/megamaser/config_maser.toml" 2>/dev/null || true
}

queue_requests_gpu() {
    case "$CANDEL_CLUSTER" in
        glamdring)
            echo "$GLAMDRING_GPU_QUEUES" | grep -qw "$QUEUE" ;;
        *)
            [[ -n "$GPUTYPE" || -n "$GPU_MEM" ]] ;;
    esac
}

chain_init_strategy() {
    if [[ -n "$INIT_STRATEGY" ]]; then
        echo "$INIT_STRATEGY"
        return
    fi
    local init
    init="$(config_value inference init_strategy)"
    echo "${init:-config}"
}

chain_variant_suffix() {
    local init="$1"
    local parts=()
    [[ "$ADD_ECC" == true ]] && parts+=("ecc")
    [[ "$ADD_QW" == true ]] && parts+=("qw")
    [[ "$MATCH_REID" == true ]] && parts+=("matchreid")
    [[ "$FIX_FLOORS_PESCE" == true ]] && parts+=("fixfloors")
    parts+=("init${init}")
    local IFS=_
    echo "_${parts[*]}"
}

fail_if_args() {
    local context="$1"; shift
    if [[ $# -gt 0 ]]; then
        echo "[ERROR] options not valid with $context: $*"
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -q|--queue) QUEUE="$2"; shift 2 ;;
        --sampler) SAMPLER="$2"; SAMPLER_EXPLICIT=true; shift 2 ;;
        --galaxy) GALAXY="$2"; shift 2 ;;
        --mem) MEM="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --gputype) GPUTYPE="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM="$2"; shift 2 ;;
        --gpu-count) GPU_COUNT="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --max-retries) MAX_RETRIES="$2"; shift 2 ;;
        --poll) WATCH_POLL="$2"; shift 2 ;;
        --local) LOCAL=true; shift ;;
        --dry) DRY=true; shift ;;
        --skip-done) SKIP_DONE=true; shift ;;
        --infer-H0) INFER_H0=true; shift ;;
        --evidence) EVIDENCE=true; shift ;;
        --f64) ALWAYS_ARGS+=("--f64"); shift ;;
        --seed) ALWAYS_ARGS+=("--seed" "$2"); shift 2 ;;
        --Vext)
            JOINT_ARGS+=("$1"); shift ;;
        --overwrite-vlos-cache)
            JOINT_ARGS+=("$1"); shift ;;
        --selection)
            SELECTION="$2"; JOINT_ARGS+=("$1" "$2"); shift 2 ;;
        --reconstruction)
            RECONSTRUCTION="$2"; JOINT_ARGS+=("$1" "$2"); shift 2 ;;
        --leave-one-out-dropped)
            LOO_DROPPED="$2"; JOINT_ARGS+=("$1" "$2"); shift 2 ;;
        --distance-prior|--field-config)
            JOINT_ARGS+=("$1" "$2"); shift 2 ;;
        --init-strategy)
            INIT_STRATEGY="$2"; SINGLE_ARGS+=("$1" "$2"); shift 2 ;;
        --num-warmup|--num-samples|--num-chains|--output|\
            --max-tree-depth|--target-accept-theta)
            MCMC_JOINT_ARGS+=("$1" "$2"); shift 2 ;;
        --compare-reid|--match-reid|--compare-reid-2x|--compute-evidence)
            [[ "$1" == "--match-reid" ]] && MATCH_REID=true
            [[ "$1" == "--compute-evidence" ]] && COMPUTE_EVIDENCE=true
            MCMC_ARGS+=("$1"); shift ;;
        --save-latents)
            MCMC_ARGS+=("$1"); shift ;;
        --spot-batch)
            SPOT_BATCH="$2"; SINGLE_ARGS+=("$1" "$2"); shift 2 ;;
        --phi-integration)
            PHI_INTEGRATION="$2"; DE_ARGS+=("$1" "$2"); shift 2 ;;
        --peak-candidates-per-wave)
            PEAK_CANDIDATES_PER_WAVE="$2"
            DE_ARGS+=("$1" "$2"); shift 2 ;;
        --add-ecc|--add-quadratic-warp)
            case "$1" in
                --add-ecc) ADD_ECC=true ;;
                --add-quadratic-warp) ADD_QW=true ;;
            esac
            VARIANT_ARGS+=("$1"); shift ;;
        --resume|--fix-globals)
            DE_ARGS+=("$1"); shift ;;
        --fix-globals-pesce|--fix-floors-pesce)
            case "$1" in
                --fix-floors-pesce) FIX_FLOORS_PESCE=true ;;
            esac
            case "$1" in
                --fix-globals-pesce)
                    DE_ARGS+=("$1") ;;
                *)
                    SINGLE_ARGS+=("$1") ;;
            esac
            shift ;;
        --)
            shift
            PASSTHRU_ARGS+=("$@")
            break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$GALAXY" ]]; then
    echo "[ERROR] --galaxy is required. Choices: $ALL_GALS"; exit 1
fi
if [[ "$SAMPLER" != "mcmc" && "$SAMPLER" != "de" ]]; then
    echo "[ERROR] --sampler must be mcmc or de"; exit 1
fi
if [[ -n "$PHI_INTEGRATION" && "$PHI_INTEGRATION" != "fixed-grid" \
      && "$PHI_INTEGRATION" != "peak-partition" ]]; then
    echo "[ERROR] --phi-integration must be fixed-grid or peak-partition"
    exit 1
fi
if [[ -n "$PEAK_CANDIDATES_PER_WAVE" ]]; then
    case "$PEAK_CANDIDATES_PER_WAVE" in
        1|2|4|8) ;;
        *) echo "[ERROR] --peak-candidates-per-wave must be 1, 2, 4, or 8"
           exit 1 ;;
    esac
    if [[ "$PHI_INTEGRATION" != "peak-partition" ]]; then
        echo "[ERROR] --peak-candidates-per-wave requires" \
             "--phi-integration peak-partition"
        exit 1
    fi
fi
JOINT_H0_MODE=false
if [[ "$INFER_H0" == true ]]; then
    JOINT_H0_MODE=true
fi
if [[ "$EVIDENCE" == true && "$JOINT_H0_MODE" == true ]]; then
    echo "[ERROR] --evidence cannot be combined with joint H0 modes"; exit 1
fi
if [[ "$EVIDENCE" == true && "$SAMPLER_EXPLICIT" == true ]]; then
    echo "[ERROR] --sampler is not valid with --evidence"; exit 1
fi
if [[ "$EVIDENCE" == false && "$JOINT_H0_MODE" == false
      && "$SAMPLER" == "de" && "$(chain_init_strategy)" == "reid" ]]; then
    echo "[ERROR] DE never initialises from Pesce/Reid."
    echo "        Use --init-strategy median or config; Pesce logP is reported separately."
    exit 1
fi
if [[ "$COMPUTE_EVIDENCE" == true ]]; then
    echo "[ERROR] --compute-evidence is no longer submitted inline;"
    echo "        use --evidence after the chain finishes"
    exit 1
fi
if [[ "$EVIDENCE" == false && "$JOINT_H0_MODE" == false
      && "$SAMPLER" == "mcmc" && ( -n "$GPU_MEM" || -n "$GPUTYPE" ) ]]; then
    echo "[ERROR] GPU options are only valid with --evidence, --sampler de,"
    echo "        or --infer-H0"
    exit 1
fi
if [[ "$LOCAL" == false && -z "$QUEUE" ]]; then
    echo "[ERROR] pass --local or -q QUEUE (cluster=$CANDEL_CLUSTER)"; exit 1
fi
if [[ "$LOCAL" == false && "$CANDEL_CLUSTER" == "glamdring"
      && ( "$EVIDENCE" == true
           || ( "$JOINT_H0_MODE" == false && "$SAMPLER" == "de" ) ) ]]; then
    echo "$GLAMDRING_GPU_QUEUES" | grep -qw "$QUEUE" || {
        echo "[ERROR] $QUEUE is a glamdring CPU queue, but this mode needs a GPU."
        echo "        Use one of: $GLAMDRING_GPU_QUEUES"
        exit 1
    }
fi
if [[ "$LOCAL" == false && "$JOINT_H0_MODE" == true
      && ( -n "$GPUTYPE" || -n "$GPU_MEM" ) ]]; then
    queue_requests_gpu || {
        echo "[ERROR] GPU options were given, but $QUEUE is not a GPU queue."
        exit 1
    }
fi
if [[ -n "$MAX_RETRIES" && "$LOCAL" == true ]]; then
    echo "[ERROR] --max-retries requires batch submission, not --local"; exit 1
fi
if [[ -n "$MAX_RETRIES" && "$EVIDENCE" == true ]]; then
    echo "[ERROR] --max-retries is not supported with --evidence"; exit 1
fi
if [[ "$SKIP_DONE" == true ]]; then
    if [[ "$EVIDENCE" == true || "$JOINT_H0_MODE" == true
          || "$SAMPLER" != "mcmc" ]]; then
        echo "[ERROR] --skip-done is only supported with --sampler mcmc"
        exit 1
    fi
fi
RUN_ARGS=()
if [[ "$EVIDENCE" == true ]]; then
    bad_args=()
    [[ ${#JOINT_ARGS[@]} -gt 0 ]] && bad_args+=("${JOINT_ARGS[@]}")
    [[ ${#DE_ARGS[@]} -gt 0 ]] && bad_args+=("${DE_ARGS[@]}")
    for arg in ${MCMC_ARGS[@]+"${MCMC_ARGS[@]}"}; do
        [[ "$arg" == "--match-reid" ]] && continue
        bad_args+=("$arg")
    done
    [[ ${#MCMC_JOINT_ARGS[@]} -gt 0 ]] && bad_args+=("${MCMC_JOINT_ARGS[@]}")
    [[ ${#bad_args[@]} -gt 0 ]] && fail_if_args "--evidence" "${bad_args[@]}"
elif [[ "$JOINT_H0_MODE" == true ]]; then
    # Joint H0 modes are always one joint NUTS chain; --sampler is ignored.
    bad_args=()
    [[ ${#SINGLE_ARGS[@]} -gt 0 ]] && bad_args+=("${SINGLE_ARGS[@]}")
    [[ ${#MCMC_ARGS[@]} -gt 0 ]] && bad_args+=("${MCMC_ARGS[@]}")
    [[ ${#DE_ARGS[@]} -gt 0 ]] && bad_args+=("${DE_ARGS[@]}")
    [[ ${#bad_args[@]} -gt 0 ]] && fail_if_args "--infer-H0" "${bad_args[@]}"
    [[ ${#ALWAYS_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${ALWAYS_ARGS[@]}")
    [[ ${#VARIANT_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${VARIANT_ARGS[@]}")
    [[ ${#MCMC_JOINT_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${MCMC_JOINT_ARGS[@]}")
    [[ ${#JOINT_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${JOINT_ARGS[@]}")
    [[ ${#PASSTHRU_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${PASSTHRU_ARGS[@]}")
elif [[ "$SAMPLER" == "mcmc" ]]; then
    bad_args=()
    [[ ${#JOINT_ARGS[@]} -gt 0 ]] && bad_args+=("${JOINT_ARGS[@]}")
    [[ ${#DE_ARGS[@]} -gt 0 ]] && bad_args+=("${DE_ARGS[@]}")
    [[ ${#bad_args[@]} -gt 0 ]] && fail_if_args "--sampler mcmc" "${bad_args[@]}"
    [[ ${#ALWAYS_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${ALWAYS_ARGS[@]}")
    [[ ${#SINGLE_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${SINGLE_ARGS[@]}")
    [[ ${#VARIANT_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${VARIANT_ARGS[@]}")
    [[ ${#MCMC_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${MCMC_ARGS[@]}")
    [[ ${#MCMC_JOINT_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${MCMC_JOINT_ARGS[@]}")
    [[ ${#PASSTHRU_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${PASSTHRU_ARGS[@]}")
else
    bad_args=()
    [[ ${#JOINT_ARGS[@]} -gt 0 ]] && bad_args+=("${JOINT_ARGS[@]}")
    [[ ${#MCMC_ARGS[@]} -gt 0 ]] && bad_args+=("${MCMC_ARGS[@]}")
    [[ ${#MCMC_JOINT_ARGS[@]} -gt 0 ]] && bad_args+=("${MCMC_JOINT_ARGS[@]}")
    [[ ${#bad_args[@]} -gt 0 ]] && fail_if_args "--sampler de" "${bad_args[@]}"
    [[ ${#ALWAYS_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${ALWAYS_ARGS[@]}")
    [[ ${#SINGLE_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${SINGLE_ARGS[@]}")
    [[ ${#VARIANT_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${VARIANT_ARGS[@]}")
    [[ ${#DE_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${DE_ARGS[@]}")
    # --gpu-mem (SLURM min-VRAM request) also tells the DE planner which V100.
    [[ -n "$GPU_MEM" ]] && RUN_ARGS+=("--gpu-mem" "$GPU_MEM")
    [[ ${#PASSTHRU_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${PASSTHRU_ARGS[@]}")
fi

if [[ -n "$MAX_RETRIES" && -z "${CANDEL_WATCH_ACTIVE:-}" ]]; then
    marker="saved samples to"
    [[ "$SAMPLER" != "mcmc" ]] && marker="MAP init"
    [[ "$JOINT_H0_MODE" == true ]] && marker="saved samples to"
    watcher=("$ROOT/scripts/megamaser/watch_and_resubmit.sh"
             --marker "$marker" --max-retries "$MAX_RETRIES")
    [[ -n "$WATCH_POLL" ]] && watcher+=(--poll "$WATCH_POLL")
    if [[ "$JOINT_H0_MODE" == true || "$SAMPLER" == "mcmc" ]]; then
        watcher+=(--no-resume)
    fi

    submit_args=()
    while IFS= read -r -d '' arg; do
        submit_args+=("$arg")
    done < <(strip_watcher_args "${ORIG_ARGS[@]}")

    watch_kind="$SAMPLER"
    [[ "$INFER_H0" == true ]] && watch_kind="jointh0"
    safe_gal="$(printf '%s' "$GALAXY" | tr ', /' '___' | tr -cd '[:alnum:]_.-')"
    stamp="$(date '+%Y%m%d_%H%M%S')"
    session="maser_${watch_kind}_${safe_gal}_${stamp}"
    logfile="$CANDEL_WATCHER_DIR/${session}.log"
    mkdir -p "$CANDEL_WATCHER_DIR"

    echo "[submit] Launching retry watcher: marker='$marker', max_retries=$MAX_RETRIES"
    echo "[submit] Watcher log: $logfile"
    launch_detached "$session" "$logfile" \
        env CANDEL_WATCH_ACTIVE=1 "${watcher[@]}" -- \
        "$ROOT/scripts/megamaser/submit.sh" "${submit_args[@]}"
    exit $?
fi

dry_flag=()
[[ "$DRY" == true ]] && dry_flag=(--dry)

if [[ "$EVIDENCE" == true && -z "$CPUS" ]]; then
    CPUS=2
fi
if [[ "$JOINT_H0_MODE" == true && -z "$CPUS" ]]; then
    CPUS=4
fi

# Default memory: 7 GB per CPU for every submitted job type. MEM is per-CPU
# (matches addqueue -m / SLURM --mem-per-cpu); submit_job scales it to a total
# for backends that want one (arc).
if [[ -z "$MEM" ]]; then
    MEM=7
fi

extra_flags=()
[[ -n "$CPUS" ]] && extra_flags+=(--cpus "$CPUS")
[[ -n "$GPUTYPE" ]] && extra_flags+=(--gputype "$GPUTYPE")
[[ -n "$GPU_MEM" ]] && extra_flags+=(--gpu-mem "$GPU_MEM")
[[ -n "$TIME" ]] && extra_flags+=(--time "$TIME")

if [[ "$JOINT_H0_MODE" == true ]]; then
    # One joint NUTS chain over all requested galaxies (shared H0).  The galaxy
    # list (comma-separated, or "all" = the five MCP H0 galaxies) is passed
    # straight to run_joint_H0.py, which is NOT expanded into per-galaxy jobs.
    runner="$ROOT/scripts/megamaser/run_joint_H0.py"
    GALAXY="$(normalise_galaxy_csv "$GALAXY")"
    if [[ "$GALAXY" == *all,* || "$GALAXY" == *,all* ]]; then
        echo "[ERROR] --galaxy all cannot be combined with explicit galaxies"
        exit 1
    fi
    if [[ ",$GALAXY," == *",NGC4258,"* ]]; then
        echo "[ERROR] joint H0 inference excludes NGC4258"
        exit 1
    fi
    gal_tag="$(printf '%s' "$GALAXY" | tr ', /' '___' | tr -cd '[:alnum:]_.-')"
    joint_label="joint H0"
    # Tag the job (hence the scheduler log filename) with selection and
    # reconstruction, matching the joint_H0 result-file convention; defaults
    # mirror run_joint_H0.py (redshift / none).
    job_name="maser_jointh0_${gal_tag}_${SELECTION:-redshift}_${RECONSTRUCTION:-none}"
    [[ -n "$LOO_DROPPED" ]] && job_name="${job_name}_loo${LOO_DROPPED}"
    if [[ "$LOCAL" == true ]]; then
        echo "Running $joint_label ($GALAXY) locally"
        cmd=(/usr/bin/env JAX_PLATFORMS=cpu "$CANDEL_PYTHON" -u "$runner"
             --galaxy "$GALAXY")
        [[ ${#RUN_ARGS[@]} -gt 0 ]] && cmd+=("${RUN_ARGS[@]}")
        if [[ "$DRY" == true ]]; then
            printf '[dry]'; printf ' %q' "${cmd[@]}"; printf '\n'
        else
            "${cmd[@]}"
        fi
    else
        # Joint H0 runs on GPU when submitted to a glamdring GPU queue
        # (gpulong|cmbgpu|optgpu); on a CPU queue it runs as a plain CPU job
        # (JAX_PLATFORMS=cpu, no GPU request) so it can't land on a CPU node
        # with CUDA forced.  Non-glamdring clusters keep the GPU path.
        joint_gpu=true
        if [[ "$CANDEL_CLUSTER" == "glamdring" ]]; then
            case "$QUEUE" in
                gpulong|cmbgpu|optgpu) joint_gpu=true ;;
                *) joint_gpu=false ;;
            esac
        fi
        echo "Submitting $joint_label ($GALAXY) -> $CANDEL_CLUSTER:$QUEUE" \
             "($([[ "$joint_gpu" == true ]] && echo GPU || echo CPU))"
        if [[ "$joint_gpu" == true ]]; then
            pycmd="$CANDEL_PYTHON -u $runner --galaxy $GALAXY"
        else
            pycmd="/usr/bin/env JAX_PLATFORMS=cpu $CANDEL_PYTHON -u $runner --galaxy $GALAXY"
        fi
        [[ ${#RUN_ARGS[@]} -gt 0 ]] && pycmd+=" ${RUN_ARGS[*]}"
        # Joint output is a flat results/Megamaser/joint_H0_*.hdf5 (no per-galaxy
        # dir), so copy the scheduler log into a shared <root_output>/logs.
        joint_root="$(config_value io root_output)"
        [[ -z "$joint_root" ]] && joint_root="results/Megamaser"
        submit_args=(--queue "$QUEUE" --mem "$MEM"
                     --name "$job_name"
                     --logdir "$ROOT/$joint_root/logs")
        if [[ "$joint_gpu" == true ]]; then
            submit_args=(--gpu "${submit_args[@]}")
        fi
        # CPU job: a single threaded JAX process, not MPI. --cpus (in
        # extra_flags) requests N shared cores on one node (addqueue -s -n N);
        # the -s -n 1xN node-form grabs a whole node on glamdring.
        [[ ${#extra_flags[@]} -gt 0 ]] && submit_args+=("${extra_flags[@]}")
        [[ ${#dry_flag[@]} -gt 0 ]] && submit_args+=("${dry_flag[@]}")
        submit_job "${submit_args[@]}" -- $pycmd
    fi
    exit 0
fi

GALAXY="${GALAXY//,/ }"
expanded_galaxies=()
for gal in $GALAXY; do
    gal="$(normalise_galaxy "$gal")"
    if [[ "$gal" == "all" ]]; then
        expanded_galaxies+=($SUBMIT_GALS)
        continue
    fi
    echo "$ALL_GALS" | grep -qw "$gal" || {
        echo "Error: unknown galaxy '$gal'. Choices: $ALL_GALS or all"; exit 1;
    }
    expanded_galaxies+=("$gal")
done
GALAXY="${expanded_galaxies[*]}"

RUNNER="$ROOT/scripts/megamaser/run_maser.py"
case "$SAMPLER" in
    de) JOB_PREFIX="maser_de" ;;
    mcmc) JOB_PREFIX="maser_mcmc" ;;
    *) JOB_PREFIX="maser_mcmc" ;;
esac

# Local runs are tee'd to a per-run log under the galaxy's output subdir, so the
# full transcript (sampler progress, the Reid/Pesce comparison table, JAX/XLA
# warnings) is saved beside that galaxy's HDF5 outputs. run_maser.py writes to
# <root_output>/<gal>/; derive root_output from config_maser.toml [io]
# (defaults to results/Maser only when that key is absent) so it can't drift.
# root_results resolves to the repo root (local_config.toml).
maser_root_output="$(
    awk -F'=' '
        /^\[/ { s = $1$2 }
        s == "[io]" && /^[[:space:]]*root_output[[:space:]]*=/ {
            gsub(/[" ]/, "", $2); print $2; exit
        }
    ' "$ROOT/scripts/megamaser/config_maser.toml" 2>/dev/null || true
)"
[[ -z "$maser_root_output" ]] && maser_root_output="results/Maser"
MASER_OUT="$ROOT/$maser_root_output"
stamp="$(date '+%Y%m%d_%H%M%S')"

if [[ "$EVIDENCE" == true ]]; then
    EVIDENCE_RUNNER="$ROOT/scripts/megamaser/evidence_single_galaxy.py"
    init_strategy="$(chain_init_strategy)"
    suffix="blackjax_mcmc_rphi$(chain_variant_suffix "$init_strategy")"
    for gal in $GALAXY; do
        chain="$MASER_OUT/$gal/${gal}_${suffix}.hdf5"
        logdir="$MASER_OUT/$gal/logs"
        echo "[evidence] $gal: $chain"
        if [[ ! -f "$chain" ]]; then
            if [[ "$DRY" == true ]]; then
                echo "[evidence] WARNING: chain does not exist yet."
            else
                echo "[ERROR] evidence chain not found: $chain" >&2
                exit 1
            fi
        fi
        evidence_args=(--chain "$chain")
        [[ -n "$GPU_MEM" ]] && evidence_args+=(--gpu-mem "$GPU_MEM")
        [[ -n "$SPOT_BATCH" ]] && evidence_args+=(--spot-batch "$SPOT_BATCH")
        [[ ${#PASSTHRU_ARGS[@]} -gt 0 ]] && evidence_args+=("${PASSTHRU_ARGS[@]}")
        if [[ "$LOCAL" == true ]]; then
            cmd=("$CANDEL_PYTHON" -u "$EVIDENCE_RUNNER" "$gal")
            cmd+=("${evidence_args[@]}")
            if [[ "$DRY" == true ]]; then
                printf '[dry]'; printf ' %q' "${cmd[@]}"; printf '\n'
            else
                "${cmd[@]}"
            fi
            continue
        fi
        echo "Submitting $gal evidence -> $CANDEL_CLUSTER:$QUEUE"
        pycmd="$CANDEL_PYTHON -u $EVIDENCE_RUNNER $gal"
        pycmd+=" ${evidence_args[*]}"
        submit_args=(--gpu --queue "$QUEUE" --mem "$MEM"
                     --name "maser_evidence_${gal}" --logdir "$logdir")
        [[ ${#extra_flags[@]} -gt 0 ]] && submit_args+=("${extra_flags[@]}")
        [[ ${#dry_flag[@]} -gt 0 ]] && submit_args+=("${dry_flag[@]}")
        submit_job "${submit_args[@]}" -- $pycmd
    done
    exit 0
fi

variant_tag=""
if [[ ${#RUN_ARGS[@]} -gt 0 ]]; then
    for a in "${RUN_ARGS[@]}"; do
        case "$a" in
            --add-quadratic-warp) variant_tag="${variant_tag}_qw" ;;
            --add-ecc)            variant_tag="${variant_tag}_ecc" ;;
            --compare-reid)       variant_tag="${variant_tag}_reid" ;;
            --match-reid)         variant_tag="${variant_tag}_matchreid" ;;
            --fix-floors-pesce)   variant_tag="${variant_tag}_fixfloors" ;;
        esac
    done
fi
[[ "$PHI_INTEGRATION" == "peak-partition" ]] && \
    variant_tag="${variant_tag}_peakpartition"

for gal in $GALAXY; do
    if [[ "$SKIP_DONE" == true ]]; then
        init_strategy="$(chain_init_strategy)"
        suffix="blackjax_mcmc_rphi$(chain_variant_suffix "$init_strategy")"
        outpath="$MASER_OUT/$gal/${gal}_${suffix}.hdf5"
        if [[ -f "$outpath" ]]; then
            echo "[skip-done] $gal: $outpath"
            continue
        fi
    fi

    if [[ "$LOCAL" == true ]]; then
        echo "Running $gal ($SAMPLER) locally"
        if [[ "$SAMPLER" == "mcmc" ]]; then
            cmd=(/usr/bin/env JAX_PLATFORMS=cpu "$CANDEL_PYTHON" -u "$RUNNER"
                 "$gal" --sampler "$SAMPLER")
        else
            cmd=("$CANDEL_PYTHON" -u "$RUNNER" "$gal" --sampler "$SAMPLER")
        fi
        if [[ ${#RUN_ARGS[@]} -gt 0 ]]; then
            cmd+=("${RUN_ARGS[@]}")
        fi
        if [[ "$DRY" == true ]]; then
            printf '[dry]'; printf ' %q' "${cmd[@]}"; printf '\n'
        else
            logdir="$MASER_OUT/$gal/logs"
            mkdir -p "$logdir"
            logfile="$logdir/${gal}_${SAMPLER}${variant_tag}_${stamp}.log"
            echo "[submit] tee-ing output to: $logfile"
            # pipefail (set at top) propagates the runner's exit status here.
            "${cmd[@]}" 2>&1 | tee "$logfile"
        fi
        continue
    fi

    echo "Submitting $gal ($SAMPLER) -> $CANDEL_CLUSTER:$QUEUE"
    if [[ "$SAMPLER" == "mcmc" ]]; then
        pycmd="/usr/bin/env JAX_PLATFORMS=cpu $CANDEL_PYTHON -u $RUNNER $gal --sampler $SAMPLER"
    else
        pycmd="$CANDEL_PYTHON -u $RUNNER $gal --sampler $SAMPLER"
    fi
    if [[ ${#RUN_ARGS[@]} -gt 0 ]]; then
        pycmd+=" ${RUN_ARGS[*]}"
    fi
    logdir="$MASER_OUT/$gal/logs"
    submit_args=(--queue "$QUEUE" --mem "$MEM" --name "${JOB_PREFIX}_${gal}"
                 --logdir "$logdir")
    if [[ "$SAMPLER" != "mcmc" ]]; then
        submit_args=(--gpu "${submit_args[@]}")
        if [[ -n "$GPU_COUNT" && "$GPU_COUNT" -gt 1 ]]; then
            submit_args+=(--gpu-count "$GPU_COUNT")
        fi
    fi
    # MCMC CPU job: a single threaded JAX process, not MPI. --cpus (in
    # extra_flags) requests N shared cores on one node (addqueue -s -n N); the
    # -s -n 1xN node-form grabs a whole node on glamdring.
    if [[ ${#extra_flags[@]} -gt 0 ]]; then
        submit_args+=("${extra_flags[@]}")
    fi
    if [[ ${#dry_flag[@]} -gt 0 ]]; then
        submit_args+=("${dry_flag[@]}")
    fi
    submit_job "${submit_args[@]}" -- $pycmd
done
