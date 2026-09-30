#!/bin/bash -l
# Submit megamaser sampler or MAP jobs.
set -euo pipefail

ROOT="${CANDEL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
ORIG_ARGS=("$@")
# shellcheck source=../../../scripts/_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

QUEUE=""
GALAXY=""
SAMPLER="mcmc"
MEM=""            # unset -> default memory request (computed post-parse)
CPUS=""
NUM_CHAINS=""
CHAIN_WORKERS=""
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
SINGLE_ERROR_FLOOR=false
CLUMP2_ACCELERATION_FLOOR_ONLY=false
DA2_PRIOR=false
TEMP_OUTPUT=false
TEMP_ROOT_OUTPUT="results_test/Megamaser"
FIX_GLOBALS=false
LOO_DROPPED=""
DISTANCE_SOURCE="candel"

DATASET=""
ALWAYS_ARGS=()
SINGLE_ARGS=()
INIT_ARGS=()
VARIANT_ARGS=()       # --add-ecc/--add-quadratic-warp: valid for single + joint
MCMC_JOINT_ARGS=()
MCMC_ARGS=()
DE_ARGS=()
ITERATIVE_CLIP=false
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
                         to a batch backend.  Single-galaxy local runs are tee'd to
                         <root_output>/<dataset>/<gal>/logs/
                         <gal>_<sampler><variant>_<stamp>.log
                         (root_output from config_maser.toml [io], e.g.
                         results/Megamaser; full runner transcript).
  -q, --queue QUEUE      Queue/partition for batch submission
                         (glamdring CPU: redwood|berg|cmb;
                         glamdring GPU: gpulong|cmbgpu|optgpu;
                         arc: short|medium|long).
  --dataset NAME[,NAME,...]
                         Spot-table dataset(s): original_published, fiducial,
                         unpruned, or clipped. Default: [io].dataset from
                         config_maser.toml (currently fiducial). Selects the tables,
                         the init_<dataset>.toml best points, and the
                         <root_output>/<dataset>/ results namespace.
  --temp-output          Write test chains, checkpoints, diagnostics, and
                         copied logs below results_test/Megamaser/<dataset>/.
                         Iterative clipping will not update data/Megamaser/.
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
                         rejected). Automatic outputs go below
                         <root_output>/<dataset>/H0/.
  --evidence             Submit the single-galaxy harmonic marginal-objective
                         diagnostic for an existing MCMC HDF5 chain. The chain
                         path is resolved from --galaxy, --init-strategy, and
                         model flags. This is not rigorous absolute evidence.

Joint H0 options (with --infer-H0), passed to run_joint_H0.py:
  --selection none|distance|redshift
                         Default: redshift (selection ON by default).
  --distance-prior distance|volume|log-distance
                         Default flips with --selection: distance when
                         selection=none, else volume.  distance is rejected
                         when selection is active (volume prior required).
                         log-distance is uniform in log(D_A) and also requires
                         selection=none.
  --distance-source candel|p20
                         Stage-1 distance posteriors (default: candel). p20
                         reads the archived Dom text files and removes their
                         uniform-in-log(D_A) prior.
  --single-error-floor   When NGC5765b is included, select its stage-1 chain
                         fitted with one standard floor per observable.
  --clump2-acceleration-floor-only
                         Select the NGC5765b stage-1 chain fitted with only a
                         separate clump-2 acceleration floor.
  --reconstruction none|Carrick2015|ManticoreLocalCOLA
  --field-config PATH
  --Vext                 Sample external bulk flow Vext (off by default).
  --overwrite-vlos-cache
  (plus --num-warmup/--num-samples)

Common options passed to run_maser.py:
  --init-strategy median|config|reid
                         MCMC/evidence initial point. Real DE searches ignore
                         it and use their model-specific seed policy;
                         median/config only select the separate --fix-globals
                         diagnostic point.
  --spot-batch N         Maser spots evaluated together per pass (DE, mcmc and
                         --evidence). Default: all at once (auto-shrunk only if
                         one candidate's spots overflow VRAM); lower to cut
                         memory at the cost of sequential spot passes.
  --add-ecc              Also valid with --infer-H0 (applies to all galaxies;
                         selects matching distance files).
  --add-quadratic-warp   Also valid with --infer-H0 (applies to all galaxies;
                         selects matching distance files).
  --f64                  Force float64 for DE; MCMC already always uses it.
  --seed N               Random seed (all samplers; default: config
                         inference/seed). DE checkpoints are separated by seed.
  --single-error-floor   NGC5765b only: disable its separate sampled clump-2
                         floors and use the standard floor per observable.
  --clump2-acceleration-floor-only
                         NGC5765b only: sample a separate clump-2 acceleration
                         floor; position and velocity use the standard floors.
  --da2-prior            Use p(D_A) proportional to D_A^2 over the configured
                         D_A bounds instead of the default uniform prior.

MCMC/joint quick overrides passed to the Python runner:
  --num-warmup N
  --num-samples N
  --num-chains N         Run N chains in one job, concurrently up to the
                         allocated CPU and worker limits. Every chain count
                         defaults to the configured initial point.
  --chain-workers N      Single-galaxy MCMC only. Run at most N chains
                         concurrently (default: 8). Without --cpus, requests
                         min(chains, workers) CPUs; --cpus overrides only that
                         scheduler request.
  --output PATH
  --max-tree-depth N     NUTS max tree depth (default: config inference).
  --target-accept-theta F
                         NUTS global target acceptance (default: config
                         inference; mcmc + joint only).

Experimental MCMC options passed to run_maser.py --sampler mcmc:
  --compute-evidence    Not supported through submit.sh; run a separate
                         --evidence diagnostic after the chain finishes.
  --save-latents        Save per-spot r_ang/phi samples in the HDF5 output.
                         Disabled by default.

DE optimiser options passed to run_maser.py --sampler de:
  DE always uses L-SHADE. The initial population normally uses a data ridge
                         plus Sobol points. Quadratic-warp models use the exact
                         lifted base-model point, an expansion-only cloud, a
                         ridge anchored to the linear-fit mass, and Sobol.
                         Pesce/Reid is never seeded.
                         Population reduction follows DE fitness evaluations,
                         independently of the generation ceiling.
  --skip-base-model-seed Explicitly omit that quadratic base-model seed cloud.
  --resume               Resume from the DE checkpoint if present.
  --fix-globals          Skip the DE search; score logP and the conditional
                         r_ang MAP at the config [init] globals.
  --fix-globals-pesce    Like --fix-globals but at the published Pesce/Reid
                         globals, scoring the data-only marginal.
  --phi-integration fixed-grid|peak-partition
                         Phi integration for the 2D marginal. Default:
                         config_maser.toml (currently peak-partition).
                         peak-partition numerically locates and refines peaks
                         in two independent systemic half-planes and one
                         half-plane for each high-velocity group.
  --peak-candidates-per-wave 1|2|4|8
                         Concurrent candidates per GPU for peak-partition.
                         Default: 8; try 2 or 4 when calibrating throughput.
  --iterative-clip-sigma [SIGMA]
                         On the unpruned dataset, rerun DE and cumulatively
                         remove MAP x/y/velocity residual outliers. SIGMA
                         defaults to 2.5; writes the canonical clipped mask.
  --clip-max-attempts N  Maximum DE fits in the clipping loop (default: 5).
  --patience N           Stop DE after N generations without a >0.1 logP
                         improvement (default: [optimise].patience from the
                         config). Raise this to tolerate more stale generations.
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
  Put runner-only options at the end of the command. The first unrecognised
  option and everything after it are passed directly to the selected Python
  runner; an explicit -- separator remains supported but is optional.
  Example:
    $0 -q cmbgpu --galaxy NGC6264 --sampler de \
       --checkpoint-interval-minutes 1

Retries:
  --max-retries N       Launch this submit command through the detached
                         retry watcher.
  --poll S              Retry watcher poll interval in seconds (default: 120).
  DE marker:   "MAP init" ("iterative clipping complete" for clipping loops)
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
            sub(/^[[:space:]]*/, "", line)
            gsub(/^[[:space:]]*["'\'']|["'\''][[:space:]]*$/, "", line)
            print line
            exit
        }
    ' "$ROOT/packages/candel-maser/configs/config_maser.toml" 2>/dev/null || true
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
    local galaxy="$2"
    local parts=()
    [[ "$ADD_ECC" == true ]] && parts+=("ecc")
    [[ "$ADD_QW" == true ]] && parts+=("qw")
    if [[ "$galaxy" == "NGC5765b" \
          && "$CLUMP2_ACCELERATION_FLOOR_ONLY" == true ]]; then
        parts+=("accelfloor")
    elif [[ "$galaxy" == "NGC5765b" \
            && "$SINGLE_ERROR_FLOOR" == true ]]; then
        parts+=("singlefloor")
    fi
    [[ "$DA2_PRIOR" == true ]] && parts+=("da2")
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
        --temp-output) TEMP_OUTPUT=true; shift ;;
        --infer-H0) INFER_H0=true; shift ;;
        --evidence) EVIDENCE=true; shift ;;
        --dataset) DATASET="$2"; ALWAYS_ARGS+=("$1" "$2"); shift 2 ;;
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
        --distance-source)
            DISTANCE_SOURCE="$2"; JOINT_ARGS+=("$1" "$2"); shift 2 ;;
        --init-strategy)
            INIT_STRATEGY="$2"; INIT_ARGS+=("$1" "$2"); shift 2 ;;
        --num-chains)
            NUM_CHAINS="$2"
            MCMC_JOINT_ARGS+=("$1" "$2"); shift 2 ;;
        --chain-workers)
            CHAIN_WORKERS="$2"
            MCMC_ARGS+=("$1" "$2"); shift 2 ;;
        --num-warmup|--num-samples|--output|\
            --max-tree-depth|--target-accept-theta)
            MCMC_JOINT_ARGS+=("$1" "$2"); shift 2 ;;
        --compute-evidence)
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
        --iterative-clip-sigma)
            ITERATIVE_CLIP=true
            if [[ $# -gt 1 && "$2" != -* ]]; then
                DE_ARGS+=("$1" "$2"); shift 2
            else
                DE_ARGS+=("$1" "2.5"); shift
            fi ;;
        --clip-max-attempts)
            DE_ARGS+=("$1" "$2"); shift 2 ;;
        --patience)
            DE_ARGS+=("$1" "$2"); shift 2 ;;
        --add-ecc|--add-quadratic-warp)
            case "$1" in
                --add-ecc) ADD_ECC=true ;;
                --add-quadratic-warp) ADD_QW=true ;;
            esac
            VARIANT_ARGS+=("$1"); shift ;;
        --single-error-floor)
            SINGLE_ERROR_FLOOR=true
            VARIANT_ARGS+=("$1"); shift ;;
        --clump2-acceleration-floor-only)
            CLUMP2_ACCELERATION_FLOOR_ONLY=true
            VARIANT_ARGS+=("$1"); shift ;;
        --da2-prior)
            DA2_PRIOR=true
            MCMC_ARGS+=("$1"); shift ;;
        --resume|--fix-globals|--skip-base-model-seed)
            [[ "$1" == "--fix-globals" ]] && FIX_GLOBALS=true
            DE_ARGS+=("$1"); shift ;;
        --fix-globals-pesce)
            DE_ARGS+=("$1")
            shift ;;
        --)
            shift
            PASSTHRU_ARGS+=("$@")
            break ;;
        -h|--help) usage; exit 0 ;;
        *) PASSTHRU_ARGS+=("$@"); break ;;
    esac
done

if [[ "$SINGLE_ERROR_FLOOR" == true \
      && "$CLUMP2_ACCELERATION_FLOOR_ONLY" == true ]]; then
    echo "[ERROR] --single-error-floor and" \
         "--clump2-acceleration-floor-only are mutually exclusive"
    exit 1
fi

if [[ -z "$NUM_CHAINS" || -z "$CHAIN_WORKERS" ]]; then
    for ((i = 0; i < ${#PASSTHRU_ARGS[@]}; i++)); do
        case "${PASSTHRU_ARGS[$i]}" in
            --num-chains)
                if [[ -z "$NUM_CHAINS" ]]; then
                    j=$((i + 1)); NUM_CHAINS="${PASSTHRU_ARGS[$j]:-}"
                fi ;;
            --num-chains=*)
                if [[ -z "$NUM_CHAINS" ]]; then
                    NUM_CHAINS="${PASSTHRU_ARGS[$i]#*=}"
                fi ;;
            --chain-workers)
                if [[ -z "$CHAIN_WORKERS" ]]; then
                    j=$((i + 1)); CHAIN_WORKERS="${PASSTHRU_ARGS[$j]:-}"
                fi ;;
            --chain-workers=*)
                if [[ -z "$CHAIN_WORKERS" ]]; then
                    CHAIN_WORKERS="${PASSTHRU_ARGS[$i]#*=}"
                fi ;;
        esac
    done
fi

if [[ -z "$GALAXY" ]]; then
    echo "[ERROR] --galaxy is required. Choices: $ALL_GALS"; exit 1
fi
[[ -z "$DATASET" ]] && DATASET="$(config_value io dataset)"
[[ -z "$DATASET" ]] && DATASET="fiducial"
if [[ "$DATASET" == ,* || "$DATASET" == *, || "$DATASET" == *,,* ]]; then
    echo "[ERROR] --dataset contains an empty entry"; exit 1
fi
IFS=',' read -ra DATASETS <<< "$DATASET"
for dataset in "${DATASETS[@]}"; do
    case "$dataset" in
        original_published|fiducial|unpruned|clipped) ;;
        *) echo "[ERROR] --dataset must contain only original_published, fiducial, unpruned, or clipped"; exit 1 ;;
    esac
done
if [[ ${#DATASETS[@]} -gt 1 ]]; then
    if [[ "$ITERATIVE_CLIP" == true ]]; then
        echo "[ERROR] --iterative-clip-sigma requires only --dataset unpruned"
        exit 1
    fi
    for dataset in "${DATASETS[@]}"; do
        child_args=("${ORIG_ARGS[@]}")
        for ((i = 0; i + 1 < ${#child_args[@]}; i++)); do
            if [[ "${child_args[$i]}" == "--dataset" ]]; then
                child_args[$((i + 1))]="$dataset"
                break
            fi
        done
        echo "[submit] === dataset=$dataset ==="
        "$ROOT/packages/candel-maser/scripts/submit.sh" "${child_args[@]}"
    done
    exit 0
fi
RUNNER_ENV=(/usr/bin/env)
RUNNER_ENV_STR="/usr/bin/env"
if [[ "$TEMP_OUTPUT" == true ]]; then
    RUNNER_ENV+=("CANDEL_MEGAMASER_ROOT_OUTPUT=$TEMP_ROOT_OUTPUT")
    RUNNER_ENV_STR+=" CANDEL_MEGAMASER_ROOT_OUTPUT=$TEMP_ROOT_OUTPUT"
    echo "[submit] temporary output root: $ROOT/$TEMP_ROOT_OUTPUT/$DATASET"
fi
if [[ "$SAMPLER" != "mcmc" && "$SAMPLER" != "de" ]]; then
    echo "[ERROR] --sampler must be mcmc or de"; exit 1
fi
if [[ "$ITERATIVE_CLIP" == true && "$SAMPLER" != "de" ]]; then
    echo "[ERROR] --iterative-clip-sigma requires --sampler de"; exit 1
fi
if [[ "$ITERATIVE_CLIP" == true && "$DATASET" != "unpruned" ]]; then
    echo "[ERROR] --iterative-clip-sigma requires --dataset unpruned"; exit 1
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
    [[ ${#MCMC_ARGS[@]} -gt 0 ]] && bad_args+=("${MCMC_ARGS[@]}")
    [[ ${#MCMC_JOINT_ARGS[@]} -gt 0 ]] && bad_args+=("${MCMC_JOINT_ARGS[@]}")
    [[ ${#bad_args[@]} -gt 0 ]] && fail_if_args "--evidence" "${bad_args[@]}"
elif [[ "$JOINT_H0_MODE" == true ]]; then
    # Joint H0 modes are always one joint NUTS chain; --sampler is ignored.
    bad_args=()
    [[ ${#SINGLE_ARGS[@]} -gt 0 ]] && bad_args+=("${SINGLE_ARGS[@]}")
    [[ ${#INIT_ARGS[@]} -gt 0 ]] && bad_args+=("${INIT_ARGS[@]}")
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
    [[ ${#INIT_ARGS[@]} -gt 0 ]] && RUN_ARGS+=("${INIT_ARGS[@]}")
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
    if [[ "$FIX_GLOBALS" == true && ${#INIT_ARGS[@]} -gt 0 ]]; then
        RUN_ARGS+=("${INIT_ARGS[@]}")
    fi
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
    [[ "$ITERATIVE_CLIP" == true ]] && marker="iterative clipping complete"
    [[ "$JOINT_H0_MODE" == true ]] && marker="saved samples to"
    watcher=("$ROOT/packages/candel-maser/scripts/watch_and_resubmit.sh"
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
        "$ROOT/packages/candel-maser/scripts/submit.sh" "${submit_args[@]}"
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
if [[ "$EVIDENCE" == false && "$JOINT_H0_MODE" == false
      && "$SAMPLER" == "mcmc" ]]; then
    [[ -z "$NUM_CHAINS" ]] && \
        NUM_CHAINS="$(config_value inference num_chains)"
    [[ -z "$CHAIN_WORKERS" ]] && \
        CHAIN_WORKERS="$(config_value inference chain_workers)"
    if [[ ! "$NUM_CHAINS" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] --num-chains must be a positive integer"
        exit 1
    fi
    if [[ ! "$CHAIN_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] --chain-workers must be a positive integer"
        exit 1
    fi
    if [[ -z "$CPUS" ]]; then
        if (( NUM_CHAINS < CHAIN_WORKERS )); then
            CPUS="$NUM_CHAINS"
        else
            CPUS="$CHAIN_WORKERS"
        fi
    fi
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
    runner="candel_maser.run_joint_H0"
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
    ds_tag="${DATASET%%_*}"
    joint_label="joint H0"
    # Tag the job (hence the scheduler log filename) with selection and
    # reconstruction, matching the joint_H0 result-file convention; defaults
    # mirror run_joint_H0.py (redshift / none).
    job_name="maser_jointh0_${ds_tag}_${gal_tag}_${SELECTION:-redshift}_${RECONSTRUCTION:-none}"
    [[ "$DISTANCE_SOURCE" == "p20" ]] && job_name="${job_name}_p20"
    [[ -n "$LOO_DROPPED" ]] && job_name="${job_name}_loo${LOO_DROPPED}"
    if [[ "$LOCAL" == true ]]; then
        echo "Running $joint_label ($GALAXY) locally"
        cmd=("${RUNNER_ENV[@]}" JAX_PLATFORMS=cpu "$CANDEL_PYTHON" -u -m "$runner"
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
            pycmd="$RUNNER_ENV_STR $CANDEL_PYTHON -u -m $runner --galaxy $GALAXY"
        else
            pycmd="$RUNNER_ENV_STR JAX_PLATFORMS=cpu $CANDEL_PYTHON -u -m $runner --galaxy $GALAXY"
        fi
        [[ ${#RUN_ARGS[@]} -gt 0 ]] && pycmd+=" ${RUN_ARGS[*]}"
        # Keep stage-2 outputs and scheduler logs below the dataset's H0 folder.
        joint_root="$(config_value io root_output)"
        [[ "$TEMP_OUTPUT" == true ]] && joint_root="$TEMP_ROOT_OUTPUT"
        [[ -z "$joint_root" ]] && joint_root="results/Megamaser"
        submit_args=(--queue "$QUEUE" --mem "$MEM"
                     --name "$job_name"
                     --logdir "$ROOT/$joint_root/$DATASET/H0/logs")
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

RUNNER="candel_maser.run_maser"
# Short tag so the same galaxy can run on both datasets concurrently without
# colliding on job name or scheduler-log destination.
ds_tag="${DATASET%%_*}"

case "$SAMPLER" in
    de) JOB_PREFIX="maser_de_${ds_tag}" ;;
    mcmc) JOB_PREFIX="maser_mcmc_${ds_tag}" ;;
    *) JOB_PREFIX="maser_mcmc_${ds_tag}" ;;
esac

# Local runs are tee'd to a per-run log under the galaxy's output subdir, so the
# full transcript (sampler progress, the Reid/Pesce comparison table, JAX/XLA
# warnings) is saved beside that galaxy's HDF5 outputs. run_maser.py writes to
# <root_output>/<dataset>/<gal>/; derive root_output from config_maser.toml [io]
# (defaults to results/Megamaser only when that key is absent) so it can't drift.
# root_results resolves to the repo root (local_config.toml).
maser_root_output="$(
    awk -F'=' '
        /^\[/ { s = $1$2 }
        s == "[io]" && /^[[:space:]]*root_output[[:space:]]*=/ {
            gsub(/[" ]/, "", $2); print $2; exit
        }
    ' "$ROOT/packages/candel-maser/configs/config_maser.toml" 2>/dev/null || true
)"
[[ -z "$maser_root_output" ]] && maser_root_output="results/Megamaser"
[[ "$TEMP_OUTPUT" == true ]] && maser_root_output="$TEMP_ROOT_OUTPUT"
# run_maser.py/run_de_map.py namespace root_output by dataset, so mirror that
# here or the chain and log paths below point at the wrong dataset.
MASER_OUT="$ROOT/$maser_root_output/$DATASET"
stamp="$(date '+%Y%m%d_%H%M%S')"

if [[ "$EVIDENCE" == true ]]; then
    EVIDENCE_RUNNER="candel_maser.evidence_single_galaxy"
    init_strategy="$(chain_init_strategy)"
    for gal in $GALAXY; do
        suffix="blackjax_mcmc_rphi$(chain_variant_suffix \
            "$init_strategy" "$gal")"
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
        evidence_args=(--chain "$chain" --dataset "$DATASET")
        [[ -n "$GPU_MEM" ]] && evidence_args+=(--gpu-mem "$GPU_MEM")
        [[ -n "$SPOT_BATCH" ]] && evidence_args+=(--spot-batch "$SPOT_BATCH")
        [[ ${#PASSTHRU_ARGS[@]} -gt 0 ]] && evidence_args+=("${PASSTHRU_ARGS[@]}")
        if [[ "$LOCAL" == true ]]; then
            cmd=("${RUNNER_ENV[@]}" "$CANDEL_PYTHON" -u -m "$EVIDENCE_RUNNER" "$gal")
            cmd+=("${evidence_args[@]}")
            if [[ "$DRY" == true ]]; then
                printf '[dry]'; printf ' %q' "${cmd[@]}"; printf '\n'
            else
                "${cmd[@]}"
            fi
            continue
        fi
        echo "Submitting $gal evidence -> $CANDEL_CLUSTER:$QUEUE"
        pycmd="$RUNNER_ENV_STR $CANDEL_PYTHON -u -m $EVIDENCE_RUNNER $gal"
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
            --single-error-floor) variant_tag="${variant_tag}_singlefloor" ;;
            --clump2-acceleration-floor-only)
                variant_tag="${variant_tag}_accelfloor" ;;
            --da2-prior)         variant_tag="${variant_tag}_da2" ;;
        esac
    done
fi
[[ "$PHI_INTEGRATION" == "peak-partition" ]] && \
    variant_tag="${variant_tag}_peakpartition"

for gal in $GALAXY; do
    if [[ "$SKIP_DONE" == true ]]; then
        init_strategy="$(chain_init_strategy)"
        suffix="blackjax_mcmc_rphi$(chain_variant_suffix \
            "$init_strategy" "$gal")"
        outpath="$MASER_OUT/$gal/${gal}_${suffix}.hdf5"
        if [[ -f "$outpath" ]]; then
            echo "[skip-done] $gal: $outpath"
            continue
        fi
    fi

    if [[ "$LOCAL" == true ]]; then
        echo "Running $gal ($SAMPLER) locally"
        if [[ "$SAMPLER" == "mcmc" ]]; then
            cmd=("${RUNNER_ENV[@]}" JAX_PLATFORMS=cpu "$CANDEL_PYTHON" -u -m "$RUNNER"
                 "$gal" --sampler "$SAMPLER")
        else
            cmd=("${RUNNER_ENV[@]}" "$CANDEL_PYTHON" -u -m "$RUNNER"
                 "$gal" --sampler "$SAMPLER")
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
            term_columns="$(tput cols 2>/dev/null || true)"
            term_lines="$(tput lines 2>/dev/null || true)"
            term_env=()
            if [[ "$term_columns" =~ ^[1-9][0-9]*$ \
                  && "$term_lines" =~ ^[1-9][0-9]*$ ]]; then
                term_env=(COLUMNS="$term_columns" LINES="$term_lines")
            fi
            # pipefail (set at top) propagates the runner's exit status here.
            /usr/bin/env "${term_env[@]}" "${cmd[@]}" 2>&1 | tee "$logfile"
        fi
        continue
    fi

    echo "Submitting $gal ($SAMPLER) -> $CANDEL_CLUSTER:$QUEUE"
    if [[ "$SAMPLER" == "mcmc" ]]; then
        pycmd="$RUNNER_ENV_STR JAX_PLATFORMS=cpu $CANDEL_PYTHON -u -m $RUNNER $gal --sampler $SAMPLER"
    else
        pycmd="$RUNNER_ENV_STR $CANDEL_PYTHON -u -m $RUNNER $gal --sampler $SAMPLER"
    fi
    if [[ ${#RUN_ARGS[@]} -gt 0 ]]; then
        pycmd+=" ${RUN_ARGS[*]}"
    fi
    logdir="$MASER_OUT/$gal/logs"
    submit_args=(--queue "$QUEUE" --mem "$MEM" --name "${JOB_PREFIX}_${gal}${variant_tag}"
                 --logdir "$logdir")
    if [[ "$SAMPLER" != "mcmc" ]]; then
        submit_args=(--gpu "${submit_args[@]}")
        if [[ -n "$GPU_COUNT" && "$GPU_COUNT" -gt 1 ]]; then
            submit_args+=(--gpu-count "$GPU_COUNT")
        fi
    fi
    # MCMC CPU job: one process with bounded chain threads, not MPI. --cpus
    # requests N shared cores on one node (addqueue -s -n N); the -s -n 1xN
    # node-form grabs a whole node on glamdring.
    if [[ ${#extra_flags[@]} -gt 0 ]]; then
        submit_args+=("${extra_flags[@]}")
    fi
    if [[ ${#dry_flag[@]} -gt 0 ]]; then
        submit_args+=("${dry_flag[@]}")
    fi
    submit_job "${submit_args[@]}" -- $pycmd
done
