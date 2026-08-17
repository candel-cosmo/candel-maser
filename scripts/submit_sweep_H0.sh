#!/bin/bash -l
# Submit the eleven joint-H0 measurements: distance/redshift selection with no
# velocity field, Carrick, or Manticore (six), plus no-selection baselines with
# uniform-in-volume, uniform-in-distance, and uniform-in-log(D_A) priors
# (five). Each variant is one joint NUTS chain over GALAXIES, routed through
# submit.sh --infer-H0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="$ROOT/submit.sh"

# Galaxies analysed jointly (comma-separated, or "all" = the five MCP H0
# galaxies). Override with --galaxy. NGC4258 is rejected by joint H0.
# GALAXIES="all"
GALAXIES="CGCG074-064,NGC5765b,NGC6264,NGC6323,UGC3789"
# GALAXIES="NGC6264,NGC6323"
RECONSTRUCTIONS="none,Carrick2015,ManticoreLocalCOLA"
# RECONSTRUCTIONS="Carrick2015"

# Full MCP H0 set, used to expand --galaxy all under --leave-one-out.
MCP_ALL="CGCG074-064,NGC5765b,NGC6264,NGC6323,UGC3789"

LOCAL=false
QUEUE=""
CPUS=""
MEM=""
GPU_MEM=""
NUM_WARMUP="3000"
NUM_SAMPLES="15000"
NUM_CHAINS="1"
MAX_TREE_DEPTH=""
DATASETS=""
DRY=false
YES=false
EXTRA=()
LEAVE_ONE_OUT=false
SELECTION="redshift"
SEL_EXPLICIT=false
ADD_QW=false
DISTANCE_SOURCE="candel"
TEMP_OUTPUT=false

usage() {
    cat <<EOF
Usage: $0 (--local | -q QUEUE) [--galaxy GAL,GAL,...|all] [--cpus N] [--mem GB] \\
          [--gpu-mem GB] [--num-warmup N] [--num-samples N] [--num-chains N] \\
          [--max-tree-depth N] [--dataset NAME[,NAME,...]] [--reconstruction LIST] \\
          [--distance-source candel|p20] \\
          [--temp-output] [--dry] [-y] \\
          [-- extra submit.sh args]

Runs the joint-H0 sweep over the GALAXIES set (default: $GALAXIES),
all galaxies analysed jointly with a shared H0:

  selected       : distance, redshift x $RECONSTRUCTIONS
  no selection   : volume, distance, log(D_A) priors (no reconstruction),
                   plus distance and log(D_A) priors with Carrick

= eleven submit.sh calls per dataset with the default reconstructions.

With --leave-one-out the grid is replaced by one joint job per dropped galaxy
and reconstruction at a fixed selection/warp config. The dropped galaxy is
visible in each output filename.

  --local            Run locally (submit.sh --local), one job at a time.
  -q, --queue QUEUE  Submit to the cluster queue (glamdring CPU: redwood|berg|cmb;
                     glamdring GPU: gpulong|cmbgpu|optgpu; arc: short|medium|long).
                     CPU queues run the joint chain as a CPU job; GPU queues on GPU.
  --galaxy LIST      Galaxies analysed jointly (comma list or all). Default: $GALAXIES.
  --reconstruction LIST
                     Comma list of reconstruction variants. Choices:
                     none,Carrick2015,ManticoreLocalCOLA. Default: $RECONSTRUCTIONS.
                     Non-none variants automatically sample --Vext.
                     Manticore uses the configured which_MAS field product.
  --cpus N           Forwarded to submit.sh; glamdring CPU joint uses -n 1xN.
  --mem GB           Forwarded to submit.sh; omit for the backend-aware default.
  --gpu-mem GB       Forwarded to submit.sh (GPU queues only).
  --num-warmup N     Forwarded to submit.sh; omit for the config default.
  --num-samples N    Forwarded to submit.sh; omit for the config default.
  --num-chains N     Forwarded to submit.sh; omit for the config default.
  --max-tree-depth N Forwarded to submit.sh (NUTS max tree depth).
  --dataset LIST     Comma-separated spot-table/distance-chain datasets.
  --distance-source candel|p20
                     Stage-1 distance posteriors (default: candel). p20 uses
                     the archived Dom files and removes their log(D_A) prior.
  --temp-output      Forward submit.sh --temp-output for every joint run.
  --leave-one-out    LOO mode: one joint job per dropped galaxy at a single
                     fixed config, instead of the eleven-run sweep. Needs >=2 galaxies
                     (--galaxy all expands to the five MCP galaxies).
  --selection SEL    LOO only: none|distance|redshift for the fixed config
                     (default redshift).
  --add-quadratic-warp
                     LOO only: enable the quadratic warp in the fixed config.
  --dry              Forward --dry: print the runner commands without submitting.
  -y, --yes          Do not ask for confirmation before submitting.
  -- extra...        Everything after -- is forwarded verbatim to every submit.sh
                     call (e.g. --time 24, --max-retries 4).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local) LOCAL=true; shift ;;
        -q|--queue) QUEUE="$2"; shift 2 ;;
        --galaxy) GALAXIES="$2"; shift 2 ;;
        --reconstruction|--reconstructions) RECONSTRUCTIONS="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --mem) MEM="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM="$2"; shift 2 ;;
        --num-warmup) NUM_WARMUP="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --num-chains) NUM_CHAINS="$2"; shift 2 ;;
        --max-tree-depth) MAX_TREE_DEPTH="$2"; shift 2 ;;
        --dataset) DATASETS="$2"; shift 2 ;;
        --distance-source) DISTANCE_SOURCE="$2"; shift 2 ;;
        --temp-output) TEMP_OUTPUT=true; shift ;;
        --leave-one-out) LEAVE_ONE_OUT=true; shift ;;
        --selection) SELECTION="$2"; SEL_EXPLICIT=true; shift 2 ;;
        --add-quadratic-warp) ADD_QW=true; shift ;;
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
if [[ "$LEAVE_ONE_OUT" == false
      && ( "$SEL_EXPLICIT" == true || "$ADD_QW" == true ) ]]; then
    echo "[ERROR] --selection/--add-quadratic-warp are only valid with --leave-one-out"
    echo "        (without it the fixed eleven-run configuration is used)"
    exit 1
fi
case "$DISTANCE_SOURCE" in
    candel|p20) ;;
    *) echo "[ERROR] --distance-source must be candel|p20"; exit 1 ;;
esac
if [[ "$LEAVE_ONE_OUT" == true ]]; then
    case "$SELECTION" in
        none|distance|redshift) ;;
        *) echo "[ERROR] --selection must be none|distance|redshift"; exit 1 ;;
    esac
fi
recon_arr=()
for recon in ${RECONSTRUCTIONS//,/ }; do
    case "$recon" in
        none|Carrick2015|ManticoreLocalCOLA) recon_arr+=("$recon") ;;
        *) echo "[ERROR] --reconstruction must contain only none,Carrick2015,ManticoreLocalCOLA"; exit 1 ;;
    esac
done
if [[ ${#recon_arr[@]} -eq 0 ]]; then
    echo "[ERROR] --reconstruction list is empty"; exit 1
fi
dataset_arr=("__default__")
if [[ -n "$DATASETS" ]]; then
    if [[ "$DATASETS" == ,* || "$DATASETS" == *, || "$DATASETS" == *,,* ]]; then
        echo "[ERROR] --dataset contains an empty entry"; exit 1
    fi
    IFS=',' read -ra dataset_arr <<< "$DATASETS"
    for dataset in "${dataset_arr[@]}"; do
        case "$dataset" in
            original_published|fiducial|unpruned|clipped) ;;
            *) echo "[ERROR] --dataset must contain only original_published,fiducial,unpruned,clipped"; exit 1 ;;
        esac
    done
fi

if [[ "$LOCAL" == true ]]; then
    target=(--local)
else
    target=(-q "$QUEUE")
fi

# Args shared by every variant (--galaxy/--selection vary per job, added below).
common=("--infer-H0")
[[ "$DISTANCE_SOURCE" != "candel" ]] && common+=(
    --distance-source "$DISTANCE_SOURCE")
[[ -n "$CPUS" ]] && common+=(--cpus "$CPUS")
[[ -n "$MEM" ]] && common+=(--mem "$MEM")
[[ -n "$GPU_MEM" ]] && common+=(--gpu-mem "$GPU_MEM")
[[ -n "$NUM_WARMUP" ]] && common+=(--num-warmup "$NUM_WARMUP")
[[ -n "$NUM_SAMPLES" ]] && common+=(--num-samples "$NUM_SAMPLES")
[[ -n "$NUM_CHAINS" ]] && common+=(--num-chains "$NUM_CHAINS")
[[ -n "$MAX_TREE_DEPTH" ]] && common+=(--max-tree-depth "$MAX_TREE_DEPTH")
[[ "$DRY" == true ]] && common+=(--dry)
[[ "$TEMP_OUTPUT" == true ]] && common+=(--temp-output)
[[ ${#EXTRA[@]} -gt 0 ]] && common+=("${EXTRA[@]}")

# Build the job list. Each entry is
# DATASET<US>GALSET<US>SELECTION<US>RECON<US>PRIOR<US>WARPFLAG<US>DROPPED.
jobs=()
if [[ "$LEAVE_ONE_OUT" == true ]]; then
    base="$GALAXIES"
    [[ "$base" == "all" ]] && base="$MCP_ALL"
    IFS=',' read -ra gal_arr <<< "$base"
    n=${#gal_arr[@]}
    if (( n < 2 )); then
        echo "[ERROR] --leave-one-out needs >=2 galaxies (got: $base)"; exit 1
    fi
    warp_flag=""
    [[ "$ADD_QW" == true ]] && warp_flag="--add-quadratic-warp"
    for dataset in "${dataset_arr[@]}"; do
        for recon in "${recon_arr[@]}"; do
            for ((i = 0; i < n; i++)); do
                sub=()
                for ((j = 0; j < n; j++)); do
                    if [[ $j -ne $i ]]; then sub+=("${gal_arr[j]}"); fi
                done
                sub_csv="$(IFS=,; echo "${sub[*]}")"
                jobs+=("${dataset}"$'\x1f'"${sub_csv}"$'\x1f'"${SELECTION}"$'\x1f'"${recon}"$'\x1f'"volume"$'\x1f'"${warp_flag}"$'\x1f'"${gal_arr[i]}")
            done
        done
    done
else
    for dataset in "${dataset_arr[@]}"; do
        for sel in distance redshift; do
            for recon in "${recon_arr[@]}"; do
                jobs+=("${dataset}"$'\x1f'"${GALAXIES}"$'\x1f'"${sel}"$'\x1f'"${recon}"$'\x1f'"volume"$'\x1f')
            done
        done
        for prior in volume distance log-distance; do
            jobs+=("${dataset}"$'\x1f'"${GALAXIES}"$'\x1f'"none"$'\x1f'"none"$'\x1f'"${prior}"$'\x1f')
        done
        for recon in "${recon_arr[@]}"; do
            if [[ "$recon" == "Carrick2015" ]]; then
                for prior in distance log-distance; do
                    jobs+=("${dataset}"$'\x1f'"${GALAXIES}"$'\x1f'"none"$'\x1f'"${recon}"$'\x1f'"${prior}"$'\x1f')
                done
            fi
        done
    done
fi

echo "[sweep] target: ${target[*]} | joint H0"
echo "[sweep] datasets: ${DATASETS:-config default}"
echo "[sweep] distance source: $DISTANCE_SOURCE"
if [[ "$LEAVE_ONE_OUT" == true ]]; then
    echo "[sweep] mode: leave-one-out (single config per dropped galaxy)"
    echo "[sweep] config: selection=$SELECTION" \
         "warp=$([[ "$ADD_QW" == true ]] && echo on || echo off)" \
         "reconstruction=$RECONSTRUCTIONS"
    echo "[sweep] jobs: drop-one subsets x ${#recon_arr[@]} reconstruction(s) x ${#dataset_arr[@]} dataset(s) = ${#jobs[@]} joint jobs"
    echo "[sweep] base galaxies: $base"
else
    echo "[sweep] jobs: selected variants + no-selection prior baselines x ${#dataset_arr[@]} dataset(s) = ${#jobs[@]} joint jobs"
    echo "[sweep] galaxies (joint): $GALAXIES"
fi
echo "[sweep] will run:"
for spec in "${jobs[@]}"; do
    IFS=$'\x1f' read -r dataset galset sel recon prior warp dropped <<< "$spec"
    args=("${common[@]}" --galaxy "$galset" --selection "$sel"
          --reconstruction "$recon" --distance-prior "$prior")
    [[ "$dataset" != "__default__" ]] && args+=(--dataset "$dataset")
    [[ "$recon" != "none" ]] && args+=(--Vext)
    [[ -n "$warp" ]] && args+=("$warp")
    [[ -n "$dropped" ]] && args+=(--leave-one-out-dropped "$dropped")
    printf '  bash %q' "$SUBMIT"
    printf ' %q' "${target[@]}" "${args[@]}"
    printf '\n'
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
for spec in "${jobs[@]}"; do
    IFS=$'\x1f' read -r dataset galset sel recon prior warp dropped <<< "$spec"
    args=("${common[@]}" --galaxy "$galset" --selection "$sel"
          --reconstruction "$recon" --distance-prior "$prior")
    [[ "$dataset" != "__default__" ]] && args+=(--dataset "$dataset")
    [[ "$recon" != "none" ]] && args+=(--Vext)
    [[ -n "$warp" ]] && args+=("$warp")
    [[ -n "$dropped" ]] && args+=(--leave-one-out-dropped "$dropped")
    echo
    echo "[sweep] === dataset=${dataset/__default__/config-default} galaxy=$galset selection=$sel prior=$prior reconstruction=$recon vext=$([[ "$recon" != "none" ]] && echo on || echo off) warp=${warp:-off} ==="
    if ! bash "$SUBMIT" "${target[@]}" "${args[@]}"; then
        echo "[sweep] WARNING: (galaxy=$galset selection=$sel" \
             "reconstruction=$recon warp=${warp:-off}) failed" >&2
        fail=1
    fi
done

if [[ $fail -ne 0 ]]; then
    echo "[sweep] one or more variants failed" >&2
fi
exit $fail
