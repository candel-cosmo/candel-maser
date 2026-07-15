#!/bin/bash -l
# Compile background_info/fit_disk_Reid_reflection once and run one or more
# independent chains locally.
#
# Reid's native "M-H strands" (num_walkers>1 in fit_disk_control.inp) run
# SEQUENTIALLY inside one process -- there is no internal parallelism.  This
# script instead launches --chains independent one-strand processes in the
# background (one core each) and waits for all of them, giving genuine
# parallel chains for a real between-chain Gelman-Rubin check.
#
# use_gibbs, p_reflect, n_inner, use_gcov and track_lat are control-file
# fields (read by control_parameters), not hardwired constants, so a single
# compiled binary covers every variant:
#   --gibbs F --reflect 0                    -> original sampler, bit-for-bit
#   --gibbs T --reflect 0                    -> Gibbs restructuring only
#   --gibbs T --reflect 0.25 (default)       -> Gibbs + azimuthal reflection
#   --gibbs T --reflect 0.25 --global-cov T  -> + adaptive-covariance globals
#
# Usage:
#   run_gibbs_chains.sh <control_template> <data_file> <output_dir> [options]
#
# <control_template> is any existing fit_disk_control.inp for the target
# galaxy (Ho/Mbh/... priors, Ho range, floors). Only the burn-in trial count
# (line 2), the trials/walkers/Ho-range line (line 3), the seed (line 5), and
# the trailing use_gibbs/p_reflect line are overridden here; everything else
# is taken verbatim from the template.  The checked-in
# background_info/fit_disk_Reid_reflection/fit_disk_control.inp (NGC4258) can
# be used as a starting template.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOURCE="$ROOT/background_info/fit_disk_Reid_reflection/fit_disk_v24d_unblinded.f"

CHAINS=1
GIBBS=T
REFLECT=0.25
N_INNER=20
GCOV=F
ETA=T
MATCH_PRIORS=F
GALAXY=""
TRACK_LATENTS=F
WRITE_THIN=10
WARMUP=1000000
SAMPLES=1000000
SEED=47351937
SEED_STEP=1009

usage() {
    cat <<EOF
Usage: $0 <control_template> <data_file> <output_dir> [options]

Options:
  --chains N       number of independent parallel chains (default: $CHAINS)
  --gibbs T|F      enable the Metropolis-within-Gibbs restructuring (default: $GIBBS)
  --reflect P      azimuthal reflection-move probability, 0 disables (default: $REFLECT)
  --n-inner N      per-spot latent sweeps per Gibbs iteration (CANDEL n_inner; default: $N_INNER)
  --global-cov T|F adaptive-covariance global-block proposal, needs --gibbs T (default: $GCOV)
  --eta T|F        reparameterize the BH mass as eta = log10(Mbh/D_A)
                   (CANDEL's mass coordinate) and target a measure flat
                   in (D_A, eta) via the -2 ln Ho Jacobian; fort.7 keeps
                   the Mbh column (default: $ETA)
  --track-latents T|F write the fort.74 per-spot (r,dphi) latent diagnostic
                   chains (~1.5x the fort.7 volume) (default: $TRACK_LATENTS)
  --galaxy NAME    galaxy name; only used by --match-priors orig to pick that
                   galaxy's published error floors (default: generic fallback)
  --match-priors T|F|orig  error-floor treatment (default: $MATCH_PRIORS):
                   T    = CANDEL Gaussian floor priors (config_maser.toml
                          [model.priors]); floors are SAMPLED (Pesce-style),
                          re-initialised at the CANDEL prior means since
                          fit_disk centres the Gaussian on the value column
                   F    = template error-floor lines (21-25) verbatim
                   orig = original published floors, FIXED (prior_unc=0 =>
                          frozen, NOT sampled), selected per --galaxy from that
                          galaxy's MCP disk paper (UGC3789=Reid 2013,
                          NGC6264=Kuo 2013, NGC6323=Kuo 2015); an unsupported
                          galaxy is a hard error
  --write-thin N   write only every Nth STORED sample to fort.7/fort.74
                   (the internal Ho-M bootstrap still sees every stored
                   sample; this only shrinks the output files)
                   (default: $WRITE_THIN)
  --warmup N       primary burn-in trials (default: $WARMUP)
  --samples N      production trials stored per chain (default: $SAMPLES)
  --seed N         seed for chain 0 (default: $SEED)
  --seed-step N    seed_i = seed + i*step (default: $SEED_STEP)
  -h, --help
EOF
    exit 1
}

[[ $# -ge 3 ]] || usage
CONTROL_TEMPLATE=$1; shift
DATA_FILE=$1; shift
OUTPUT_DIR=$1; shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chains) CHAINS=$2; shift 2 ;;
        --gibbs) GIBBS=$2; shift 2 ;;
        --reflect) REFLECT=$2; shift 2 ;;
        --n-inner) N_INNER=$2; shift 2 ;;
        --global-cov) GCOV=$2; shift 2 ;;
        --eta) ETA=$2; shift 2 ;;
        --match-priors) MATCH_PRIORS=$2; shift 2 ;;
        --galaxy) GALAXY=$2; shift 2 ;;
        --track-latents) TRACK_LATENTS=$2; shift 2 ;;
        --write-thin) WRITE_THIN=$2; shift 2 ;;
        --warmup) WARMUP=$2; shift 2 ;;
        --samples) SAMPLES=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --seed-step) SEED_STEP=$2; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

[[ "$CHAINS" =~ ^[0-9]+$ && "$CHAINS" -ge 1 ]] || { echo "[ERROR] --chains must be a positive integer, got: $CHAINS" >&2; exit 1; }
[[ "$WRITE_THIN" =~ ^[0-9]+$ && "$WRITE_THIN" -ge 1 ]] || { echo "[ERROR] --write-thin must be a positive integer, got: $WRITE_THIN" >&2; exit 1; }
[[ "$MATCH_PRIORS" =~ ^(T|F|orig)$ ]] || { echo "[ERROR] --match-priors must be T, F, or orig, got: $MATCH_PRIORS" >&2; exit 1; }
[[ -f "$CONTROL_TEMPLATE" ]] || { echo "[ERROR] missing control template: $CONTROL_TEMPLATE" >&2; exit 1; }
[[ -f "$DATA_FILE" ]] || { echo "[ERROR] missing data file: $DATA_FILE" >&2; exit 1; }
[[ -f "$SOURCE" ]] || { echo "[ERROR] missing Fortran source: $SOURCE" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
BINARY="$OUTPUT_DIR/fit_disk_x"
echo "Compiling $SOURCE ..."
gfortran -O2 -std=legacy -fno-automatic -o "$BINARY" "$SOURCE"

HO_RANGE="$(awk 'NR==3{print $3, $4}' "$CONTROL_TEMPLATE")"
STEP_FRACTION="$(awk 'NR==5{print $1}' "$CONTROL_TEMPLATE")"
[[ -n "$HO_RANGE" && -n "$STEP_FRACTION" ]] || { echo "[ERROR] could not read Ho range (line 3) / step size (line 5) from $CONTROL_TEMPLATE" >&2; exit 1; }

# Template post_unc (3rd field) of the five error-floor lines (params 16-20,
# template lines 21-25), reused when --match-priors rewrites their priors.
if [[ "$MATCH_PRIORS" == "T" ]]; then
    read -r PU16 PU17 PU18 PU19 PU20 <<< "$(awk 'NR>=21 && NR<=25 {printf "%s ", $3}' "$CONTROL_TEMPLATE")"
    [[ -n "$PU20" ]] || { echo "[ERROR] could not read floor post_unc (template lines 21-25) from $CONTROL_TEMPLATE" >&2; exit 1; }
fi

echo "Launching $CHAINS chain(s): gibbs=$GIBBS reflect=$REFLECT gcov=$GCOV eta=$ETA match_priors=$MATCH_PRIORS warmup=$WARMUP samples=$SAMPLES write_thin=$WRITE_THIN"

pids=()
chain_dirs=()
for ((i = 0; i < CHAINS; i++)); do
    chain_seed=$((SEED + i * SEED_STEP))
    chain_dir="$OUTPUT_DIR/chain_$(printf '%02d' "$i")_seed_${chain_seed}"
    mkdir -p "$chain_dir"
    cp "$DATA_FILE" "$chain_dir/fit_disk_data.inp"
    cp "$BINARY" "$chain_dir/fit_disk_x"

    {
        sed -n '1p' "$CONTROL_TEMPLATE"
        printf '%12d                      ! Total "burn-in" trials (done in stages of 100,000)\n' "$WARMUP"
        printf '%12d     1  %s ! Total # trials (stored); # of M-H strands (parallelism is across processes here, not strands); Ho range\n' "$SAMPLES" "$HO_RANGE"
        sed -n '4p' "$CONTROL_TEMPLATE"
        printf '    %s -%d            ! Initial McMC parameter step size in burnin phase; random number seed\n' "$STEP_FRACTION" "$chain_seed"
        if [[ "$MATCH_PRIORS" == "T" ]]; then
            sed -n '6,20p' "$CONTROL_TEMPLATE"
            # CANDEL floor priors (value column = Gaussian prior centre):
            printf '   0.010       0.005     %s ! x floor (mas); CANDEL prior N(10,5) uas\n' "$PU16"
            printf '   0.010       0.005     %s ! y floor (mas); CANDEL prior N(10,5) uas\n' "$PU17"
            printf '   2.0         1.0       %s ! Vsys floor (km/s); CANDEL prior N(2,1)\n' "$PU18"
            printf '   2.0         1.0       %s ! Vhv floor (km/s); CANDEL prior N(2,1)\n' "$PU19"
            printf '   0.3         0.15      %s ! Acc floor (km/s/yr); CANDEL prior N(0.3,0.15)\n' "$PU20"
        elif [[ "$MATCH_PRIORS" == "orig" ]]; then
            sed -n '6,20p' "$CONTROL_TEMPLATE"
            # Original published error floors: FIXED (prior_unc=0 => frozen, NOT
            # sampled), taken per galaxy from its MCP disk paper. Reid's model
            # splits the velocity floor (systemic vs high-vel) but applies a
            # SINGLE scalar acceleration floor to all acceleration data.
            case "$GALAXY" in
                UGC3789)  # Reid et al. 2013 (MCP IV, arXiv:1207.7292)
                    printf '   0.010       0         0 ! x floor (mas); FIXED (Reid 2013, 0.01 mas)\n'
                    printf '   0.010       0         0 ! y floor (mas); FIXED (Reid 2013, 0.01 mas)\n'
                    printf '   1.0         0         0 ! Vsys floor (km/s); FIXED (Reid 2013)\n'
                    printf '   0.3         0         0 ! Vhv floor (km/s); FIXED (Reid 2013)\n'
                    printf '   0.57        0         0 ! Acc floor (km/s/yr); FIXED (Reid 2013)\n'
                    ;;
                NGC6264)  # Kuo et al. 2013 (MCP V, arXiv:1207.7273)
                    printf '   0.008       0         0 ! x floor (mas); FIXED (Kuo 2013, 8 uas)\n'
                    printf '   0.016       0         0 ! y floor (mas); FIXED (Kuo 2013, 16 uas)\n'
                    printf '   1.0         0         0 ! Vsys floor (km/s); FIXED (Kuo 2013)\n'
                    printf '   0.3         0         0 ! Vhv floor (km/s); FIXED (Kuo 2013)\n'
                    printf '   0.5         0         0 ! Acc floor (km/s/yr); FIXED (Kuo 2013: systemic 0.3-0.7, HV 1.0; single scalar ~mid systemic)\n'
                    ;;
                NGC6323)  # Kuo et al. 2015 (MCP VI, arXiv:1411.5106)
                    printf '   0.010       0         0 ! x floor (mas); FIXED (Kuo 2015)\n'
                    printf '   0.010       0         0 ! y floor (mas); FIXED (Kuo 2015)\n'
                    printf '   1.8         0         0 ! Vsys floor (km/s); FIXED (Kuo 2015)\n'
                    printf '   1.8         0         0 ! Vhv floor (km/s); FIXED (Kuo 2015)\n'
                    printf '   0.0         0         0 ! Acc floor (km/s/yr); FIXED ~0 (Kuo 2015 used measured errs; code clamps to 0.001)\n'
                    ;;
                *)
                    echo "[ERROR] --match-priors orig has no published error floors for galaxy '$GALAXY'; supported: UGC3789, NGC6264, NGC6323" >&2
                    exit 1
                    ;;
            esac
        else
            sed -n '6,25p' "$CONTROL_TEMPLATE"
        fi
        printf '    %s      %s      %d      %s      %s      %s      %d ! use_gibbs (T/F); p_reflect (0 disables); n_inner latent sweeps; use_gcov (T/F); track_lat fort.74 (T/F); use_eta (T/F); write_thin (fort.7/fort.74 cadence)\n' "$GIBBS" "$REFLECT" "$N_INNER" "$GCOV" "$TRACK_LATENTS" "$ETA" "$WRITE_THIN"
    } > "$chain_dir/fit_disk_control.inp"

    ( cd "$chain_dir" && ./fit_disk_x > run.stdout 2>&1 ) &
    pids+=("$!")
    chain_dirs+=("$chain_dir")
    echo "  launched chain $i seed=$chain_seed pid=$! dir=$chain_dir"
done

failed=0
for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
        failed=1
        echo "[ERROR] chain $idx failed: ${chain_dirs[$idx]}" >&2
        tail -n 40 "${chain_dirs[$idx]}/run.stdout" >&2 || true
    fi
done

echo
echo "chain_dirs:"
printf '  %s\n' "${chain_dirs[@]}"

if [[ "$failed" -ne 0 ]]; then
    echo "[ERROR] at least one chain failed; see logs above." >&2
    exit 1
fi
echo "All $CHAINS chain(s) finished. Each chain_dir above has its own fort.7 (chain) and run.stdout (log)."
