# candel-maser

Megamaser disk model and H0 pipeline for CANDEL. A probe package for
[CANDEL](https://github.com/candel-cosmo/CANDEL), part of the
[candel-cosmo](https://github.com/candel-cosmo) organisation. It imports the
core `candel` library; the core never imports it. See the
[CANDEL README](https://github.com/candel-cosmo/CANDEL#how-the-repositories-fit-together)
for how the repositories fit together.

## What it provides

Spot-level warped-disk model for water megamasers (NGC 5765b, NGC 6264,
NGC 6323, UGC 3789, CGCG 074-064, NGC 4258) and a two-stage $H_0$ pipeline:

1. per-galaxy disk fit — differential-evolution MAP (`run_de_map.py`) then
   BlackJAX MCMC (`run_maser.py`) for the angular-diameter distance;
2. joint $H_0$ from the per-galaxy distance chains with peculiar velocities
   (`run_joint_H0.py`).

Unlike the other probe packages this one does not register a `which_run`; it
has its own runners and uses the core for fields, cosmography and inference
utilities. `docs/README.md` is the script guide.

## Install

Clone this repository next to the CANDEL core and install both, core first:

```bash
git clone https://github.com/candel-cosmo/CANDEL.git
git clone https://github.com/candel-cosmo/candel-maser.git
cd CANDEL
python -m venv venv_candel && source venv_candel/bin/activate
pip install -e .
pip install --no-deps -e ../candel-maser
```

Data, results and the machine-local `local_config.toml` live in the CANDEL
checkout. Python code finds it through the installed `candel`
(`candel.util.CANDEL_ROOT`); shell scripts use `$CANDEL_ROOT`, defaulting to
`../CANDEL`.

## Layout

- `candel_maser/` — the package
- `configs/` — run configurations
- `scripts/` — preprocessing, mocks and submission helpers
- `papers/` — scripts and notebooks behind each paper
- `tests/` — tests (`pytest`)
- `docs/` — script guide and reproduction notes

## Papers

- `papers/MMH0/` — A reanalysis of the megamaser Hubble constant, [arXiv:2609.17684](https://arxiv.org/abs/2609.17684)

## Run

```bash
python -m candel_maser.run_maser NGC5765b              # single galaxy, MCMC
bash scripts/submit.sh -q cmbgpu --galaxy NGC5765b --sampler mcmc
python -m candel_maser.run_maser --help
```

Sweeps: `scripts/submit_sweep.sh` (disk fits) and `scripts/submit_sweep_H0.sh` (joint H0).

## License

MIT; see `LICENSE`.
