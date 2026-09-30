# candel-maser

Megamaser disk model and H0 pipeline for CANDEL. A probe package for
[CANDEL](https://github.com/candel-cosmo/CANDEL): it imports the core `candel`
library and provides its own runners.

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

- `candel_maser/` - the package
- `configs/` - run configurations
- `scripts/` - preprocessing, mocks and submission helpers
- `papers/` - scripts and notebooks behind each paper (MMH0)
- `tests/` - tests (`pytest`)
- `docs/` - workflow notes, starting with `docs/README.md`

## Run

```bash
python -m candel_maser.run_maser --help
```

## License

MIT; see `LICENSE`.
