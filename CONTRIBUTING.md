# Contributing

This repository is a lean experiment harness. Changes should directly help run,
reproduce, compare, or explain experiments.

## Development setup

Use Python 3.10 or newer in a virtual environment:

```bash
python -m pip install -e ".[dev]"
make test
make check
```

CI installs dependencies through `constraints/requirements-ci.txt` on Linux
and Windows. Dependency updates must update that snapshot intentionally and
pass the full matrix.

`make test` removes its repository-local pytest scratch directory even when the
test process fails. Generated datasets, run outputs, logs, and figures remain
ignored unless a release plan explicitly selects them.

## Change expectations

- Keep code, configs, documentation, result records, and plot labels in English.
- Give every experiment a committed config before launch.
- Preserve random initialization for Pythia pretraining unless a continuation
  or fine-tuning experiment explicitly calls for checkpoint weights.
- Keep naive pressure and Adam-step orthogonal pressure distinct in configs,
  metrics, and explanations.
- Do not invent results or paper claims. Use `TODO:` when required information
  is not yet available.
- Add tests for behavior changes and keep plotting inputs explicit.
- Do not commit credentials, machine-local paths, raw datasets, checkpoints, or
  ad hoc generated artifacts.

Before opening a pull request, run `make test` and `make check`, describe the
scope of the change, and note any command, config, or artifact needed to verify
it. Keep pull requests focused; additional abstractions or files should have a
clear reproducibility benefit.
