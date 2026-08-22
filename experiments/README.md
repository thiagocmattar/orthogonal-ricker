# Experiment Scaffolds

`experiments/` is the single home for launch recipes, raw attempts, and
generated figures. Each plan-defined tranche owns one chronological scaffold:

```text
experiments/NN-<phase>-<tranche>/
  run/
    runner.py
    CCC-<case>.yaml
  raw/
    CCC-<case>/
      001-YYYYMMDD-HHMMSS-xxxxxxxx/
  figs/
    01-descriptive-name.pdf
```

Use lowercase ASCII letters, digits, and hyphens in the scaffold name. `NN` is
a globally unique, sequential two-digit tranche prefix. Prefix `00` is reserved
for `00-infrastructure-smoke`; scientific tranches start at `01`.

This scaffold is the ownership boundary. Do not create repository-level
`configs/`, `runners/`, `results/`, `figures/`, or `run-logs/` trees.
Shared immutable inputs and token caches remain under `data/` because they can
serve more than one tranche.

## Tracked Recipe

Everything needed to reproduce a tranche belongs in its `run/` directory and
is tracked by Git. A scientific scaffold contains exactly one thin
`run/runner.py` and all of its immutable configs directly beside it. The runner
declares only the ordered config paths and delegates to the shared parent:

```python
from paper_exp.runner import run_launch

CONFIGS = (
    "experiments/NN-phase-tranche/run/CCC-first-case.yaml",
    "experiments/NN-phase-tranche/run/DDD-second-case.yaml",
)

if __name__ == "__main__":
    run_launch(__file__, CONFIGS)
```

The parent requires the tuple to list every YAML config in `run/` exactly once,
holds one experiment lock, runs configs serially in increasing prefix order,
and stops on the first failure. Even a one-config scientific tranche uses its
runner. Never run case runners in parallel.

`00-infrastructure-smoke/run/00-smoke.yaml` is the only runner-free exception.
It is an infrastructure check, not a paper experiment or a scientific config
template. Do not add scientific scaffolds while
`docs/experiment_plan.md` says `Plan status: placeholder`.

## Config and Attempt Identity

Definitive configs are named `CCC-<case>.yaml`, where `CCC` is globally unique
and sequential across all scientific scaffolds. A config is immutable after its
first attempt starts. A scientific change, seed, budget, or diagnostic source
set therefore gets a new config; an infrastructure-only retry gets the next
attempt under the unchanged config.

Every config's `output.dir` must be the portable repository-relative path to
its own scaffold's `raw/` directory:

```yaml
output:
  dir: experiments/NN-phase-tranche/raw
```

The config stem becomes the raw result-group ID. Attempts use a sequential
prefix, UTC timestamp, and short unique suffix:

```text
experiments/NN-phase-tranche/raw/CCC-case/001-YYYYMMDD-HHMMSS-xxxxxxxx/
```

Diagnostic selections pin `tranche_id`, `config_id`, and `run_id`; this makes
cross-tranche sources exact without scanning for a latest run. Post-hoc outputs
stay under the source or explicitly owning scaffold recorded by their recipe.

## Raw Lifecycle

Each attempt starts by durably writing `config.yaml` and a `status: running`
`manifest.json` with launch Git provenance. A completed common envelope is:

```text
config.yaml
manifest.json
metrics.json
predictions.jsonl
```

Training adds `events.jsonl` and, when configured, `checkpoints/final/`.
Specialized workflows add their documented primary artifact. The lifecycle is
`running -> completed` or `running -> failed`; incomplete and failed attempts
are never silently selected. A retry never overwrites an earlier attempt.

A source is consumable only when its config, scaffold, run, terminal status,
and required artifacts agree. Statusless historical records remain historical
evidence only under the explicit acceptance rules in the reviewed plan.

## Figures and Provenance

Generated figures go only in the owning scaffold's `figs/` directory. A
single-run diagnostic figure belongs to the source run's scaffold. A
cross-tranche paper figure belongs to the later plan-defined scaffold whose
tracked recipe pins every input.

Use a unique sequential figure prefix within the owning scaffold:

```text
01-descriptive-name.pdf
01-descriptive-name.png
01-descriptive-name.provenance.json
```

Every export must be regenerable from exact saved identities and include its
provenance sidecar. See `docs/plotting.md` for the rendering contract.

## Git Policy

Git tracks:

- this contract and every scaffold directory keeper;
- all `run/` contents, including runner and config files;
- shared harness, documentation, and plotting source.

Git ignores generated payloads beneath `raw/` and `figs/`. This prevents logs,
checkpoints, event streams, and rendered outputs from entering commits while
the tracked `.gitkeep` files preserve the scaffold. If a reviewed release needs
a compact derived artifact in Git, document it and add it explicitly; never
weaken the default ignore boundary for raw or heavy output.
