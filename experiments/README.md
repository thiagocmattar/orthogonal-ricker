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
declares the ordered config paths and delegates to the shared parent:

```python
from paper_exp.runner import run_launch

CONFIGS = (
    "experiments/NN-phase-tranche/run/CCC-first-case.yaml",
    "experiments/NN-phase-tranche/run/DDD-second-case.yaml",
)

if __name__ == "__main__":
    run_launch(__file__, CONFIGS)
```

A reviewed bounded execution exception may additionally declare tracked
operational authorization in that runner. The A1 runner retains dormant
worker-authorization metadata from a superseded `001`–`008` launch shape. It
did not cover configs `009`–`011` or authorize worker-slot execution for
those runs.

The parent requires the tuple to list every YAML config in `run/` exactly once,
holds one experiment lock, and runs configs serially by default. The completed
original three-cell A1 launch used a reviewed one-coordinator/two-A40
exception; Git history preserves that recipe. The later configs `004` and
`005` completed serially. High-LR configs `006`–`008` completed serially on one
A40 from clean commit `d410572`; very-high-LR configs `009`–`011` also
completed serially on one A40, with no `--worker-slot` arguments:

```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/01-a1-lr-screen/run/runner.py
```

All A1 configs `001`–`011` are immutable completed evidence and must never be
rerun. The runner lists all eleven A1 configs, and their exact attempts are
completed-reuse state; no A1 invocation is authorized. Other definitive
tranches remain serial unless a later reviewed policy says otherwise. Even a
one-config scientific tranche uses its runner. Never run case runners in
parallel, pack two workers onto one GPU, mix GPU types, or dispatch one launch
across Pods.

`00-infrastructure-smoke/run/00-smoke.yaml` is the only runner-free exception.
It is an infrastructure check, not a paper experiment or a scientific config
template. Do not add scientific scaffolds while
`docs/experiment_plan.md` says `Plan status: placeholder`, or for a case group
outside its reviewed scope. After the A1 decision was frozen, the plan returned
to placeholder with no reviewed case groups; later groups require exact-SHA
review before any config is materialized.

One preservation-only exception applies to already tracked history: an
immutable config may remain outside the current reviewed scope when the
experiment log identifies an exact coherent completed pretraining run and its
saved config snapshot matches. Integrity acceptance preserves evidence only;
it never authorizes materialization, retry, rerun, or launch.

## Config and Attempt Identity

Definitive configs are named `CCC-<case>.yaml`, where `CCC` is globally unique
and sequential across all scientific scaffolds. A config is immutable after its
first attempt starts. A scientific change, seed, budget, or diagnostic source
set therefore gets a new config; an infrastructure-only retry gets the next
attempt under the unchanged config.

Before allocating that number, resolve the condition in
`docs/experimental-design/cases.yaml` and apply its fingerprint/reuse contract.
There is exactly one physical config per scientific condition and seed across
all stages. A later stage records itself as a consumer of an existing config;
it does not copy the config into its scaffold or rerun seed 0 for promotion.

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

Runner restart and retry semantics are owned by
[`docs/runbook.md`](../docs/runbook.md#6-completion-failure-and-retry).

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
