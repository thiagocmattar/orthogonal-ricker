# Orthogonal Sparsity Pressure

This repository is a lean experiment harness for studying activation sparsity
during random-initialized language-model pretraining. It contains reusable
training methods, diagnostics, run lifecycle controls, and publication-quality
plotting utilities.

## Status

[`docs/experimental-design/`](docs/experimental-design/README.md) contains the
modular proposal for the lean 14M discovery and 70M/410M replication program;
its [executive brief](docs/experimental-design/executive.md) is advisor-facing.
The sole scientific-scope authority is
[`docs/experiment_plan.md`](docs/experiment_plan.md). All eleven A1 configs
`001`–`011` are completed, eligible, valid evidence and must never be rerun.
The exact selection rule chose `6.4e-2` from config `008-a1-lr-6p4e-2`, run
`001-20260826-190546-4df1c441`, at `checkpoints/final`; `lr_14m` is frozen to
that value. Configs `009`–`011` completed serially on one A40. This is the best
tested setting for seed 0 at the fixed 400M-token horizon, not evidence of a
global, horizon-independent, or converged optimum.

The tracked tree contains the definitive A1 recipes and their evidence index;
raw attempts remain ignored and are addressed by exact identities from
[`docs/experiment_log.md`](docs/experiment_log.md). The exact A2 groups
`[A2-relu-control, A2-l1-screen]` were reviewed at design commit
`3a4b047b1f4712d07b32314461913aae09cc46a7`. Training configs `012`–`017`
and diagnostics `018`–`020` are completed, accepted, immutable evidence.
Diagnostic config `020` completed locally as run
`001-20260828-130123-cefae393` at launch Git
`5f1f5a7aa079d46c4d2855ee7c2a16d027abe37d`; its 30-point primary artifact
has SHA-256 `9322c54c61fa68e5edb5b191d417a77e5c985369d2a10ed66b5cc05676322f23`.
A3 is deferred in the backlog, with no
config materialization, calibration, cloud spending, or launch authority.

## Scientific Invariants

- Pythia experiments are pretraining runs unless a plan explicitly says
  otherwise.
- `model.architecture` identifies the architecture/config source.
- `model.initialization: random` means released checkpoint weights are not
  loaded.
- Naive loss pressure and Adam-step orthogonal pressure are separate methods,
  with separate config names, metrics, and interpretation.
- `model.topology_id` selects a canonical set of active transformer gate ports;
  `model.site_gate` selects the operator and any `kappa`. Optimizer and
  activation-pressure settings remain separate. See the exact site and
  topology registries in [`docs/methods.md`](docs/methods.md).
- Exact zeros, near-zero activations, logical zero-product opportunities, and
  measured runtime speedups are different quantities.
- Every launched experiment has an immutable numbered config and a durable run
  artifact envelope.

See [`docs/methods.md`](docs/methods.md) and
[`docs/diagnostics.md`](docs/diagnostics.md) for the complete contracts.

## Install

Create an isolated Python environment, then install the package and development
dependencies:

```bash
make install
```

[`constraints/requirements-ci.txt`](constraints/requirements-ci.txt) records
the exact dependency snapshot exercised by the Linux/Python 3.11 and
Windows/Python 3.12 CI matrix. Release experiments should install with that
constraint file unless the reviewed plan deliberately records and tests a new
snapshot.

Run the repository checks:

```bash
make test
make check
make smoke
```

`make` is a convenience layer. On systems without GNU Make, use the equivalent
cross-platform commands:

```bash
python -m pip install -c constraints/requirements-ci.txt -e ".[dev]"
python scripts/run_tests.py
python -m paper_exp.cli check --strict
python -m paper_exp.cli smoke --config experiments/00-infrastructure-smoke/run/00-smoke.yaml
```

The smoke config is infrastructure-only. It writes and validates a tiny local
artifact envelope without downloading a model or dataset, and it does not
authorize or stand in for a scientific run.

## Experiment Workflow

After the relevant case groups enter the definitive plan's reviewed scope:

1. Split the plan into ordered launch tranches, including separate screening
   and promotion tranches when later configs depend on earlier evidence.
2. Resolve the case group and reuse aliases in
   `docs/experimental-design/cases.yaml`; never allocate a second config for an
   existing condition fingerprint and seed.
3. Give each tranche one chronological scaffold containing its tracked runner
   and configs plus its ignored raw outputs and figures.
4. Commit the reviewed runner and configs before launch.
5. Prepare the declared dataset cache.
6. Run a calibration when no reliable same-hardware throughput estimate exists.
   A1 historically calibrated distinct configs concurrently under one
   coordinator and lock, with one process per distinct homogeneous GPU.
7. Execute the case runner; it reuses coherent completed configs and runs new
   cases serially by default under one lock. A reviewed infrastructure retry
   requires the explicit `--retry-failed` flag.
8. Verify terminal artifacts before running diagnostics or plotting.
9. Record accepted evidence in the experiment and paper maps.

Data preparation and solo throughput calibration remain explicit
single-config operations. The path below shows the naming pattern; it is not
an allocated launch:

```bash
make prepare-data CONFIG=experiments/NN-phase-tranche/run/CCC-case.yaml
make calibrate CONFIG=experiments/NN-phase-tranche/run/CCC-case.yaml
```

The historical A1 calibration passed its three distinct configs and two
explicit GPU slots to one CLI coordinator; see the runbook. That operational
timing workflow was not reused as pretraining evidence. The separately
approved definitive A1 launch later used the same bounded two-GPU decomposition
through its case runner and is now complete; neither record authorizes a rerun.
Configs `004` and `005` also ran serially. The high-LR cells `006`–`008`
completed serially on one A40 from clean commit `d410572` with the plain
case-runner invocation:

```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/01-a1-lr-screen/run/runner.py
```

Every definitive training tranche, even one containing a single config, uses
its committed case runner:

```bash
python experiments/NN-phase-tranche/run/runner.py
```

The case runner contains the ordered config paths and delegates to
`paper_exp.runner.run_launch`. A reviewed bounded exception may also carry
tracked operational authorization. The A1 runner's dormant worker
authorization records a superseded `001`–`008` launch shape; it was inert for
the serial high-LR runs and did not cover configs `009`–`011`. Those three
configs also completed serially on one A40. No A1 config may be rerun.
The parent validates the complete tranche, requires every config to be a
direct sibling of the runner, and stops admission on the first failure while
draining work already admitted.
See
[`experiments/README.md`](experiments/README.md).

Before launch, report the estimated time to completion (ETC) for the first run
and the complete tranche, including the estimate basis and uncertainty. Monitor
the active run from its saved manifest and event stream without mutating it.

See [`docs/runbook.md`](docs/runbook.md) for preflight, monitoring, ETC, failure,
and retry requirements.

## Artifacts

Each chronological tranche is self-contained:

```text
experiments/NN-<phase>-<tranche>/
  run/                 # tracked runner and immutable configs
  raw/<config-id>/<run-id>/
  figs/                # generated figures and provenance
```

A completed pretraining run contains at least:

```text
config.yaml
manifest.json
metrics.json
predictions.jsonl
events.jsonl
checkpoints/final/    # when required by the config
```

The launch config and provenance are written before work begins. A completed
manifest is published only after required outputs are durable; an escaping
exception leaves a failed terminal manifest. See
[`experiments/README.md`](experiments/README.md).

Local datasets, token caches, payloads beneath each `raw/`, and generated
figures beneath each `figs/` are ignored by Git. Scaffold directories and all
`run/` contents remain tracked so every attempt has a reproducible recipe.
Paper figures must be fully regenerable from pinned saved artifacts.

The wheel contains the Python library and console entry point, not repository
configs or the experiment plan. Experiment execution is checkout-scoped: pass
an explicit config, and run scientific commands from the clean Git checkout
that owns it. CI installs the built wheel and exercises `paper-exp --help` so
the distribution boundary remains explicit.

## Plotting

Generate a supported diagnostic figure from one saved run with:

```bash
make plot KIND=run \
  RUN_DIR=experiments/NN-phase-tranche/raw/<config-id>/<run-id> \
  OUTPUT=experiments/NN-phase-tranche/figs/01-run-diagnostics.pdf PNG=1
```

Supported plot kinds are listed by `paper-exp plot --help`. The clean plotting
boundary, deterministic provenance sidecar, and publication requirements are documented in
[`docs/plotting.md`](docs/plotting.md).

## Repository Map

- `src/paper_exp/`: training, methods, diagnostics, lifecycle, runner, and
  plotting implementation.
- `experiments/`: chronological tranche scaffolds; tracked recipes live in
  `run/`, ignored attempts in `raw/`, and ignored generated figures in `figs/`.
- `docs/experiment_plan.md`: launch-status manifest and definitive authority
  when reviewed.
- `docs/experimental-design/`: focused protocol, stages, case catalog, outputs,
  decisions, reuse contract, and workboard.
- `docs/methods.md`: mathematical and optimization semantics.
- `docs/diagnostics.md`: metric definitions and interpretation limits.
- `docs/runbook.md`: launch, monitoring, ETC, and terminal verification.
- `docs/plotting.md`: artifact-to-figure contract.
- `docs/code_map.md`: code ownership and change routes.
- `tests/`: focused scientific and infrastructure contracts.

## Releasing and Citing

`TODO:` add the user-selected open-source license and final citation metadata
before public release. Dataset and model licenses must also be reviewed for the
exact resources named by the definitive experiment plan.
