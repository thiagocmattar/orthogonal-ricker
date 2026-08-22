# Launch Runners

Each reviewed launch tranche gets one committed
`NN-<phase>-<tranche>.py` runner and a same-named config folder. Runners are not
created until the definitive plan fixes the tranche. See
[`docs/runbook.md`](../docs/runbook.md) for ordering, ETC, and launch rules.

A case runner contains only its ordered configs and a call to the shared
parent:

```python
from paper_exp.runner import run_launch

CONFIGS = (
    "configs/NN-phase-tranche/CCC-first-case.yaml",
    "configs/NN-phase-tranche/DDD-second-case.yaml",
)

if __name__ == "__main__":
    run_launch(__file__, CONFIGS)
```

The parent requires the matching folder, validates that the tuple lists every
config exactly once, holds one launch lock, and executes the tranche serially
in increasing config order. It stops on the first failure.
