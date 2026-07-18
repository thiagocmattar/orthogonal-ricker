# Experiments

Use this directory for short notes that do not belong in a config or the paper docs.

Preferred workflow:

1. Add or copy a numbered config under `configs/`, such as `02-ablation-name.yaml`.
2. Commit the reviewed config before launch.
3. Run the command.
4. Record the result path in `docs/experiment_log.md`.
5. Map paper artifacts in `docs/paper_map.md`.

The Pythia scaling campaign is large enough to require a durable tracker. Use
[`docs/experimental-design/`](../docs/experimental-design/README.md) for its
design matrix, config/run registries, statuses, promotion rules, and handoff
runbook. Do not create a second tracker for that campaign.
