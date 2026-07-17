# Internal Reports

This folder is used for internal discussion reports. Keep report source and final PDFs trackable; keep run logs and LaTeX auxiliary files ignored.

## Report Index

| Report | Topic | PDF |
| --- | --- | --- |
| `01` | Initial status update | `01-2026-06-29-status-update/01-2026-06-29-status-update.pdf` |
| `02` | Method candidate review | `02-2026-07-05-status-update/02-2026-07-05-status-update.pdf` |
| `03` | ReLU site-scope comparison | `03-2026-07-09-relu-site-scope-comparison/03-2026-07-09-relu-site-scope-comparison.pdf` |
| `04` | Post-LayerNorm Three-ReLU comparison | `04-2026-07-11-post-layernorm-relu-ol1-comparison/04-2026-07-11-post-layernorm-relu-ol1-comparison.pdf` |
| `05` | Post-QKV ReLU placement comparison | `05-2026-07-17-post-qkv-relu-placement-comparison/05-2026-07-17-post-qkv-relu-placement-comparison.pdf` |

## Naming

Use one numbered folder per report:

```text
NN-YYYY-MM-DD-topic/
```

Use the same prefix for the source and PDF:

```text
NN-YYYY-MM-DD-topic/
├── NN-YYYY-MM-DD-topic.tex
└── NN-YYYY-MM-DD-topic.pdf
```

Examples:

```text
01-2026-06-29-status-update/
02-2026-07-05-method-candidate-review/
```

## Build

Build reports with MiKTeX from inside the report subfolder:

```bash
pdflatex -interaction=nonstopmode -halt-on-error NN-YYYY-MM-DD-topic.tex
pdflatex -interaction=nonstopmode -halt-on-error NN-YYYY-MM-DD-topic.tex
```

Run twice so references, links, and figure labels settle.

## Figures

Reports may include existing paper figures from `figures/`. From a numbered report subfolder, use paths like:

```tex
\includegraphics[width=0.94\textwidth]{../../figures/NN-figure-name.pdf}
```

Do not move canonical experiment figures into `report/`; keep them in `figures/` so the report references the same artifacts used by the harness.

## Style

- Keep report text in English.
- Keep filenames ASCII, lowercase, and hyphen-separated.
- Prefer concise status reports over broad prose.
- State whether a result is exploratory, one-seed, fixed-budget, or paper-ready.
- Keep source `.tex` and final `.pdf`; remove LaTeX auxiliary files after building.
