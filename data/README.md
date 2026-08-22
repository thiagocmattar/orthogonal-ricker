# Data

Downloaded data and tokenized caches live under this directory but are not
committed.

The definitive experiment plan will name the dataset, revision, tokenizer,
splits, preprocessing, partitions, and license requirements. Do not infer those
choices from historical caches or `configs/00-smoke.yaml`.

After a scientific config is authorized:

```bash
make prepare-data CONFIG=configs/01-example.yaml
# or
paper-exp prepare-data --config configs/01-example.yaml
```

A token cache must record enough metadata to reject incompatible reuse:

- dataset name, configuration, split, and revision;
- tokenizer name and revision;
- text column and document limits;
- block size, EOS policy, and token dtype;
- source-document and token counts;
- validation partition scheme, seed, indices hash, and excluded tail where
  applicable;
- creation provenance and content hashes required by the plan.

Paper runs use local, validated caches rather than an unpinned streaming data
source. Cache compatibility is determined from metadata, not directory name.

Before public release, review and document the dataset and tokenizer licenses
specified by the definitive plan. Never commit credentials, access tokens,
download URLs containing secrets, raw restricted data, or private cache paths.
