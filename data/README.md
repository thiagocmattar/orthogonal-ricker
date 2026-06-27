# Data

Local datasets and tokenized caches belong here, but the data itself is not committed.

Workflow:

```bash
make prepare-minipile
make calibrate-pythia-14m
```

The first command downloads MiniPile through Hugging Face datasets and writes an int32 token cache under `data/tokenized/<config_id>/`. The second command trains for a few calibration steps from that local token cache and records throughput metrics under `results/`.

When validation is enabled in the config, preparation also writes:

```text
data/tokenized/<config_id>/validation/
|-- metadata.json
`-- tokens.int32.bin
```

For paper runs, use local cached/tokenized data rather than streaming. The manifest and tokenized metadata should record dataset name, split, revision, tokenizer, block size, number of documents, and number of tokens.
