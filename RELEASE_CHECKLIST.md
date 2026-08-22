# Release Checklist

Release status: **blocked**.

The repository must not be described as open source or published as the paper's
definitive experiment package until both blocking decisions are resolved:

- [ ] The repository owner selects and adds a `LICENSE`.
- [ ] The forthcoming definitive experiment plan is reviewed and committed.

After those decisions:

- [ ] Update `CITATION.cff` with the final paper title, author list, and DOI or
      archival identifier when available.
- [ ] Confirm every public experiment has a pinned config and documented data,
      model, tokenizer, and environment requirements.
- [ ] Confirm `constraints/requirements-ci.txt` matches the release environment
      and passes both CI matrix jobs; update it only as a reviewed snapshot.
- [ ] Run `make test` and `make check` from a clean clone and require CI to pass.
- [ ] Build the wheel and source distribution with `python -m build`, inspect
      their contents, install the wheel, and exercise the `paper-exp` entry point.
- [ ] Run one documented smoke experiment and verify its terminal artifact
      envelope.
- [ ] Decide which compact results and figures, if any, are release artifacts;
      do not publish old local outputs by accident.
- [ ] Review tracked files for credentials, private data, usernames, absolute
      paths, and machine-specific launch commands.
- [ ] Review third-party dataset, model, and dependency terms and acknowledgments.
- [ ] Tag the release only after the repository is clean and the tagged commit
      matches the cited archival snapshot.
