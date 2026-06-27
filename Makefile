PYTHON ?= python

.PHONY: install test smoke baseline prepare-minipile calibrate-pythia-14m pretrain-pythia-14m-full-10min plots

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest --basetemp=.pytest_tmp

smoke:
	$(PYTHON) -m paper_exp.cli smoke --config configs/01-pythia-14m-minipile-smoke.yaml

baseline:
	$(PYTHON) -m paper_exp.cli baseline --config configs/02-pythia-14m-minipile-baseline.yaml

prepare-minipile:
	$(PYTHON) -m paper_exp.cli prepare-data --config configs/01-pythia-14m-minipile-smoke.yaml

calibrate-pythia-14m:
	$(PYTHON) -m paper_exp.cli calibrate --config configs/01-pythia-14m-minipile-smoke.yaml

pretrain-pythia-14m-full-10min:
	$(PYTHON) -m paper_exp.cli prepare-data --config configs/03-pythia-14m-minipile-random-full-10min.yaml
	$(PYTHON) -m paper_exp.cli pretrain --config configs/03-pythia-14m-minipile-random-full-10min.yaml

plots:
	$(PYTHON) -m paper_exp.cli plots --results results --figures figures --png
