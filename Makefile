PYTHON ?= python

.PHONY: install test smoke baseline plots

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest --basetemp=.pytest_tmp

smoke:
	$(PYTHON) -m paper_exp.cli smoke --config configs/01-baseline.yaml

baseline:
	$(PYTHON) -m paper_exp.cli baseline --config configs/01-baseline.yaml

plots:
	$(PYTHON) -m paper_exp.cli plots --results results --figures figures --png
