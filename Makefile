PYTHON ?= python
CONSTRAINTS ?= constraints/requirements-ci.txt

.PHONY: install test check smoke prepare-data calibrate plot

install:
	$(PYTHON) -m pip install -c "$(CONSTRAINTS)" -e ".[dev]"

test:
	$(PYTHON) scripts/run_tests.py

check:
	$(PYTHON) -m paper_exp.cli check --strict

smoke:
	$(PYTHON) -m paper_exp.cli smoke --config experiments/00-infrastructure-smoke/run/00-smoke.yaml

prepare-data:
	$(if $(strip $(CONFIG)),,$(error CONFIG is required. Usage: make prepare-data CONFIG=experiments/<tranche>/run/<file>.yaml))
	$(PYTHON) -m paper_exp.cli prepare-data --config "$(CONFIG)"

calibrate:
	$(if $(strip $(CONFIG)),,$(error CONFIG is required. Usage: make calibrate CONFIG=experiments/<tranche>/run/<file>.yaml))
	$(PYTHON) -m paper_exp.cli calibrate --config "$(CONFIG)"

plot:
	$(if $(strip $(KIND)),,$(error KIND is required. Usage: make plot KIND=run RUN_DIR=experiments/<tranche>/raw/<config>/<run> OUTPUT=experiments/<tranche>/figs/<file>.pdf))
	$(if $(strip $(RUN_DIR)),,$(error RUN_DIR is required. Usage: make plot KIND=run RUN_DIR=experiments/<tranche>/raw/<config>/<run> OUTPUT=experiments/<tranche>/figs/<file>.pdf))
	$(if $(strip $(OUTPUT)),,$(error OUTPUT is required. Usage: make plot KIND=run RUN_DIR=experiments/<tranche>/raw/<config>/<run> OUTPUT=experiments/<tranche>/figs/<file>.pdf))
	$(PYTHON) -m paper_exp.cli plot --kind "$(KIND)" --run-dir "$(RUN_DIR)" --output "$(OUTPUT)" $(if $(filter 1 true yes,$(PNG)),--png,)
