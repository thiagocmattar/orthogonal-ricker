PYTHON ?= python
CONSTRAINTS ?= constraints/requirements-ci.txt

.PHONY: install test check smoke prepare-data calibrate pretrain run-configs run-status plot

install:
	$(PYTHON) -m pip install -c "$(CONSTRAINTS)" -e ".[dev]"

test:
	$(PYTHON) scripts/run_tests.py

check:
	$(PYTHON) -m paper_exp.cli check --strict

smoke:
	$(PYTHON) -m paper_exp.cli smoke --config configs/00-smoke.yaml

prepare-data:
	$(if $(strip $(CONFIG)),,$(error CONFIG is required. Usage: make prepare-data CONFIG=configs/<file>.yaml))
	$(PYTHON) -m paper_exp.cli prepare-data --config "$(CONFIG)"

calibrate:
	$(if $(strip $(CONFIG)),,$(error CONFIG is required. Usage: make calibrate CONFIG=configs/<file>.yaml))
	$(PYTHON) -m paper_exp.cli calibrate --config "$(CONFIG)"

pretrain:
	$(if $(strip $(CONFIG)),,$(error CONFIG is required. Usage: make pretrain CONFIG=configs/<file>.yaml))
	$(PYTHON) -m paper_exp.cli pretrain --config "$(CONFIG)"

run-configs:
	$(if $(strip $(CONFIGS)),,$(error CONFIGS is required. Usage: make run-configs CONFIGS="configs/01.yaml configs/02.yaml"))
	$(PYTHON) -m paper_exp.cli run-configs $(foreach config,$(CONFIGS),--config "$(config)")

run-status:
	$(if $(strip $(STATE)),,$(error STATE is required. Usage: make run-status STATE=run-logs/runner-state.json))
	$(PYTHON) -m paper_exp.cli run-status --state "$(STATE)"

plot:
	$(if $(strip $(KIND)),,$(error KIND is required. Usage: make plot KIND=run RUN_DIR=results/<run> OUTPUT=figures/<file>.pdf))
	$(if $(strip $(RUN_DIR)),,$(error RUN_DIR is required. Usage: make plot KIND=run RUN_DIR=results/<run> OUTPUT=figures/<file>.pdf))
	$(if $(strip $(OUTPUT)),,$(error OUTPUT is required. Usage: make plot KIND=run RUN_DIR=results/<run> OUTPUT=figures/<file>.pdf))
	$(PYTHON) -m paper_exp.cli plot --kind "$(KIND)" --run-dir "$(RUN_DIR)" --output "$(OUTPUT)" $(if $(filter 1 true yes,$(PNG)),--png,)
