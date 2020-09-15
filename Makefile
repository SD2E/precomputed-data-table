CONTAINER_IMAGE=$(shell bash scripts/container_image.sh)
PYTHON ?= "python3"
PYTEST_OPTS ?= "-s -vvv"
PYTEST_DIR ?= "tests"
ABACO_DEPLOY_OPTS ?= ""
SCRIPT_DIR ?= "scripts"
PREF_SHELL ?= "bash"
ACTOR_ID ?=
NOCLEANUP ?= 0

GITREF=$(shell git rev-parse --short HEAD)

export PDT_OMICS_TOOLS_INIFILE := app/precomputed-data-table-omics-tools/app.ini
export PDT_OMICS_TOOLS_DIR := app/precomputed-data-table-omics-tools

export PDT_WASSERSTEIN_INIFILE := app/precomputed-data-table-wasserstein/app.ini
export PDT_WASSERSTEIN_DIR := app/precomputed-data-table-wasserstein

export PDT_GROWTH_ANALYSIS_INIFILE := app/precomputed-data-table-growth-analysis/app.ini
export PDT_GROWTH_ANALYSIS_DIR := app/precomputed-data-table-growth-analysis

export PDT_FCS_SIGNAL_PREDICTION_INIFILE := app/precomputed-data-table-fcs-signal-prediction/app.ini
export PDT_FCS_SIGNAL_PREDICTION_DIR := app/precomputed-data-table-fcs-signal-prediction

export PDT_LIVE_DEAD_PREDICTION_INIFILE := app/precomputed-data-table-live-dead-prediction/app.ini
export PDT_LIVE_DEAD_PREDICTION_DIR := app/precomputed-data-table-live-dead-prediction

.PHONY: tests app-container tests-local tests-reactor tests-deployed
.SILENT: tests app-container tests-local tests-reactor tests-deployed

all: reactor-image app-image

reactor-image:
	python record_product_info.py > version.txt; \
	abaco deploy -R -F Dockerfile -k -B reactor.rc -R -t $(GITREF) $(ABACO_DEPLOY_OPTS)

# Apparently apps-build-container ignores the -f flag, thus we have to move the two Dockerfiles around below

omics-tools-image:
	cd $(PDT_OMICS_TOOLS_DIR); \
	find . -name '*.pyc' -delete ; \
	apps-build-container -V ; \
	echo "The app container is done building."
	echo "  make shell - explore the container interactively"
	echo "  make tests-pytest - run Python tests in the container"
	echo "  make tests-local - execute container (and wrapper) under emulation"

wasserstein-image:
	cd $(PDT_WASSERSTEIN_DIR); \
	cp -r ../common src; \
	find . -name '*.pyc' -delete ; \
	apps-build-container -V ; \
	rm -r src/common; \
	echo "The app container is done building."
	echo "  make shell - explore the container interactively"
	echo "  make tests-pytest - run Python tests in the container"
	echo "  make tests-local - execute container (and wrapper) under emulation"
	
growth-analysis-image:
	cd $(PDT_GROWTH_ANALYSIS_DIR); \
	cp -r ../common src; \
	find . -name '*.pyc' -delete ; \
	apps-build-container -V ; \
	rm -r src/common; \
	echo "The app container is done building."
	echo "  make shell - explore the container interactively"
	echo "  make tests-pytest - run Python tests in the container"
	echo "  make tests-local - execute container (and wrapper) under emulation"

fcs-signal-prediction-image:
	cd $(PDT_FCS_SIGNAL_PREDICTION_DIR); \
	cp -r ../common src; \
	find . -name '*.pyc' -delete ; \
	apps-build-container -V ; \
	rm -r src/common; \
	echo "The app container is done building."
	echo "  make shell - explore the container interactively"
	echo "  make tests-pytest - run Python tests in the container"
	echo "  make tests-local - execute container (and wrapper) under emulation"	

live-dead-prediction-image:
	cd $(PDT_LIVE_DEAD_PREDICTION_DIR); \
	cp -r ../common src; \
	find . -name '*.pyc' -delete ; \
	apps-build-container -V ; \
	rm -r src/common; \
	echo "The app container is done building."
	echo "  make shell - explore the container interactively"
	echo "  make tests-pytest - run Python tests in the container"
	echo "  make tests-local - execute container (and wrapper) under emulation"	

shell:
	bash $(SCRIPT_DIR)/run_container_process.sh bash

tests-local: 
	bash scripts/run_container_message.sh tests/data/msg-test-samples.json $(ACTOR_ID)

tests-pytest:
#	bash $(SCRIPT_DIR)/run_container_process.sh $(PYTHON) -m "pytest" $(PYTEST_DIR) $(PYTEST_OPTS)
	echo "not implemented"

tests-deployed:
	echo "not implemented"

clean: clean-reactor-image clean-tests clean-app-image

clean-reactor-image:
	docker rmi -f $(CONTAINER_IMAGE)

clean-wasserstein-image:
	bash scripts/remove_images.sh $(PDT_WASSERSTEIN_INIFILE)

clean-growth-analysis-image:
	bash scripts/remove_images.sh $(PDT_GROWTH_ANALYSIS_INIFILE)

clean-fcs-signal-prediction-image:
	bash scripts/remove_images.sh $(PDT_FCS_SIGNAL_PREDICTION_INIFILE)

clean-omics-tools-image:
	bash scripts/remove_images.sh $(PDT_OMICS_TOOLS_INIFILE)

clean-tests:
	rm -rf .hypothesis .pytest_cache __pycache__ */__pycache__ tmp.* *junit.xml
	
deploy:
	abaco deploy -t $(GITREF) $(ABACO_DEPLOY_OPTS) -U $(ACTOR_ID)

deploy-wasserstein:
	cd $(PDT_WASSERSTEIN_DIR); \
	cp -r ../common src; \
	apps-deploy; \
	rm -r src/common

deploy-growth-analysis:
	cd $(PDT_GROWTH_ANALYSIS_DIR); \
	cp -r ../common src; \
	apps-deploy; \
	rm -r src/common

deploy-fcs-signal-prediction:
	cd $(PDT_FCS_SIGNAL_PREDICTION_DIR); \
	cp -r ../common src; \
	apps-deploy; \
	rm -r src/common

deploy-live-dead-prediction:
	cd $(PDT_LIVE_DEAD_PREDICTION_DIR); \
	cp -r ../common src; \
	apps-deploy; \
	rm -r src/common
	apps-deploy

deploy-omics-tools:
	cd $(PDT_OMICS_TOOLS_DIR); \
	apps-deploy

postdeploy:
	bash tests/run_after_deploy.sh