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

export INIFILE := app/precomputed-data-table-app/app.ini
export APPDIR := app/precomputed-data-table-app

export PDT_OMICS_TOOLS_INIFILE := app/precomputed-data-table-omics-tools/app.ini
export PDT_OMICS_TOOLS_DIR := app/precomputed-data-table-omics-tools

export PDT_WASSERSTEIN_INIFILE := app/precomputed-data-table-wasserstein/app.ini
export PDT_WASSERSTEIN_DIR := app/precomputed-data-table-wasserstein

export PDT_GROWTH_ANALYSIS_INIFILE := app/precomputed-data-table-growth-analysis/app.ini
export PDT_GROWTH_ANALYSIS_DIR := app/precomputed-data-table-growth-analysis

export PDT_FCS_SIGNAL_PREDICTION_INIFILE := app/precomputed-data-table-fcs-signal-prediction/app.ini
export PDT_FCS_SIGNAL_PREDICTION_DIR := app/precomputed-data-table-fcs-signal-prediction

.PHONY: tests app-container tests-local tests-reactor tests-deployed
.SILENT: tests app-container tests-local tests-reactor tests-deployed

all: reactor-image app-image

reactor-image:
	abaco deploy -R -F Dockerfile -k -B reactor.rc -R -t $(GITREF) $(ABACO_DEPLOY_OPTS)

# Apparently apps-build-container ignores the -f flag, thus we have to move the two Dockerfiles around below
app-image: 
	cd $(APPDIR); \
	find . -name '*.pyc' -delete ; \
	apps-build-container -V ; \
	echo "The app container is done building."
	echo "  make shell - explore the container interactively"
	echo "  make tests-pytest - run Python tests in the container"
	echo "  make tests-local - execute container (and wrapper) under emulation"

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
	find . -name '*.pyc' -delete ; \
	apps-build-container -V ; \
	echo "The app container is done building."
	echo "  make shell - explore the container interactively"
	echo "  make tests-pytest - run Python tests in the container"
	echo "  make tests-local - execute container (and wrapper) under emulation"
	
growth-analysis-image:
	cd $(PDT_GROWTH_ANALYSIS_DIR); \
	find . -name '*.pyc' -delete ; \
	apps-build-container -V ; \
	echo "The app container is done building."
	echo "  make shell - explore the container interactively"
	echo "  make tests-pytest - run Python tests in the container"
	echo "  make tests-local - execute container (and wrapper) under emulation"

fcs-signal-prediction-image:
	cd $(PDT_FCS_SIGNAL_PREDICTION_DIR); \
	find . -name '*.pyc' -delete ; \
	apps-build-container -V ; \
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

clean-app-image:
	bash scripts/remove_images.sh $(INIFILE)

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
	
deploy-app:
	cd $(APPDIR); \
	apps-deploy

deploy-wasserstein:
	cd $(PDT_WASSERSTEIN_DIR); \
	apps-deploy

deploy-growth-analysis:
	cd $(PDT_GROWTH_ANALYSIS_DIR); \
	apps-deploy

<<<<<<< HEAD
deploy-fcs-signal-prediction:
	cd $(PDT_FCS_SIGNAL_PREDICTION_DIR); \
	apps-deploy
		
=======
>>>>>>> 8884d69c9a050931b45dda37a1af609d1faf70d3
deploy-omics-tools:
	cd $(PDT_OMICS_TOOLS_DIR); \
	apps-deploy

postdeploy:
	bash tests/run_after_deploy.sh