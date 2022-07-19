.PHONY: clean clean-build clean-pyc clean-test dist help install lint lint/flake8 lint/black
.DEFAULT_GOAL := help
SHELL := /bin/bash
define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint/flake8: ## check style with flake8
	flake8 ceruleanml tests

lint/black: ## check style with black
	black --check ceruleanml tests

lint: lint/flake8 lint/black ## check style

# test: ## run tests quickly with the default Python
# 	pytest

# test-all: ## run tests on every Python version with tox
# 	tox

# dist: clean ## builds source and wheel package
# 	python setup.py sdist
# 	python setup.py bdist_wheel
# 	ls -l dist

install: clean ## install the package to the fastai2 env on the terraform gcp vm
	mamba env update --name fastai2 --file environment.yml --prune
	/root/miniconda3/envs/fastai2/bin/pip install -e . # install local ceruleanml package after deps installed with conda

setup-icevision-env: ## install the package to the icevision env on the terraform gcp vm
	# cd ..
	# git clone https://github.com/airctic/icevision --depth 1
	# cd work
	# requires activating env outside of makefile after this step
	rm -rf .ice-env
	python -m venv .ice-env
	chmod +x ./.ice-env/bin/activate
	source ./.ice-env/bin/activate # outside of a script, source needs to be used when activating

install-icevision-deps:
	./.ice-env/bin/pip install -e . # install local ceruleanml package after deps installed with conda
	./.ice-env/bin/pip install "git+https://github.com/waspinator/pycococreator.git@0.2.0"
	./.ice-env/bin/pip install jupyterlab
	python -m ipykernel install --user --name=icevision
	# ./.ice-env/bin/pip install -e .[dev] # run this in terminal after cd to icevision

install-pl: ## install the package to the pl env on the terraform gcp vm
	rm -rf .pl
	python -m venv .pl
	chmod +x ./.pl/bin/activate
	./.pl/bin/activate # outside of a script, source needs to be used when activiating
	./.pl/bin/pip install pytorch-lightning lightning-flash torch torchvision torchaudio icevision --extra-index-url https://download.pytorch.org/whl/cu113
	./.pl/bin/pip install -e . # install local ceruleanml package after deps installed with conda
	./.pl/bin/pip install "git+https://github.com/waspinator/pycococreator.git@0.2.0"
	./.pl/bin/pip install jupyterlab
	./.pl/bin/python -m ipykernel install --user --name=pl
