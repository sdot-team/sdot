.PHONY: requirements all test help docs

PYTHON = python
PIP = pip

all: test

reqs:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	npm install

docs:
	npm run docs:build

test:
	SDOT_BUILD=1 PYTHONPATH=$(CURDIR)/src/python $(PYTHON) -m pytest tests/

help:
	@echo "  make all    : Alias for 'make test'"
	@echo "  make reqs   : Get all the requirements"
	@echo "  make docs   : Build the doc site"
	@echo "  make test   : Build and run all tests (pytest)"
