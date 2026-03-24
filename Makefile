.PHONY: venv setup build all clean test help docs

VENV = $(CURDIR)/.venv

PYTHON = $(VENV)/bin/python
MESON = $(VENV)/bin/meson
PIP = $(VENV)/bin/pip

setup:
	@if [ ! -f build/build.ninja ]; then \
		$(MESON) setup build; \
	fi

build: setup
	@$(MESON) compile -C build

all: build

venv:
	@python3 -m venv $(VENV)
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

docs:
	@$(VENV)/bin/mkdocs serve --config-file docs/mkdocs.yml


test: build
	# 	@echo  "\n>>> Running C++ Tests (meson test) ------------------------------------------"
	# 	@$(MESON) test -C build --print-errorlogs
	# 	@echo "\n>>> Running Python Tests (pytest) --------------------------------------------"
	@PYTHONPATH=$(CURDIR)/src/python:$(CURDIR)/build/src/python $(PYTHON) -m pytest -s -q --tb=short tests/

clean:
	@rm -rf build
	@rm -rf .cache/clangd

help:
	@echo "Meson Makefile:"
	@echo "  make venv   : Create venv and install deps (including meson). Use . '.venv/bin/activate' to activate it"
	@echo "  make setup  : Setup Meson build directory"
	@echo "  make build  : Build library and extensions via Meson"
	@echo "  make all    : Alias for 'make build'"
	@echo "  make test   : Build and run all tests (meson test + pytest)"
	@echo "  make clean  : Cleanup build artifacts"
