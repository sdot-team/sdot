.PHONY: venv setup build all clean test help docs

VENV = $(CURDIR)/.venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
MESON = $(VENV)/bin/meson

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
	@$(PIP) install torch jax jaxlib nanobind meson ninja numpy pytest timing mkdocs-material mkdocstrings[python] pyyaml

docs:
	$(VENV)/bin/mkdocs serve --config-file docs/mkdocs.yml


test: build
	@echo  "\n>>> Running C++ Tests (meson test) ------------------------------------------"
	@$(MESON) test -C build --print-errorlogs
	@echo "\n>>> Running Python Tests (pytest) --------------------------------------------"
	@PYTHONPATH=$(CURDIR)/python/pytorch:$(CURDIR)/python/jax:$(PYTHONPATH) $(PYTHON) -m pytest -s -q --tb=short tests/

ct_reco: build
	@PYTHONPATH=$(CURDIR)/python/pytorch:$(CURDIR)/python/jax $(PYTHON) examples/ct_reconstruction/ct_reconstruction.py

clean:
	@rm -rf build
	@rm -f python/pytorch/*.so python/jax/*.so
	@rm -rf .cache/clangd
	@rm -f compile_commands.json

help:
	@echo "Meson Makefile:"
	@echo "  make venv   : Create venv and install deps (including meson)"
	@echo "  make setup  : Setup Meson build directory"
	@echo "  make build  : Build library and extensions via Meson"
	@echo "  make all    : Alias for 'make build'"
	@echo "  make test   : Build and run all tests (meson test + pytest)"
	@echo "  make clean  : Cleanup build artifacts"
