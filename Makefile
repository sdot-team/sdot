.PHONY: venv all clean test help

VENV = $(CURDIR)/.venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

all:
	@mkdir -p build
	@if [ ! -f build/rules.ninja ]; then \
		cd build && cmake .. -GNinja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON; \
	fi
	@cmake --build build --parallel
	@ln -sf $(CURDIR)/build/sdot_pytorch_cpp.cpython* $(CURDIR)/python/pytorch/
	@ln -sf $(CURDIR)/build/sdot_jax_cpp.cpython* $(CURDIR)/python/jax/

venv:
	@python3 -m venv $(VENV)
	@$(PIP) install --upgrade pip
	@$(PIP) install torch jax jaxlib nanobind cmake ninja numpy pytest

test: all
	@echo  "\n>>> Running C++ Tests (ctest) -----------------------------------------------"
	@cd build && ctest --output-on-failure --test-dir .
	@echo "\n>>> Running Python Tests (pytest) --------------------------------------------"
	@PYTHONPATH=$(CURDIR)/build:$(CURDIR)/python/pytorch:$(CURDIR)/python/jax:$(PYTHONPATH) $(PYTHON) -m pytest -q --tb=short tests/

clean:
	@rm -rf build
	@rm -f python/pytorch/*.so python/jax/*.so
	@rm -rf .cache/clangd
	@rm -f compile_commands.json

help:
	@echo "Modern CMake Makefile:"
	@echo "  make venv   : Create venv and install deps"
	@echo "  make all    : Build library and extensions via CMake"
	@echo "  make test   : Build and run all tests (ctest + pytest)"
	@echo "  make clean  : Cleanup build artifacts"
