.PHONY: venv all clean test test-cpu test-pytorch test-jax help

VENV = $(CURDIR)/.venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip3

all:
	echo $(VENV)
	mkdir -p build
	cd build && cmake .. && cmake --build .
	# Link extensions so they are importable
	ln -sf $(CURDIR)/build/sdot_pytorch_cpp.cpython* $(CURDIR)/python/pytorch/
	ln -sf $(CURDIR)/build/sdot_jax_cpp.cpython* $(CURDIR)/python/jax/

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install torch jax jaxlib nanobind cmake ninja numpy

test: all
	cd build && ./test_w2_cpu
	source $(VENV)/bin/activate && python3 tests/test_w2_pytorch.py
	source $(VENV)/bin/activate && python3 tests/test_w2_jax.py

clean:
	rm -rf build
	rm -f python/pytorch/*.so python/jax/*.so
	rm -f compile_commands.json

help:
	@echo "Modern CMake Makefile:"
	@echo "  make venv   : Create venv and install deps"
	@echo "  make all    : Build library and extensions via CMake"
	@echo "  make test   : Build and run all tests"
	@echo "  make clean  : Cleanup build artifacts"
