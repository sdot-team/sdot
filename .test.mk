PYDIRS = PYTHONPATH=$(CURDIR)/src/python:$(CURDIR)/build/src/python
PYTHON = .venv/bin/python
PYTEST = .venv/bin/pytest

# 	make build
# PYTHONPATH=$(CURDIR)/src/python:$(CURDIR)/build/src/python .venv/bin/python tests/test_vanilla.py
# PYTHONPATH=$(CURDIR)/src/python:$(CURDIR)/build/src/python .venv/bin/python docs/examples/ct_reconstruction/ct_reconstruction.py
# ${PYDIRS} ${PYTEST} tests/test_pytorch.py
# 	${PYDIRS} ${PYTHON} docs/examples/ct_reconstruction/ct_reconstruction.py
# 	micromamba run -n vfs vfs_build run tests/test_SimpleSquareMatrix.cpp
# 	micromamba run -n vfs vfs_build run tests/test_PowerDiagram.cpp
# 	${PYDIRS} ${PYTHON} tests/test_dask.py
# 	${PYDIRS} ${PYTHON} tests/python/test_OtPlan.py

all:
	make test

ct:
	make build
	${PYDIRS} ${PYTHON} docs/examples/ct_reconstruction/ct_reconstruction.py

vg:
	rsync -a --exclude='build' --exclude='*.so' --exclude='.venv' . lmo://home/leclerc/sdot_with_interfaces
	ssh lmo "make -C ~/sdot_with_interfaces -f .test.mk vg_loc"

vg_loc:
	make build
# 	${PYDIRS} valgrind ${PYTHON} docs/examples/ct_reconstruction/ct_reconstruction.py
	${PYDIRS} valgrind ${PYTHON} tests/test_dask.py
