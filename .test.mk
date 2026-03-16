all:
	make build
	# PYTHONPATH=$(CURDIR)/src/python:$(CURDIR)/build/src/python .venv/bin/pytest tests/test_pytorch.py
	PYTHONPATH=$(CURDIR)/src/python:$(CURDIR)/build/src/python .venv/bin/python docs/examples/ct_reconstruction/ct_reconstruction.py
