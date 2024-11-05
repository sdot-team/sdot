# PYTHON = python

deps:
	mkdir -p ext
	test -e ext/eigen || ( cd ext && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2 && tar xjf eigen-3.4.0.tar.bz2 && mv eigen-3.4.0 eigen && rm eigen-3.4.0.tar.bz2 )
	test -e ext/Catch2 || ( cd ext && wget https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz && tar xzf v3.7.1.tar.gz && mv Catch2-3.7.1 Catch2 && rm v3.7.1.tar.gz )

pip_upload:
	# sudo apt install python3-venv
	# ${PYTHON} -m pip install --upgrade --break-system-packages twine
	# ${PYTHON} -m build
	# ${PYTHON} -m twine upload --repository pypi --verbose dist/*

python_link_install:
	cd src/python; flit install -s

.PHONY: all \
	deps \
	pip_upload \
	python_link_install