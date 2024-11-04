# PYTHON = python

pip_upload:
	# sudo apt install python3-venv
	# ${PYTHON} -m pip install --upgrade --break-system-packages twine
	# ${PYTHON} -m build
	# ${PYTHON} -m twine upload --repository pypi --verbose dist/*

python_link_install:
	cd src/python; flit install -s


.PHONY: all \
	pip_upload \
	python_link_install