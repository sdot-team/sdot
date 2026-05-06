.PHONY: requirements all test help docs release wheel wheel-all-linux

PYTHON = python
PIP = pip

all: test

reqs:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	npm install

docs:
	npm run docs:build

docs_dev:
	npm run docs:dev

# Bump la version (YYYY.MM.DD.N), commit, tag → déclenche GitHub Actions
release:
	${PYTHON} scripts/bump_version.py
	@echo "Lancez maintenant : git push && git push origin $$(git describe --tags --abbrev=0)"

make_common_dylibs:
	SDOT_XMAKE_MODE=release SDOT_FORCE_BUILD=1 PYTHONPATH=$(CURDIR)/src/python ${PYTHON} scripts/make_common_dylibs.py

# Wheel pour le Python courant seulement — itération rapide, pas besoin de python.org
wheel: make_common_dylibs
	pip install build hatchling
	python -m build --wheel --outdir wheelhouse/

# Tous les Pythons en local (macOS) via uv.
# cibuildwheel requiert des "framework builds" python.org incompatibles avec uv/homebrew.
# On boucle directement sur les Pythons uv : uv python install 3.10 3.11 3.12 3.13
wheel-all-macos:
	@for version in 3.10 3.11 3.12 3.13; do \
		echo ""; echo "=== Python $$version ==="; \
		rm -f src/python/sdot/generated_files/*.so \
		      src/python/sdot/generated_files/*.dylib; \
		SDOT_XMAKE_MODE=release SDOT_FORCE_BUILD=1 \
		PYTHONPATH=$(CURDIR)/src/python \
		    uv run --python $$version \
		    --with jax,jaxlib,numpy,nanobind,xmake,meson,ninja \
		    python scripts/make_common_dylibs.py; \
		uv run --python $$version --with build \
		    python -m build --wheel --outdir wheelhouse/; \
	done

# Wheel Linux via Docker (nécessite Docker lancé)
wheel-all-linux:
	pip install cibuildwheel
	cibuildwheel --platform linux

publish-docs:
	npm run docs:deploy

docker-build:
	docker build -t sdot-devel .

docker-build-x86:
	docker build --platform linux/amd64 -t sdot-devel-x86 .

docker-run:
	docker run -it --rm -v $(CURDIR):/workspace sdot-devel

test:
	SDOT_FORCE_BUILD=1 PYTHONPATH=$(CURDIR)/src/python $(PYTHON) -m pytest -v tests/python/

help:
	@echo "  make release      : Bump version, commit, tag → déclenche la CI de publication"
	@echo "  make all          : Alias for 'make test'"
	@echo "  make reqs         : Get all the requirements"
	@echo "  make docs         : Build the doc site"
	@echo "  make publish-docs : Build and publish the doc site to GitHub Pages"
	@echo "  make wheel        : Build wheel pour le Python courant (itération rapide)"
	@echo "  make wheel-all-macos    : Build tous les Pythons macOS (nécessite python.org ou uv)"
	@echo "  make wheel-all-linux  : Build wheels Linux via Docker (cibuildwheel)"
	@echo "  make docker-build : Build the devel Docker image (native)"
	@echo "  make docker-run   : Run an interactive shell in the Docker container"
	@echo "  make test         : Build and run all tests (pytest)"
