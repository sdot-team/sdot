# The loom website lives under docs/ ( VitePress ), with its own package.json, so the project root
# stays a Python package root. `npm install` is idempotent, so there is no separate step to remember.

.PHONY: help site site-build site-deploy api-docs anim local_kernels wheels local_wheels publish

help:
	@echo "  make site          the website, live reloading   ( docs/, VitePress )"
	@echo "  make site-build    the website into docs/.vitepress/dist"
	@echo "  make api-docs      the API reference ( docs/api/*.md ) from the docstrings"
	@echo "  make site-deploy   site-build, then push it to the gh-pages branch ( https://sdot-team.github.io/sdot/ )"
	@echo "  make local_kernels measure the choices and build the kernels ( src/sdot/precompiled-kernels/ ) for this machine, per framework"
	@echo "  make wheels        every wheel ( mac, manylinux, windows ) under dist/, from the environments of errand-envs.py"
	@echo "  make local_wheels  local_kernels, then the loom + sdot wheels of this machine under dist/, validated"
	@echo "  make publish       upload dist/ to PyPI ( TWINE_PASSWORD = your PyPI token )"
	@echo "  make anim          regenerate the tutorials' animations ( docs/public/anim/*.gif ) -- needs jax + matplotlib"

site: docs/node_modules
	cd docs && npm run dev

site-build: docs/node_modules
	cd docs && npm run build

# the API reference ( docs/api/*.md ), from the docstrings -- `tests/test_api_docs.py` says when it is stale
api-docs:
	$(PYTHON) scripts/api_docs.py

# The built site becomes the only commit of the gh-pages branch ( force-pushed: no history to keep ).
# .nojekyll, or GitHub Pages drops the _-prefixed files of VitePress. Once, on GitHub:
# Settings > Pages > Source = "Deploy from a branch", branch gh-pages, folder / ( root ).
SITE_REMOTE ?= $(shell git remote get-url origin)
SITE_BRANCH ?= gh-pages

site-deploy: site-build
	cd docs/.vitepress/dist && touch .nojekyll && rm -rf .git \
	  && git init -q -b $(SITE_BRANCH) \
	  && git add -A \
	  && git commit -q -m "site: $$(cd ../../.. && git rev-parse --short HEAD)" \
	  && git push -f $(SITE_REMOTE) $(SITE_BRANCH) \
	  && rm -rf .git

docs/node_modules: docs/package.json
	cd docs && npm install
	@touch $@


# ── the wheels ──

PYTHON     ?= python
FRAMEWORKS ?= jax numpy torch

# make.exe and a Python are all Windows has: the loops and the paths are in scripts/wheels.py
local_kernels:
	$(PYTHON) scripts/wheels.py --kernels $(FRAMEWORKS) --no-build

# each environment builds its own wheels ( every framework ) in its copy of the tree, `--pull dist` brings them back
WHEEL_ENVS ?= mac_wheels,lin_wheels,win_wheels

wheels:
	errand shell --env $(WHEEL_ENVS) --pull dist -- make local_wheels

local_wheels:
	$(PYTHON) scripts/wheels.py --kernels $(FRAMEWORKS)

publish:
	$(PYTHON) scripts/wheels.py --no-build --publish https://upload.pypi.org/legacy/
