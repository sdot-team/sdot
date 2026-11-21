# The loom website lives under docs/ ( VitePress ), with its own package.json, so the project root
# stays a Python package root. `npm install` is idempotent, so there is no separate step to remember.

.PHONY: help viewer site site-build site-deploy api-docs anim version bump_version kernels wheels wheels_all check_wheels publish

help:
	@echo "  make site          the website, live reloading   ( docs/, VitePress )"
	@echo "  make site-build    the website into docs/.vitepress/dist"
	@echo "  make api-docs      the API reference ( docs/api/*.md ) from the docstrings"
	@echo "  make site-deploy   site-build, then push it to the gh-pages branch ( https://sdot-team.github.io/sdot/ )"
	@echo "  make version       print the version ( and the loom it requires )"
	@echo "  make bump_version  write today's next version ( TO=x.y.z.n to choose it )"
	@echo "  make kernels       the precompiled kernels of this machine, every framework it has"
	@echo "  make wheels        the sdot wheel of this machine into dist/: kernels, wheel, check"
	@echo "  make wheels_all    make wheels for every platform of pyproject.toml, in loom's build environments ( PLATFORMS=... for some )"
	@echo "  make check_wheels  install this version's wheel in a clean venv and run the checks"
	@echo "  make publish       upload this version's wheels of dist/ ( TWINE_PASSWORD = your PyPI token )"
	@echo "  make viewer        the viewer's bundle into docs/public/viewer, listed as published ( before site-deploy, at a release )"
	@echo "  make anim          regenerate the tutorials' animations ( docs/public/anim/*.gif ) -- needs jax + matplotlib"

site: docs/node_modules
	cd docs && npm run dev

site-build: docs/node_modules
	cd docs && npm run build

# the viewer of scene stores, pinned by its hash ( sdot ADR 0001 ): served by the site, so that stores only name it.
# Every published bundle stays in docs/public/viewer ( committed ): stores written by older versions still find theirs.
viewer:
	$(PYTHON) -m sdot.viz.sviz bundle docs/public/viewer --publish

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


# ── releasing ──
# reminders: the work is in loom ( `python -m loom.release`, docs/guide/releasing.md in loom ), configured by [tool.loom.release]
PYTHON ?= python

version:
	$(PYTHON) -m loom.release show_version

bump_version:
	$(PYTHON) -m loom.release bump_version $(if $(TO),--to $(TO))

kernels:
	$(PYTHON) -m loom.release make_kernels

wheels:
	$(PYTHON) -m loom.release make_wheels

# every platform of [tool.loom.release] in loom's build environments ( through errand ), the wheels brought back into dist/.
# Some of them only: PLATFORMS=linux-x86_64,windows-x86_64
PLATFORMS ?= all

wheels_all:
	$(PYTHON) -m loom.release dispatch_wheels $(PLATFORMS)

check_wheels:
	$(PYTHON) -m loom.release check_wheels

publish:
	$(PYTHON) -m loom.release publish_wheels
