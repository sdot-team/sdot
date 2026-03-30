"""sdot.bindings — compiled extension modules.

The compiled .so / .dylib files live in the user cache directory rather than
in the source tree.  This __init__.py adds that directory to __path__ so that
`importlib.import_module("sdot.bindings.<name>")` finds them transparently.
"""
from .._cache import bindings_cache_dir

_cache = bindings_cache_dir()
_cache.mkdir(parents=True, exist_ok=True)

__path__.append(str(_cache))
