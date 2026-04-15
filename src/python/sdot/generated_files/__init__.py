"""sdot.generated_files — storage for compiled extensions and generated C++ sources.

The singleton ``generated_files`` exposes the directories where artefacts are
written.  Adding its dylib directory to ``__path__`` makes compiled modules
importable as ``sdot.generated_files.<name>``.
"""

import platform
import sys
import os
from pathlib import Path


class GeneratedFiles:
    """Manages where sdot writes and finds generated artefacts."""

    @property
    def dylib_dir( self ) -> Path:
        """Directory that holds compiled extension modules (.so / .dylib)."""
        d = self._base_dir / "dylibs" / self._abi_tag
        d.mkdir( parents = True, exist_ok = True )
        return d

    def src_dir( self, name: str ) -> Path:
        """Directory for the generated C++ source of a named binding."""
        d = self.dylib_dir / "src" / name
        d.mkdir( parents = True, exist_ok = True )
        return d

    @property
    def _base_dir( self ) -> Path:
        if d := os.getenv( "SDOT_CACHE_DIR" ):
            return Path( d )
        if sys.platform == "win32":
            base = Path( os.getenv( "LOCALAPPDATA" ) or ( Path.home() / "AppData" / "Local" ) )
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Caches"
        else:
            base = Path( os.getenv( "XDG_CACHE_HOME" ) or ( Path.home() / ".cache" ) )
        return base / "sdot"

    @property
    def _abi_tag( self ) -> str:
        return f"{ sys.implementation.cache_tag }-{ platform.machine() }"


generated_files = GeneratedFiles()

# Make ``sdot.generated_files.<name>`` importable transparently.
__path__.append( str( generated_files.dylib_dir ) )
