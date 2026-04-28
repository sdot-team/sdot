from pathlib import Path
import platform
import sys
import os

def dylib_dir() -> Path:
    """Directory that holds compiled extension modules (.so / .dylib)."""
    d = _base_dir() / "dylibs" / _abi_tag()
    d.mkdir( parents = True, exist_ok = True )
    return d


def src_dir( name: str ) -> Path:
    """Directory for the generated C++ source of a named binding."""
    d = dylib_dir() / "src" / name
    d.mkdir( parents = True, exist_ok = True )
    return d

def generated_includes_dir() -> Path:
    """Directory for the generated C++ source of a named binding."""
    d = additional_includes_dir() / "sdot" / "generated_includes"
    d.mkdir( parents = True, exist_ok = True )
    return d

def additional_includes_dir() -> Path:
    """Directory for the generated C++ source of a named binding."""
    return dylib_dir() / "include"

def _base_dir() -> Path:
    if d := os.getenv( "SDOT_CACHE_DIR" ):
        return Path( d )
    if sys.platform == "win32":
        base = Path( os.getenv( "LOCALAPPDATA" ) or ( Path.home() / "AppData" / "Local" ) )
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path( os.getenv( "XDG_CACHE_HOME" ) or ( Path.home() / ".cache" ) )
    return base / "sdot"


def _abi_tag() -> str:
    return f"{ sys.implementation.cache_tag }-{ platform.machine() }"
