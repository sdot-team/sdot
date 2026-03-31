from pathlib import Path
import platform
import sys
import os


def sdot_cache_dir() -> Path:
    """Base sdot cache directory.  Respects SDOT_CACHE_DIR env var; otherwise
    follows platform conventions (~/.cache/sdot on Linux/Mac,
    %LOCALAPPDATA%/sdot on Windows).
    """
    if d := os.getenv( "SDOT_CACHE_DIR" ):
        return Path(d)

    if sys.platform == "win32":
        base = Path( os.getenv("LOCALAPPDATA") or ( Path.home() / "AppData" / "Local" ) )
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path( os.getenv( "XDG_CACHE_HOME" ) or ( Path.home() / ".cache" ) )

    return base / "sdot"


def bindings_cache_dir() -> Path:
    """Directory where compiled binding extensions (.so/.dylib) are stored.
    Includes an ABI tag so different Python versions / architectures coexist.
    """
    abi_tag = f"{sys.implementation.cache_tag}-{platform.machine()}"
    return sdot_cache_dir() / "bindings" / abi_tag
