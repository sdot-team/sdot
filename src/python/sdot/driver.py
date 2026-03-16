def add_buid_path():
    from pathlib import Path
    import sdot

    _build_path = Path( __file__ ).absolute().parents[ 3 ] / "build" / "src" / "python" / "sdot"
    if _build_path.exists() and str( _build_path ) not in sdot.__path__:
        sdot.__path__.append( str( _build_path ) )

# get path to dylibs in build
add_buid_path()

#
from .drivers.PyTorchDriver import PyTorchDriver
driver = PyTorchDriver()
