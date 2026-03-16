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

class DriverProxy:
    def __init__( self, d ):
        self._d = d
    def __getattr__( self, name ):
        return getattr( self._d, name )

driver = DriverProxy( PyTorchDriver() )

def use_jax( **kwargs ):
    from .drivers.JaxDriver import JaxDriver
    driver._d = JaxDriver( **kwargs )

def use_pytorch( **kwargs ):
    from .drivers.PyTorchDriver import PyTorchDriver
    driver._d = PyTorchDriver( **kwargs )
