def add_buid_path():
    """
    Helper to add path to `build` directory content. Can be used when files are in the base repository (not installed e.g. via `pip -e`)
    """
    from pathlib import Path
    import sdot

    _build_path = Path( __file__ ).absolute().parents[ 3 ] / "build" / "src" / "python" / "sdot"
    if _build_path.exists() and str( _build_path ) not in sdot.__path__:
        sdot.__path__.append( str( _build_path ) )

def make_driver_instance( framework, device, dtype ):
    if str.lower( framework ) == "pytorch":
        from .drivers.PyTorchDriver import PyTorchDriver
        return PyTorchDriver( dtype = dtype, device = device )

    if str.lower( framework ) == "jax":
        from .drivers.JaxDriver import JaxDriver
        return JaxDriver( dtype = dtype, device = device )

    raise RuntimeError( f"{ framework } is not a registered framework name (for now, one can use 'pytorch' or 'jax')" )

class DriverProxy:
    """

    `dtype` is actually a "default" dtype (the user can use what he/she wants). same thing for `device`.
    """

    def __init__( self, framework = "pytorch", dtype = "float32", device = "cpu" ):
        self.framework = framework
        self.device = device
        self.dtype = dtype

        self._instance = None

    def __getattr__( self, name ):
        if self._instance is None:
            driver._instance = make_driver_instance( self.framework, self.device, self.dtype )

        return getattr( self._instance, name )

# add path to dylibs in `build` (needed by the drivers)
add_buid_path()

# start with an unknown driver instance
driver = DriverProxy()
