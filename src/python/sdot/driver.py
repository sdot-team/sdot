def add_buid_path():
    """
    Helper to add path to `build` directory content. Can be used when files are in the base repository (not installed e.g. via `pip -e`)
    """
    from pathlib import Path
    import sdot

    _build_path = Path( __file__ ).absolute().parents[ 3 ] / "build" / "src" / "python" / "sdot"
    if _build_path.exists() and str( _build_path ) not in sdot.__path__:
        sdot.__path__.append( str( _build_path ) )

def make_driver_instance( framework, scalar_type, device ):
    if str.lower( framework ) == "pytorch":
        from .drivers.PyTorchDriver import PyTorchDriver
        return PyTorchDriver( dtype = scalar_type, device = device )

    if str.lower( framework ) == "jax":
        from .drivers.JaxDriver import JaxDriver
        return JaxDriver( dtype = scalar_type, device = device )

    raise RuntimeError( f"{ framework } is not a registered framework name (for now, one can use 'pytorch' or 'jax')" )

class DriverProxy:
    """

    `scalar_type` is actually a "default" dtype (the user can use what he/she wants).
    Same thing for `device`.
    """

    def __init__( self, framework = "pytorch", scalar_type = "float32", device = "cpu" ):
        self._current_parameters = None
        self._instance = None

        self.scalar_type = scalar_type
        self.framework = framework
        self.device = device

    def __getattr__( self, name ):
        parameters = ( self.framework, self.scalar_type, self.device )
        if self._current_parameters != parameters:
            self._instance = make_driver_instance( *parameters )
            self._current_parameters = parameters

        return getattr( self._instance, name )

# add path to dylibs in `build` (needed by the drivers)
add_buid_path()

# start with an unknown driver instance
driver = DriverProxy()
