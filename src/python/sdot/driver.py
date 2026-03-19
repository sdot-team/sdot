import sys

class DriverProxy:
    """
    Type attributes:
        * `normalized_dtype`: a string like "FP32", following the TL (the Template Language) naming sheme. Used to import the right procedures
        * `user_dtype`: the string that was specified by the user
        * `dtype`: type used by the frawework, for instance `torch.float32`

    Device attributs:
        * `normalized_device`: a string like "gpu:0"
        * `user_device`: the string that was specified by the user
        * `device`: instance used by the frawework

    Framework if not specified by the user:
        * look what is imported in sys.modules
        * else, try is possible to import pytorch then jax
    """

    def __init__( self ):
        self._user_framework = None
        self._user_device = None
        self._user_dtype = None

        self._instance = None

    # ---------------------------------- framework ----------------------------------
    @property
    def framework( self ):
        return self._user_framework

    @framework.setter
    def framework( self, value: str ):
        if self._user_framework == value:
            return
        self._user_framework = value
        self._instance = self._make_driver_instance()

    # ---------------------------------- dtype ----------------------------------
    @property
    def normalized_dtype( self ) -> str:
        return str( self._normalized_dtype_for( self.user_dtype ) )

    @property
    def user_dtype( self ):
        if self._user_dtype is None:
            self._make_driver_instance()
        assert self._user_dtype is not None
        return self._user_dtype

    @property
    def dtype( self ):
        if self._instance is None:
            self._make_driver_instance()
        return self._instance.dtype

    @user_dtype.setter
    def user_dtype( self, value ):
        old_normalized_dtype = self._normalized_dtype_for( self._user_dtype )
        self._user_dtype = value
        if old_normalized_dtype != self._normalized_dtype_for( value ):
            self._make_driver_instance()

    @dtype.setter
    def dtype( self, value ):
        self.user_dtype = value

    # ---------------------------------- device ----------------------------------
    @property
    def normalized_device( self ) -> str:
        return str( self._normalized_device_for( self.user_device ) )

    @property
    def user_device( self ):
        if self._user_device is None:
            self._make_driver_instance()
        return self._user_device

    @property
    def device( self ):
        if self._instance is None:
            self._make_driver_instance()
        return self._instance.device

    @user_device.setter
    def user_device( self, value ):
        old_normalized_device = self._normalized_device_for( self._user_device )
        self._user_device = value
        if old_normalized_device != self._normalized_device_for( value ):
            self._make_driver_instance()

    @device.setter
    def device( self, value ):
        self.user_device = value

    # ---------------------------------- __getattr__ ----------------------------------
    def __getattr__( self, name ):
        if self._instance is None:
            self._make_driver_instance()
        return getattr( self._instance, name )

    # ---------------------------------- helpers ----------------------------------
    def _normalized_dtype_for( self, name ) -> str | None:
        if name is None:
            return None
        
        if not isinstance( name, str ):
            return self._normalized_dtype_for( str( name ) )

        if str.lower( name ) in [ "float32", "fp32" ]:
            return "FP32"
        if str.lower( name ) in [ "float64", "fp64" ]:
            return "FP64"

        raise RuntimeError( f"TODO { name }" )

    def _normalized_device_for( self, name ) -> str | None:
        if name is None:
            return None
        
        if not isinstance( name, str ):
            return self._normalized_device_for( str( name ) )

        if name == "cpu":
            return "cpu"

        if name in [ "mps", "metal" ]:
            return "metal"

        raise RuntimeError( f"TODO { name }" )

    def _make_driver_instance( self ):
        if self._user_framework is None:
            self._user_framework = self._find_framework()

        if str.lower( self._user_framework ) in [ "pytorch", "torch" ]:
            from .drivers.PyTorchDriver import PyTorchDriver
            self._instance = self._call_driver_instance( PyTorchDriver )
            return 

        if str.lower( self._user_framework ) == "jax":
            from .drivers.JaxDriver import JaxDriver
            self._instance = self._call_driver_instance( JaxDriver )
            return 

        raise RuntimeError( f"{ self._user_framework } is not a registered framework name (for now, one can use 'pytorch' or 'jax')" )

    def _call_driver_instance( self, Class ):
        # we start by getting the type
        if self._user_dtype is None:
            self._user_dtype = self._normalized_dtype_for( Class.default_dtype( self._normalized_device_for( self._user_device ) ) )

        # and then we find a hardware able to handle this type
        if self._user_device is None:
            self._user_device = self._normalized_device_for( Class.default_device( self._normalized_dtype_for( self._user_dtype ) ) )

        return Class( normalized_dtype = self._normalized_dtype_for( self._user_dtype ), normalized_device = self._normalized_device_for( self._user_device ) )

    def _find_framework( self ) -> str:
        # try imported modules
        for name in [ 'jax', 'torch' ]:
            if name in sys.modules:
                return name

        # # already imported
        # try:
        #     import torch
        #     return "pytorch"
        # except ImportError:
        #     pass

        # try:
        #     import jax
        #     return "jax"
        # except ImportError:
        #     pass

        raise RuntimeError( "Unable to find a driver (tested torch and jax)" )


def add_buid_path():
    """
    Helper to add path to `build` directory content. Can be used when files are in the base repository (not installed e.g. via `pip -e`)
    """
    from pathlib import Path
    import sdot

    _build_path = Path( __file__ ).absolute().parents[ 3 ] / "build" / "src" / "python" / "sdot"
    if _build_path.exists() and str( _build_path ) not in sdot.__path__:
        sdot.__path__.append( str( _build_path ) )


# add path to dylibs in `build` (needed by the drivers)
add_buid_path()

# start with an unknown driver instance
driver = DriverProxy()
