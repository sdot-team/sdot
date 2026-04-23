from .drivers.compilation.Mutable import Mutable as Mutable
from .drivers.compilation.Return import Return as Return
from .drivers.compilation.Tensor import Tensor as Tensor
from .CtInt import CtInt as CtInt
from typing import TYPE_CHECKING, Any
import numpy
import sys
import os


class DriverProxy:
    """
    Type attributes:
        * `normalized_dtype`: a string like "FP32", following the TL (the Template Language) naming sheme. Used to import the right procedures
        * `user_dtype`: the string that was specified by the user
        * `dtype`: type used by the frawework, for instance `torch.float32`

        User can write sdot.driver.dtype = ... with any format ("float32", "FP32", torch.float32, ...)

    Device attributs:
        * `normalized_device`: a string like "cuda:0"
        * `user_device`: the string that was specified by the user
        * `device`: instance used by the frawework

    Framework attributes:
        * `normalized_framework`: a string like "jax", "torch", ...
        * `user_framework`: the string that was specified by the user
        * `framework`: same thing than normalized_framework

    To find the default framework:
        * look what is imported in sys.modules
        * else, try if possible to import a module in self.prefered_frameworks ([ 'jax', 'torch' ] by default)

    Env variables that are taken into account
        * SDOT_FRAMEWORK
        * SDOT_DEVICE
        * SDOT_DTYPE
    """

    if TYPE_CHECKING:
        def t3( self, tensor, dtype = None ) -> Any: ...
        def t2( self, tensor, dtype = None ) -> Any: ...
        def t1( self, tensor, dtype = None ) -> Any: ...
        def t0( self, tensor, dtype = None ) -> Any: ...
        def tn( self, tensor, ndim = None, name = None, dtype = None ) -> Any: ...
        def array( self, a ) -> Any: ...
        def zeros( self, shape ) -> Any: ...
        def ones( self, shape ) -> Any: ...
        def linspace( self, a, b, n ) -> Any: ...
        def empty( self, shape, dtype = None ) -> Any: ...
        def expand_dims( self, tensor, index ) -> Any: ...
        def repeat( self, tensor, shape ) -> Any: ...
        def stack( self, tensors, axis ) -> Any: ...
        def linalg_solve( self, A, b ) -> Any: ...
        def moveaxis( self, tensor, source, destination ) -> Any: ...
        def hstack( self, lst ) -> Any: ...
        def to_numpy( self, t ) -> numpy.ndarray: ...
        def forward( self, forward_func, backward_func, args, input_tensors, output_tensors ) -> Any: ...
        def is_int_dtype( self, dtype ) -> bool: ...
        def any_requires_grad( self, tensors ) -> bool: ...
        array_type: type
        int_type: type

    def __init__( self ):
        self.prefered_frameworks = [ 'jax', 'torch' ]

        self._user_framework = None
        self._user_device = None
        self._user_dtype = None

        self._instance = None

        if d := os.getenv( "SDOT_FRAMEWORK" ):
            self.framework = d
        if d := os.getenv( "SDOT_DEVICE" ):
            self.device = d
        if d := os.getenv( "SDOT_DTYPE" ):
            self.dtype = d

    # ---------------------------------- framework ----------------------------------
    @property
    def normalized_framework( self ) -> str:
        # specified by the user ?
        if isinstance( self._user_framework, str ):
            return self._normalized_framework_for( self._user_framework )

        # else, look in imported modules
        for name in self.prefered_frameworks:
            if name in sys.modules:
                self._user_framework = name
                return name

        # try to import jax
        for name in self.prefered_frameworks:
            try:
                __import__( name )
                self._user_framework = name
                return name
            except ImportError:
                pass

        # argh
        raise RuntimeError( "Unable to find a driver (tested torch and jax)" )

    @property
    def user_framework( self ) -> str | None:
        return self._user_framework

    @property
    def framework( self ) -> str:
        return self.normalized_framework

    @user_framework.setter
    def user_framework( self, value: str ):
        self._user_framework = value

        # need an _instance update ?
        if value is None or ( self._instance is not None and self._instance.name != self._normalized_framework_for( value ) ):
            self._instance = None

    @framework.setter
    def framework( self, value: str ):
        self.user_framework = value

    # ---------------------------------- dtype ----------------------------------
    @property
    def normalized_dtype( self ) -> str:
        # specified by the user ?
        if isinstance( self.user_dtype, str ):
            return str( self._normalized_dtype_for( self.user_dtype ) )

        # metal => FP32
        if self.normalized_framework == "metal":
            return "FP32"

        return "FP64"

    @property
    def user_dtype( self ):
        return self._user_dtype

    @property
    def dtype( self ):
        # we need an instance to get the correct dtype object
        if self._instance is None:
            self._make_driver_instance()
        return self._instance.dtype

    @user_dtype.setter
    def user_dtype( self, value ):
        self._user_dtype = value

        # need an _instance update ?
        if value is None or ( self._instance is not None and self._normalized_dtype_for( self._instance.dtype ) != self._normalized_dtype_for( value ) ):
            self._instance = None

    @dtype.setter
    def dtype( self, value ):
        self.user_dtype = value


    # ---------------------------------- itype ----------------------------------
    @property
    def numpy_itype( self ):
        return numpy.int64

    # ---------------------------------- device ----------------------------------
    @property
    def normalized_device_type( self ) -> str:
        """ without :x """
        return self.normalized_device.split( ":" )[ 0 ]

    @property
    def normalized_device( self ) -> str:
        # specified by the user ?
        if isinstance( self._user_device, str ):
            return self._normalized_device_for( self._user_device )

        # numpy => cpu
        if self._user_framework is not None and self._normalized_framework_for( self._user_framework ) == "numpy":
            return "cpu"

        # get a driver class in order to get a default device
        Driver = DriverProxy._driver_class( self.normalized_framework )
        return Driver.default_normalized_device_for( self._normalized_dtype_for( self._user_dtype ) if self._user_dtype else None )

    @property
    def user_device( self ):
        return self._user_device

    @property
    def device( self ):
        if self._instance is None:
            self._make_driver_instance()
        return self._instance.device

    @user_device.setter
    def user_device( self, value ):
        self._user_device = value

        # need an _instance update ?
        if value is None or ( self._instance is not None and self._normalized_device_for( self._instance.device ) != self._normalized_device_for( value ) ):
            self._instance = None

    @device.setter
    def device( self, value ):
        self.user_device = value

    # ---------------------------------- arrays ----------------------------------
    def array( self, a ):
        return self.tn( a, None )

    # ---------------------------------- __getattr__ ----------------------------------
    def __getattr__( self, name ):
        if self._instance is None:
            self._make_driver_instance()
        return getattr( self._instance, name )

    # ---------------------------------- helpers ----------------------------------
    def _normalized_framework_for( self, name ) -> str:
        # ensure lowercase str
        if not isinstance( name, str ):
            return self._normalized_framework_for( str( name ) )
        name = name.lower()

        if name in [ "torch", "pytorch" ]:
            return "torch"

        if name in [ "jax" ]:
            return "jax"

        raise RuntimeError( f"Unknown framework type '{ name }'" )

    def _normalized_dtype_for( self, name ) -> str:
        # ensure lowercase str
        if not isinstance( name, str ):
            return self._normalized_dtype_for( str( name ) )
        name = name.lower()

        if ( "float32" in name ) or ( "fp32" in name ):
            return "FP32"

        if ( "float64" in name ) or ( "fp64" in name ):
            return "FP64"

        raise RuntimeError( f"TODO: find normalized dtype for { name }" )

    def _normalized_device_for( self, name ) -> str:
        # ensure lowercase str
        if not isinstance( name, str ):
            return self._normalized_device_for( str( name ) )
        name = name.lower()

        if name.startswith( "cuda" ) or name.startswith( "gpu" ):
            if ":" in name:
                raise RuntimeError( "TODO: other gpus" )
            return "cuda:0"

        if name in [ "mps", "metal" ]:
            if ":" in name:
                raise RuntimeError( "TODO: other gpus" )
            return "metal"

        if "cpu" in name:
            return "cpu"

        raise RuntimeError( f"TODO { name }" )

    @staticmethod
    def _driver_class( normalized_framework ):
        if normalized_framework == "torch":
            from .drivers.PyTorchDriver import PyTorchDriver
            return PyTorchDriver

        if normalized_framework == "jax":
            from .drivers.JaxDriver import JaxDriver
            return JaxDriver

        raise RuntimeError( f"{ normalized_framework } is not a registered framework name (for now, one can use 'torch' or 'jax')" )

    def _make_driver_instance( self ):
        Driver = DriverProxy._driver_class( self.normalized_framework )
        self._instance = Driver( self.normalized_dtype, self.normalized_device )


# def add_buid_path():
#     """
#     Helper to add path to `build` directory content. Can be used when files are in the base repository (not installed e.g. via `pip -e`)
#     """
#     from pathlib import Path
#     import sdot

#     _build_path = Path( __file__ ).absolute().parents[ 3 ] / "build" / "src" / "python" / "sdot"
#     if _build_path.exists() and str( _build_path ) not in sdot.__path__:
#         sdot.__path__.append( str( _build_path ) )


# # add path to dylibs in `build` (needed by the drivers)
# add_buid_path()

# start with an unknown driver instance
driver = DriverProxy()
