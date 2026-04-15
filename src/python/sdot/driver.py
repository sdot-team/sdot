from pathlib import Path
import importlib
import textwrap
import inspect
import hashlib
import struct
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

    def __init__( self ):
        self.prefered_frameworks = [ 'jax', 'torch' ]

        self._user_framework = None
        self._user_device = None
        self._user_dtype = None

        self._dylib_cache = {}
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
    def cpp_src( self, repl: dict[str,any], text: str ):
        """Dedent a triple-quoted C++ source string and prepend a #line directive
        pointing back to the Python file/line where the call was made, so that
        compiler errors link directly to the Python source.
        """

        text = textwrap.dedent( text ).lstrip( "\n" )

        for k, v in repl.items():
            text = text.replace( k, str( v ) )

        bmap = {
            "SDOT_NANOBIND_ARCH": f"nanobind::device::{ self.normalized_device_type }",
            "SDOT_SCALAR_TYPE"  : self.normalized_dtype,
            "SDOT_ARCH"         : str.capitalize( self.normalized_device_type ),
        }
        for k, v in bmap.items():
            text = text.replace( k, v )

        #
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename.replace( "\\", "/" )
        lineno   = frame.f_lineno + 1   # +1: content starts on the line after the opening """
        return f'#line { lineno } "{ filename }"\n' + text

    def import_bindings( self, dylib_name: str, src_func = None, src_paths: list[ Path ] = [] ) -> any:
        # in the cache ?
        if dylib_name in self._dylib_cache:
            return self._dylib_cache[ dylib_name ]

        # the dylib already exists ?
        full_name = "sdot.bindings." + dylib_name
        if "SDOT_BUILD" not in os.environ or int( os.environ[ "SDOT_BUILD" ] ) == 0:
            try:
                return importlib.import_module( full_name )
            except ( ImportError, SystemError ):
                pass

        # else, make the source, compile
        self.compile_binding_for( dylib_name, src_func, src_paths )

        # and try again to import it
        importlib.invalidate_caches()
        res = importlib.import_module( full_name )
        self._dylib_cache[ dylib_name ] = res
        return res

    def compile_binding_for( self, dylib_name: str, src_func, src_paths: list[ Path ] ):
        import subprocess
        import sysconfig
        import nanobind
        import shutil
        import sys
        import os
        import re

        #
        from ._cache import bindings_cache_dir

        project_root = Path( __file__ ).absolute().parents[ 3 ]

        # dir for the .so
        dylib_dir = bindings_cache_dir()

        # dir for the generated sources
        src_dir = dylib_dir / "src" / dylib_name
        src_dir.mkdir( parents = True, exist_ok = True )

        # nanobind paths
        nanobind_include_dir = nanobind.include_dir()
        nanobind_ext_include = os.path.join( os.path.dirname( nanobind_include_dir ), "ext", "robin_map", "include" )
        nanobind_source = os.path.join( nanobind.source_dir(), "nb_combined.cpp" )
        python_include = sysconfig.get_path( "include" )

        # make the source text
        txt = src_func()
        txt = txt.replace( "SDOT_BINDING_NAME", dylib_name )

        # file extension
        ext = ".cpp"
        if self.normalized_device_type == "cuda":
            ext = ".cu"

        # store the source text
        bnd_path = src_dir / ( "binding" + ext )
        bnd_path.write_text( txt )

        # find the xmake utility
        xmake = shutil.which( "xmake" )
        if xmake is None:
            raise RuntimeError( "xmake introuvable dans PATH — installez-le (https://xmake.io)" )

        # check xmake version
        xmake_version = subprocess.run( [ xmake, "--version" ], capture_output = True, env = { "NO_COLOR": "1" } )
        xmake_number = re.search( r"(\d+.\d+.\d+)", xmake_version.stdout.decode() ).group( 1 )
        if tuple( map( int, xmake_number.split( "." ) ) ) < ( 3, 0, 8 ):
            raise RuntimeError( f"we need xmake to be at least version 3.0.8 (found version '{ xmake_number }'). Consider making a xmake update (-> will probably instal it in ~/.local/bin)" )

        #
        defines = [
            f"SDOT_NANOBIND_ARCH={ self.normalized_device_type }",
            f"SDOT_SCALAR_TYPE={ driver.normalized_dtype }",
            f"SDOT_ARCH={ self.normalized_device_type }",
        ]

        #
        env = {
            **os.environ,
            "SDOT_BINDING_NAME"      : dylib_name,
            "SDOT_SRC_INCLUDE"       : str( project_root / "src" / "cpp" ),
            "SDOT_OUTPUT_DIR"        : str( dylib_dir ),
            "SDOT_SRC_FILES"         : str.join( ",", map( str, [ bnd_path ] + src_paths ) ),
            "SDOT_DEFINES"           : str.join( ",", defines ),
            "SDOT_ARCH"              : self.normalized_device_type,
            "SDOT_NANOBIND_INC"      : str( nanobind_include_dir ),
            "SDOT_NANOBIND_SRC"      : str( nanobind_source ),
            "SDOT_ROBIN_MAP_INC"     : str( nanobind_ext_include ),
            "SDOT_PYTHON_INC"        : str( python_include ),
            "PATH"                   : str( Path( sys.executable ).parent ) + os.pathsep + os.environ.get( "PATH", "" ),
        }
        sdot_dir = Path( __file__ ).parent
        def run( cmd ):
            out = subprocess.run( cmd, cwd = dylib_dir, env = env )
            if out.returncode:
                sys.exit( out.returncode )
        run( [ xmake, "f", "-P", str( sdot_dir ), "-y", "--require=yes" ] )
        run( [ xmake, "-P", str( sdot_dir ) ] )

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


def encode_base62( s: str, length: int = 11 ) -> str:
    chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    h = struct.unpack( '>Q', hashlib.sha256( s.encode() ).digest()[ :8 ] )[ 0 ]
    res = []
    for _ in range( length ):
        res.append( chars[ h % 62 ] )
        h //= 62
    return ''.join( reversed( res ) )

def tensor_conv_for( type, name, rank ):
    if "vector" in type:
        tt = "const TF"
        if "MF" in type:
            tt = "TF"
        return f"auto { name } = tensor_views_{ rank }( _{ name } );"

    return f"auto { name } = tensor_view_{ rank }( _{ name } );"


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
