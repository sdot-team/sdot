from pathlib import Path
import importlib
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
                return name

        # try to import jax
        for name in self.prefered_frameworks:
            try:
                __import__( name )
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

    # ---------------------------------- __getattr__ ----------------------------------
    def __getattr__( self, name ):
        if self._instance is None:
            self._make_driver_instance()
        return getattr( self._instance, name )

    # ---------------------------------- helpers ----------------------------------
    def bindings_for( self, f, g ):
        geometry_dir = Path( __file__ ).parents[ 2 ] / "cpp" / "geometry"
        ct_dim = f.dim if f.dim <= 4 else -1
        f_name = f.__class__.__name__
        g_name = g.__class__.__name__

        dylib_name = f"ot_plan_{ f_name }_{ g_name }_{ ct_dim }d_{ driver.normalized_dtype }_{ driver.normalized_device_type }"

        full_name = "sdot.bindings.generated." + dylib_name
        try:
            mod = importlib.import_module( full_name )
        except ImportError:
            self.compile_binding_for( dylib_name, self.generate_bindings_sources_for( f, g, dylib_name ) + [
                geometry_dir / "SimpleSquareMatrix_eigen.cpp",
                geometry_dir / "VtkOutput.cpp",
            ] )
            mod = importlib.import_module( full_name )

        return mod

    def generate_bindings_sources_for( self, f, g, dylib_name ) -> list[ Path ]:
        src_dir = Path( __file__ ).parent / "bindings" / "generated" / "sources" / dylib_name
        Path.mkdir( src_dir, exist_ok = True, parents = True )

        #
        bnd_text = """
            #include "../../../../../../cpp/cpu/w2_distance.h"
            #include "../../../nanobind_wrappers.h"

            namespace nb = nanobind;
            using namespace sdot;

            using NA = nanobind::device::SDOT_NANOBIND_ARCH;
            using TF = SDOT_SCALAR_TYPE;

            using Arch = ArchFor<NA>::type;
            using NF = nb::ndarray<const TF,NA>;
            using MF = nb::ndarray<TF,NA>;

            // Forward function that works with both 1D and 2D arrays
            void forward( NF dirac_xs, NF dirac_ws, NF point_xs, NF point_ys, MF distance, MF barycenters, MF potentials, MF cuts ) {
                BatchOfAffine1d<const TF,Arch> functions{ .xs = tensor_view_2( point_xs ), .ys = tensor_view_2( point_ys ) };
                BatchOfDiracSet<const TF,Arch> diracs{ .xs = tensor_view_3( dirac_xs ).squeeze( 2 ), .ws = tensor_view_2( dirac_ws ) };
                w2_distance( diracs, functions, tensor_view_1( distance ), tensor_view_2( barycenters ), tensor_view_2( potentials ), tensor_view_3( cuts ) );
            }

            void backward(
                NF barycenters, NF potentials, NF cuts,
                NF dirac_xs, NF dirac_ws, NF point_xs, NF point_ys,
                NF grad_distances, NF grad_barycenters, NF grad_potentials, NF grad_cuts,
                MF grad_dirac_xs, MF grad_dirac_ws, MF grad_point_xs, MF grad_point_ys
            ) {
                BatchOfAffine1d<TF,Arch> grad_functions{ .xs = tensor_view_2( grad_point_xs ), .ys = tensor_view_2( grad_point_ys ) };
                BatchOfDiracSet<TF,Arch> grad_diracs{ .xs = tensor_view_3( grad_dirac_xs ).squeeze( 2 ), .ws = tensor_view_2( grad_dirac_ws ) };
                BatchOfAffine1d<const TF,Arch> functions{ .xs = tensor_view_2( point_xs ), .ys = tensor_view_2( point_ys ) };
                BatchOfDiracSet<const TF,Arch> diracs{ .xs = tensor_view_3( dirac_xs ).squeeze( 2 ), .ws = tensor_view_2( dirac_ws ) };
                w2_distance_backward( tensor_view_1( grad_distances ), tensor_view_2( grad_barycenters ), tensor_view_2( barycenters ), tensor_view_2( potentials ), tensor_view_3( cuts ), diracs, functions, grad_diracs, grad_functions );
            }

            #define MK_MOD( NAME ) NB_MODULE( NAME, m )

            MK_MOD( SDOT_BINDING_NAME ) {
                m.def( "forward", &forward,
                    nb::arg( "dirac_xs" ), nb::arg( "dirac_ws" ), nb::arg( "point_xs" ), nb::arg( "point_ys" ),
                    nb::arg( "distance" ), nb::arg( "barycenters" ), nb::arg( "potentials" ), nb::arg( "cuts" ),
                    "SDOT plan to get distance and barycenters with f = sum of diracs (1D), g = piecewise affine function"
                );

                m.def( "backward", &backward,
                    nb::arg( "barycenters" ), nb::arg( "potentials" ), nb::arg( "cuts" ),
                    nb::arg( "dirac_xs" ), nb::arg( "dirac_ws" ), nb::arg( "points_xs" ), nb::arg( "points_ys" ),
                    nb::arg( "grad_distance" ), nb::arg( "grad_barycenters" ), nb::arg( "grad_potentials" ), nb::arg( "grad_cuts" ),
                    nb::arg( "grad_dirac_xs" ), nb::arg( "grad_dirac_ws" ), nb::arg( "grad_points_xs" ), nb::arg("grad_points_ys"),
                    "SDOT W2 backward implementation"
                );
            }
        """

        bnd_text = bnd_text.replace( "SDOT_NANOBIND_ARCH", self.normalized_device_type )
        bnd_text = bnd_text.replace( "SDOT_SCALAR_TYPE", self.normalized_dtype )

        ext = "cpp"
        if self.normalized_device_type == "cuda":
            ext = ".cu"

        bnd_path = src_dir / ( "ot_plan." + ext )
        bnd_path.write_text( bnd_text )

        #
        return [ bnd_path ]



    def compile_binding_for( self, dylib_name: str, src_paths: list[ Path ] ):
        import subprocess
        import shutil
        import sys
        import os
        import re

        project_root = Path( __file__ ).absolute().parents[ 3 ]
        bindings_src = Path( __file__ ).parent / "bindings"

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
        ]

        #
        env = {
            **os.environ,
            "SDOT_BINDING_NAME" : dylib_name,
            "SDOT_SRC_INCLUDE"  : str( project_root / "src" ),
            "SDOT_OUTPUT_DIR"   : str( bindings_src / "generated" ),
            "SDOT_SRC_FILES"    : str.join( ",", map( str, src_paths ) ),
            "SDOT_DEFINES"      : str.join( ",", defines ),
            "SDOT_ARCH"         : self.normalized_device_type,
            "PATH"              : str( Path( sys.executable ).parent ) + os.pathsep + os.environ.get( "PATH", "" ),
        }
        subprocess.run( [ xmake, "f", "-y" ], cwd = bindings_src, env = env, check = True )
        subprocess.run( [ xmake ], cwd = bindings_src, env=env, check = True )

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
