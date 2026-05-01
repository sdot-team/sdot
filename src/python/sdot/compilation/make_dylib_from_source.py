from sdot.generated_files import compilation_directories
from ..driver import driver
from pathlib import Path
import subprocess
import sysconfig
import nanobind
import shutil
import sys
import re
import os


def make_dylib_from_source( src: str, dylib_name: str, other_src_paths: list, device_type: str ):
    # write src
    path = compilation_directories.src_dir( dylib_name ) / "binding.cpp"
    path.write_text( src )

    # make the dylib
    make_dylib_from_files( dylib_name, [ path ] + other_src_paths, device_type )


def make_dylib_from_files( dylib_name: str, src_paths: list, device_type: str ):
    """Invoke xmake to build *dylib_name* from *src_paths*."""

    project_root = Path( __file__ ).absolute().parents[ 4 ]
    nanobind_include_dir = Path( nanobind.include_dir() )

    nanobind_ext_include = nanobind_include_dir.parent / "ext" / "robin_map" / "include"
    nanobind_source = Path( nanobind.source_dir() ) / "nb_combined.cpp"
    python_include = sysconfig.get_path( "include" )

    jax_include = []
    if driver.normalized_framework == "jax":
        try:
            import jax.ffi
            jax_include = [ jax.ffi.include_dir() ]
        except ImportError:
            pass

    xmake = shutil.which( "xmake" )
    if xmake is None:
        raise RuntimeError( "xmake not found in PATH — please install it (https://xmake.io)" )

    result = subprocess.run( [ xmake, "--version" ], capture_output = True, env = { "NO_COLOR": "1" } )
    xmake_number = re.search( r"(\d+\.\d+\.\d+)", result.stdout.decode() ).group( 1 )
    if tuple( map( int, xmake_number.split( "." ) ) ) < ( 3, 0, 8 ):
        raise RuntimeError(
            f"xmake >= 3.0.8 required (found { xmake_number }). "
            "Run: xmake update"
        )

    env = {
        **os.environ,
        "SDOT_XMAKE_OUTPUT_DIR" : str( compilation_directories.dylib_dir() ),
        "SDOT_XMAKE_NEEDS_CUDA" : str( int( device_type.startswith( "cuda" ) ) ),
        "SDOT_XMAKE_REQUIRES"   : str.join( ",", [ "zpp_bits", "eigen" ] ),
        "SDOT_XMAKE_INCLUDES"   : str.join( ",", map( str, [
                                      nanobind_ext_include,
                                      nanobind_include_dir,
                                      python_include,
                                      project_root / "src" / "cpp",
                                      compilation_directories.additional_includes_dir()
                                  ] + jax_include ) ),
        "SDOT_XMAKE_CXXFLAGS"   : str.join( ",", [ "-fdiagnostics-absolute-paths", "-fno-strict-aliasing" ] ),
        "SDOT_XMAKE_SOURCES"    : str.join( ",", map( str, list( src_paths ) + [ nanobind_source ] ) ),
        "SDOT_XMAKE_DEFINES"    : "",
        "SDOT_XMAKE_TARGET"     : dylib_name,
        "PATH"                  : str( Path( sys.executable ).parent ) + os.pathsep + os.environ.get( "PATH", "" ),
    }

    sdot_dir = Path( __file__ ).parent  # cpp_binding/ (contains xmake.lua)

    def run( cmd ):
        out = subprocess.run( cmd, cwd = compilation_directories.dylib_dir(), env = env )
        if out.returncode:
            sys.exit( out.returncode )

    run( [ "pwd" ] )
    run( [ xmake, "f", "-P", str( sdot_dir ), "-y", "--require=yes" ] )
    run( [ xmake, "-P", str( sdot_dir ) ] )
