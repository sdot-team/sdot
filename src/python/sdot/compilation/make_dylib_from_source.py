from sdot.generated_files import compilation_directories
from ..drivers.Device import Device
from ..drivers.driver import driver
from pathlib import Path
import subprocess
import sysconfig
import nanobind
import shutil
import sys
import re
import os


def make_dylib_from_source( code: str, dylib_name: str, other_src_paths: list, device: Device ):
    # .cu extension lets xmake route the file through nvcc (defines __CUDACC__)
    ext = ".cu" if device.is_cuda_gpu else ".cpp"
    path = compilation_directories.src_dir( dylib_name ) / f"binding{ ext }"
    try:
        old_code = path.read_text()
    except FileNotFoundError:
        old_code = ""
    if code != old_code:
        path.write_text( code )

    # make the dylib
    make_dylib_from_files( dylib_name, [ path ] + other_src_paths, device )


def make_dylib_from_files( dylib_name: str, src_paths: list, device: Device ):
    """Invoke xmake to build *dylib_name* from *src_paths*."""

    project_root = Path( __file__ ).absolute().parents[ 4 ]
    nanobind_include_dir = Path( nanobind.include_dir() )

    nanobind_ext_include = nanobind_include_dir.parent / "ext" / "robin_map" / "include"
    nanobind_source = Path( nanobind.source_dir() ) / "nb_combined.cpp"
    python_include = sysconfig.get_path( "include" )

    jax_include = []
    if driver.framework == "jax":
        try:
            import jax.ffi
            jax_include = [ jax.ffi.include_dir() ]
        except ImportError:
            pass

    extended_path = os.pathsep.join( p for p in [
        str( Path( sys.executable ).parent ),
        sysconfig.get_path( "scripts" ),
        str( Path.home() / ".local" / "bin" ),  # install par défaut du script xmake.io
        "/opt/homebrew/bin",                     # homebrew Apple Silicon
        "/usr/local/bin",                        # homebrew Intel / install --prefix=/usr/local
        os.environ.get( "PATH", "" ),
    ] if p )

    _xmake_bin = shutil.which( "xmake", path = extended_path )
    if _xmake_bin is None:
        searched = "\n  ".join( p for p in extended_path.split( os.pathsep ) if p )
        raise RuntimeError(
            f"xmake introuvable. Chemins cherchés :\n  { searched }\n\n"
            "Pour localiser xmake : lancez 'which xmake' dans votre terminal.\n"
            "Installation : brew install xmake  ou  https://xmake.io"
        )
    xmake_cmd = [ _xmake_bin ]

    # result = subprocess.run( [ *xmake_cmd, "--version" ], capture_output = True, env = { "NO_COLOR": "1" } )
    # xmake_number = re.search( r"(\d+\.\d+\.\d+)", result.stdout.decode() ).group( 1 )
    # if tuple( map( int, xmake_number.split( "." ) ) ) < ( 3, 0, 8 ):
    #     raise RuntimeError(
    #         f"xmake >= 3.0.8 requis (trouvé { xmake_number }). Lancez : xmake update"
    #     )

    env = {
        **os.environ,
        **( { "XMAKE_ROOT": "y" } if hasattr( os, "getuid" ) and os.getuid() == 0 else {} ),
        "SDOT_XMAKE_OUTPUT_DIR" : str( compilation_directories.dylib_dir() ),
        "SDOT_XMAKE_NEEDS_CUDA" : str( int( device.is_cuda_gpu ) ),
        "SDOT_XMAKE_REQUIRES"   : str.join( ",", [ "zpp_bits", "eigen" ] ),
        "SDOT_XMAKE_INCLUDES"   : str.join( ",", map( str, [
                                      nanobind_ext_include,
                                      nanobind_include_dir,
                                      python_include,
                                      project_root / "src" / "cpp",
                                      compilation_directories.additional_includes_dir()
                                  ] + jax_include ) ),
        "SDOT_XMAKE_CXXFLAGS"   : str.join( ",", [ "-fno-strict-aliasing" ] ),
        "SDOT_XMAKE_SOURCES"    : str.join( ",", map( str, list( src_paths ) + [ nanobind_source ] ) ),
        "SDOT_XMAKE_DEFINES"    : "",
        "SDOT_XMAKE_TARGET"     : dylib_name,
        "PATH"                  : extended_path,
    }

    sdot_dir = Path( __file__ ).parent  # cpp_binding/ (contains xmake.lua)

    def run( cmd ):
        out = subprocess.run( cmd, cwd = compilation_directories.dylib_dir(), env = env )
        if out.returncode:
            sys.exit( out.returncode )

    xmake_mode = os.environ.get( "SDOT_XMAKE_MODE", "release" )

    run( [ "pwd" ] )
    run( [ *xmake_cmd, "f", "-P", str( sdot_dir ), "-y", "--require=yes", "-m", xmake_mode ] )
    run( [ *xmake_cmd, "-P", str( sdot_dir ), "-v" ] )
