from sdot.generated_files import compilation_directories
from ..drivers.Device import Device
from pathlib import Path
import subprocess
import shutil
import sys
import os


def make_executable_from_files( exe_name: str, src_paths: list, device: Device, requires = None ) -> Path:
    """Build a standalone executable from *src_paths* using the shared compilation/xmake.lua.

    Counterpart of make_dylib_from_files for the C++/CUDA tests: produces a binary
    (SDOT_XMAKE_KIND=binary), links Catch2 instead of nanobind, and — for CUDA — routes the
    sources through nvcc via a generated .cu shim (nvcc is selected by the .cu extension,
    exactly like the bindings). Returns the path to the built executable.
    """
    project_root = Path( __file__ ).absolute().parents[ 4 ]
    requires     = list( requires or [ "catch2" ] )
    src_paths    = [ Path( p ) for p in src_paths ]

    # CUDA: wrap the .cpp sources in a .cu shim so nvcc compiles them (defines __CUDACC__)
    if device.is_cuda_gpu:
        shim = compilation_directories.src_dir( exe_name ) / f"{ exe_name }.cu"
        shim.write_text( "".join( f'#include "{ p }"\n' for p in src_paths ) )
        sources = [ shim ]
    else:
        sources = src_paths

    extended_path = os.pathsep.join( p for p in [
        str( Path( sys.executable ).parent ),
        str( Path.home() / ".local" / "bin" ),  # default xmake.io install
        "/opt/homebrew/bin",                     # homebrew Apple Silicon
        "/usr/local/bin",                        # homebrew Intel
        os.environ.get( "PATH", "" ),
    ] if p )

    xmake_bin = shutil.which( "xmake", path = extended_path )
    if xmake_bin is None:
        raise RuntimeError( "xmake introuvable (brew install xmake ou https://xmake.io)" )

    output_dir = compilation_directories.dylib_dir() / "tests"
    output_dir.mkdir( parents = True, exist_ok = True )

    env = {
        **os.environ,
        **( { "XMAKE_ROOT": "y" } if hasattr( os, "getuid" ) and os.getuid() == 0 else {} ),
        "SDOT_XMAKE_KIND"      : "binary",
        "SDOT_XMAKE_TARGET"    : exe_name,
        "SDOT_XMAKE_OUTPUT_DIR": str( output_dir ),
        "SDOT_XMAKE_NEEDS_CUDA": str( int( device.is_cuda_gpu ) ),
        "SDOT_XMAKE_REQUIRES"  : ",".join( requires ),
        "SDOT_XMAKE_INCLUDES"  : str.join( ",", map( str, [
                                      project_root / "src" / "cpp",
                                      compilation_directories.additional_includes_dir()
                                  ] ) ),
        "SDOT_XMAKE_CXXFLAGS"  : "-fno-strict-aliasing",
        "SDOT_XMAKE_SOURCES"   : ",".join( map( str, sources ) ),
        "SDOT_XMAKE_DEFINES"   : "",
        "PATH"                 : extended_path,
    }

    sdot_dir = Path( __file__ ).parent  # holds xmake.lua
    mode     = os.environ.get( "SDOT_XMAKE_MODE", "release" )

    def run( cmd ):
        if subprocess.run( cmd, cwd = output_dir, env = env ).returncode:
            raise RuntimeError( f"xmake failed: { ' '.join( map( str, cmd ) ) }" )

    run( [ xmake_bin, "f", "-P", str( sdot_dir ), "-y", "--require=yes", "-m", mode ] )
    run( [ xmake_bin, "-P", str( sdot_dir ), "-v" ] )

    return output_dir / exe_name
