"""All compilation logic for sdot C++ bindings.

Two entry points:
- ``get_module_for(func_list)``  — used by the generic ``cpp_binding`` machinery.
- ``import_binding(dylib_name, src_func, src_paths, device_type)``
  — used by ``DriverProxy.import_bindings``.

Both ultimately call ``_compile(dylib_name, src_paths, device_type)`` after the
source file has been written to the build cache.
"""

from ._types import cpp_class_name_for, to_standard_objects, from_standard_objects
from ._util import encode_base62

from pathlib import Path

import subprocess
import importlib
import sysconfig
import shutil
import sys
import os
import re


class CppFunc:
    """

    """
    def __init__( self, name, args, includes ):
        self.includes = includes
        self.name = name
        self.args = args

    def key( self ) -> str:
        res = str.join( "_", [ self.name ] + list( map( cpp_class_name_for, self.args ) ) + list( self.includes ) )
        res = res.replace( "<", "_" )
        res = res.replace( ">", "_" )
        res = res.replace( "/", "_" )
        res = res.replace( ".", "_" )
        res = res.replace( ",", "_" )
        if len( res ) > 40:
            res = res[ : 40 - 11 ] + encode_base62( res[ 40 - 11: ] )
        return res


# ── module-level import cache ────────────────────────────────────────────────

_dylib_cache: dict = {}


# ── helpers ──────────────────────────────────────────────────────────────────

def force_build() -> bool:
    return "SDOT_FORCE_BUILD" in os.environ and bool( int( os.environ[ "SDOT_FORCE_BUILD" ] ) )


def _module_name( dylib_name: str ) -> str:
    return "sdot.generated_files." + dylib_name


def _try_import( dylib_name: str ):
    """Try importing an already-compiled module; return it or None."""
    try:
        return importlib.import_module( _module_name( dylib_name ) )
    except ( ImportError, SystemError ):
        return None


# ── public entry points ───────────────────────────────────────────────────────

def get_module_for( func_list: list ):
    """Return (importing if necessary) the nanobind module that exposes all
    functions in *func_list*.
    """
    # lazy import avoids circular dependency: driver.py → _build.py → driver.py
    from ..driver import driver
    from ._types import cpp_class_name_for, to_standard_objects

    dylib_name = str.join( "__", [ func.key() for func in func_list ] )
    if len( dylib_name ) > 40:
        dylib_name = dylib_name[ : 40 - 11 ] + encode_base62( dylib_name[ 40 - 11: ] )

    if dylib_name in _dylib_cache:
        return _dylib_cache[ dylib_name ]

    if not force_build():
        if res := _try_import( dylib_name ):
            _dylib_cache[ dylib_name ] = res
            return res

    # ── generate source ──────────────────────────────────────────────────────

    from ..generated_files import generated_files

    includes = [
        "sdot/nanobind_wrappers.h",
        "nanobind/stl/vector.h",
        "nanobind/stl/tuple.h",
        "nanobind/stl/string.h"
    ]
    for func in func_list:
        for inc in func.includes:
            if inc not in includes:
                includes.append( inc )

    code = ""
    for inc in includes:
        code += f"#include <{ inc }>\n"
    code += "\n"
    code += "namespace nb = nanobind;\n"
    code += "using namespace sdot;\n"
    code += "\n"
    code += f"using NbArch = nanobind::device::{ driver.normalized_device_type };\n"
    code += "using Arch = ArchFor<NbArch>::type;\n"
    code += f"using TF = { driver.normalized_dtype };\n"
    code += "using MI = std::optional<nb::ndarray<SI,NbArch>>;\n"
    code += "using MF = std::optional<nb::ndarray<TF,NbArch>>;\n"
    code += "\n"

    for func in func_list:
        arg_decl = []
        num_cpp_arg = 0
        for arg in func.args:
            for std_obj in to_standard_objects( arg ):
                arg_decl.append( f"{ cpp_class_name_for( std_obj ) } arg_{ num_cpp_arg }" )
                num_cpp_arg += 1

        # beg def
        code += f"auto _{ func.name }( { str.join( ', ', arg_decl ) } ) {{\n"

        # arguments
        arg_names = [ f"arg_{ n }" for n in range( num_cpp_arg ) ]
        for n, arg in enumerate( func.args ):
            code += f"    auto cpp_arg_{ n } = { from_standard_objects( arg, arg_names ) };\n"

        # call
        code += f"    return { func.name }( { str.join( ', ', [ f"cpp_arg_{ n }" for n in range( len( func.args ) ) ] ) } );\n"

        # end def
        code += "}\n"

    code += f"NB_MODULE( { dylib_name }, m ) {{\n"
    for func in func_list:
        code += f"    m.def( \"{ func.name }\", &_{ func.name } );\n"
    code += "}\n"

    ext = ".cu" if driver.normalized_device_type == "cuda" else ".cpp"
    src_path = generated_files.src_dir( dylib_name ) / ( "binding" + ext )
    src_path.write_text( code )

    _compile( dylib_name, [ src_path ], driver.normalized_device_type )

    importlib.invalidate_caches()
    res = importlib.import_module( _module_name( dylib_name ) )
    _dylib_cache[ dylib_name ] = res
    return res


def import_binding( dylib_name: str, src_func, src_paths: list, device_type: str ) -> any:
    """Import (compiling if necessary) a named binding module.

    Parameters
    ----------
    dylib_name:
        Stable identifier for the module, e.g. ``"poly_coeff_c1_2d"``.
    src_func:
        Callable that returns the C++ source text (called only when compilation
        is needed).  The text may contain ``SDOT_BINDING_NAME`` which is
        replaced with *dylib_name* before writing to disk.
    src_paths:
        Additional source files to compile alongside the generated binding.
    device_type:
        Normalised device type string, e.g. ``"cpu"`` or ``"cuda"``.
    """
    if dylib_name in _dylib_cache:
        return _dylib_cache[ dylib_name ]

    if not force_build():
        if res := _try_import( dylib_name ):
            _dylib_cache[ dylib_name ] = res
            return res

    from ..generated_files import generated_files

    ext = ".cu" if device_type == "cuda" else ".cpp"
    bnd_path = generated_files.src_dir( dylib_name ) / ( "binding" + ext )

    txt = src_func()
    txt = txt.replace( "SDOT_BINDING_NAME", dylib_name )
    bnd_path.write_text( txt )

    _compile( dylib_name, [ bnd_path ] + list( src_paths ), device_type )

    importlib.invalidate_caches()
    res = importlib.import_module( _module_name( dylib_name ) )
    _dylib_cache[ dylib_name ] = res
    return res


# ── internal compilation ─────────────────────────────────────────────────────

def _compile( dylib_name: str, src_paths: list, device_type: str ):
    """Invoke xmake to build *dylib_name* from *src_paths*."""
    import nanobind

    from ..generated_files import generated_files
    dylib_dir = generated_files.dylib_dir

    nanobind_include_dir  = nanobind.include_dir()
    nanobind_ext_include  = os.path.join( os.path.dirname( nanobind_include_dir ), "ext", "robin_map", "include" )
    nanobind_source       = os.path.join( nanobind.source_dir(), "nb_combined.cpp" )
    python_include        = sysconfig.get_path( "include" )
    project_root          = Path( __file__ ).absolute().parents[ 4 ]

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
        "SDOT_XMAKE_OUTPUT_DIR" : str( dylib_dir ),
        "SDOT_XMAKE_NEEDS_CUDA" : str( int( device_type.startswith( "cuda" ) ) ),
        "SDOT_XMAKE_REQUIRES"   : str.join( ",", [ "zpp_bits", "eigen" ] ),
        "SDOT_XMAKE_INCLUDES"   : str.join( ",", map( str, [
                                      nanobind_ext_include,
                                      nanobind_include_dir,
                                      python_include,
                                      project_root / "src" / "cpp",
                                  ] ) ),
        "SDOT_XMAKE_CXXFLAGS"   : str.join( ",", [ "-fdiagnostics-absolute-paths", "-fno-strict-aliasing" ] ),
        "SDOT_XMAKE_SOURCES"    : str.join( ",", map( str, list( src_paths ) + [ nanobind_source ] ) ),
        "SDOT_XMAKE_DEFINES"    : "",
        "SDOT_XMAKE_TARGET"     : dylib_name,
        "PATH"                  : str( Path( sys.executable ).parent ) + os.pathsep + os.environ.get( "PATH", "" ),
    }

    sdot_dir = Path( __file__ ).parent  # cpp_binding/ (contains xmake.lua)

    def run( cmd ):
        out = subprocess.run( cmd, cwd = dylib_dir, env = env )
        if out.returncode:
            sys.exit( out.returncode )

    run( [ xmake, "f", "-P", str( sdot_dir ), "-y", "--require=yes" ] )
    run( [ xmake, "-P", str( sdot_dir ) ] )
