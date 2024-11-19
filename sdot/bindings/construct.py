from urllib.parse import unquote_plus
from pathlib import Path
from munch import Munch
import subprocess
import sysconfig
import pybind11
import sys
import os

# helper to get n-th parent directory
def pdir( dir, n = 1 ):
    if n == 0:
        return dir
    return pdir( os.path.dirname( dir ), n - 1 )

def args_to_obj( ARGLIST ):
    args = Munch()
    for k, v in ARGLIST:
        if k in [ 'module_name', 'suffix' ]:
            args[ k ] = v
        else:
            args[ k ] = unquote_plus( v )
    return args

def download_and_unzip( link, src, dst, ext_directory ):
    import dload
 
    print( f"Downloading { dst } in { ext_directory }" )

    dload.save_unzip( link, str( ext_directory ), delete_after = True )
    try:
        os.rename( ext_directory / src, ext_directory / dst )
    except OSError:
        pass

# def git_clone( link ):
#     from git import Repo  # pip install gitpython

#     Repo.clone_from( ,  )

def construct( Environment, VariantDir, Configure, ARGLIST, name, used_arg_names, files ):
    # common args
    args = {}
    for k, v in ARGLIST:
        if k not in [ 'ext_directory', 'source_directory', 'module_name', 'suffix' ]:
            args[ k ] = unquote_plus( v )
        else:
            args[ k ] = v

    source_directory = Path( args[ 'source_directory' ] )
    ext_directory = Path( args[ 'ext_directory' ] )
    module_name = args[ 'module_name' ]
    suffix = args[ 'suffix' ]

    # build directory
    VariantDir( 'build', source_directory, duplicate=False )
    VariantDir( 'ext', ext_directory, duplicate=False )

    # includes
    CPPPATH = [
        # src
        os.path.join( source_directory, 'cpp' ),

        # ext
        os.path.join( ext_directory, 'tl20', 'src', 'cpp' ),
        os.path.join( ext_directory, 'asimd', 'src' ),
        os.path.join( ext_directory, 'boost' ),
        os.path.join( ext_directory, 'eigen' ),

        # system
        sysconfig.get_paths()[ 'include' ], # Python.h
        pybind11.get_include(), # pybind11.h
    ]


    # CXXFLAGS
    CXXFLAGS = [
        f'-DSDOT_CONFIG_module_name={ module_name }',
        f'-DSDOT_CONFIG_suffix={ suffix }',

        '-Wdeprecated-declarations',
        '-std=c++20',
        '-fopenmp',

        '-fdiagnostics-color=always',
        
        '-ffast-math',
        '-O3',

        '-g3',
    ]

    if 'arch' in used_arg_names:
       CXXFLAGS.append( '-march=' + args[ 'arch' ].replace( '_', '-' ) )

    for name in used_arg_names:
        CXXFLAGS.append( f'"-DSDOT_CONFIG_{ name }={ args[ name ] }"' )

    # LIBS
    LIBS = [
        # 'Catch2Main',
        # 'Catch2',
        # 'gomp',
    ]

    # LIBPATH
    LIBPATH = [
        # '/home/hugo.leclerc/.vfs_build/ext/Catch2/install/lib',
        # '/home/leclerc/.vfs_build/ext/Catch2/install/lib'
    ]

    # .cpp files
    sources = files + [
    ]

    # Environment
    env = Environment( CPPPATH = CPPPATH, CXXFLAGS = CXXFLAGS, LIBS = LIBS, LIBPATH = LIBPATH, SHLIBPREFIX = '' )

    # check the libraries
    conf = Configure( env )
    if not conf.CheckCXXHeader( 'boost/multiprecision/cpp_int.hpp' ):
        download_and_unzip( "https://archives.boost.io/release/1.86.0/source/boost_1_86_0.zip", "boost_1_86_0", "boost", ext_directory )
    if not conf.CheckCXXHeader( 'Eigen/Dense' ):
        download_and_unzip( "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip", "eigen-3.4.0", "eigen", ext_directory )
    if not conf.CheckCXXHeader( 'asimd/SimdVec.h' ):
        download_and_unzip( "https://github.com/hleclerc/asimd/archive/refs/tags/asimd-v0.0.1-alpha.zip", "asimd-asimd-v0.0.1-alpha", "asimd", ext_directory )
    if not conf.CheckCXXHeader( 'tl/support/Displayer.h' ):
        download_and_unzip( "https://github.com/hleclerc/tl20/archive/refs/tags/v0.0.3.zip", "tl20-0.0.3", "tl20", ext_directory )
    env = conf.Finish()

    # register the library
    env.SharedLibrary( module_name + env[ 'SHLIBSUFFIX' ], list( map( str, sources ) ) )
