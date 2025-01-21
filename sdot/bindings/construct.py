from urllib.parse import unquote_plus
from pathlib import Path
from munch import Munch
import subprocess
import sysconfig
import platform
import pybind11
import sys
import os

boost_version = '1.86.0'
eigen_version = '3.4.0'
asimd_version = '0.0.5'
tl20_version = '0.0.7'

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

def download_and_unzip( link, ext_directory ):
    import dload
 
    print( f"Downloading { link } in { ext_directory }" )

    dload.save_unzip( link, str( ext_directory ), delete_after = True )
    #try:
    #    os.rename( ext_directory / src, ext_directory / dst )
    #except OSError:
    #    pass

# def git_clone( link ):
#     from git import Repo  # pip install gitpython

#     Repo.clone_from( ,  )

def construct( Environment, VariantDir, Configure, ARGLIST, name, used_arg_names, files, add_srcs_for_windows = [] ):
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
    br = boost_version.replace( '.', '_' )
    CPPPATH = [
        # src
        os.path.join( source_directory, 'cpp' ),

        # ext
        os.path.join( ext_directory, 'tl20', 'src', 'cpp' ),
        os.path.join( ext_directory, 'asimd', 'src' ),
        os.path.join( ext_directory, 'boost' ),
        os.path.join( ext_directory, 'eigen' ),

        os.path.join( ext_directory, f'boost_{ br }' ),
        os.path.join( ext_directory, f'tl20-{ tl20_version }', 'src', 'cpp' ),
        os.path.join( ext_directory, f'asimd-{ asimd_version }', 'src' ),
        os.path.join( ext_directory, f'eigen-{ eigen_version }' ),        

        # system
        sysconfig.get_paths()[ 'include' ], # Python.h
        pybind11.get_include(), # pybind11.h
    ]


    # CXXFLAGS
    CXXFLAGS = [
        f'-DSDOT_CONFIG_module_name={ module_name }',
        f'-DSDOT_CONFIG_suffix={ suffix }',

        # '-Wdeprecated-declarations',
        # '-fopenmp',

        # '-fdiagnostics-color=always',
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
        sysconfig.get_config_var( 'LIBDIR' ), # Python.h
    ]

    # LINKFLAGS
    LINKFLAGS = [
    ]

    if platform.system() == 'Darwin':
        LINKFLAGS += [ "-undefined", "dynamic_lookup" ]

    # .cpp files
    sources = files + [
    ]
    if os.name == 'nt':
        sources += add_srcs_for_windows
    sources = list( map( str, sources ) )

    # repl tl20 by tl20-version if present
    if not os.path.exists( os.path.join( ext_directory, 'tl20' ) ):
        sources = [ source.replace( 'tl20', f'tl20-{ tl20_version }' ) for source in sources ]

    # Environment
    env = Environment( 
        CPPPATH = CPPPATH, 
        CXXFLAGS = CXXFLAGS, 
        LIBS = LIBS, 
        LIBPATH = LIBPATH, 
        LINKFLAGS = LINKFLAGS, 
        SHLIBPREFIX = '',
        SHLIBSUFFIX = sysconfig.get_config_var( 'EXT_SUFFIX' ),
    )

    if env[ "CC" ] == 'cl':
        env.Append( CXXFLAGS = [
            '/Zc:zeroSizeArrayNew',
            '/std:c++20',
            '/EHsc',
            '/O2',
        ] )
    else:
        env.Append( CXXFLAGS = [
            '-ffast-math',
            '-std=c++20',
            '-O3',
            '-g3',
        ] )

    # check the libraries
    conf = Configure( env )
    if not conf.CheckCXXHeader( 'boost/multiprecision/cpp_int.hpp' ):
        br = boost_version.replace( '.', '_' )
        download_and_unzip( f"https://archives.boost.io/release/{ boost_version }/source/boost_{ br }.zip", ext_directory )
    if not conf.CheckCXXHeader( 'Eigen/Dense' ):
        download_and_unzip( f"https://gitlab.com/libeigen/eigen/-/archive/{ eigen_version }/eigen-{ eigen_version }.zip", ext_directory )
    if not conf.CheckCXXHeader( 'asimd/SimdVec.h' ):
        download_and_unzip( f"https://github.com/hleclerc/asimd/archive/refs/tags/v{ asimd_version }.zip", ext_directory )
    if not conf.CheckCXXHeader( 'tl/support/Displayer.h' ):
        download_and_unzip( f"https://github.com/hleclerc/tl20/archive/refs/tags/v{ tl20_version }.zip", ext_directory )
    env = conf.Finish()

    # register the library
    env.SharedLibrary( module_name, sources )
