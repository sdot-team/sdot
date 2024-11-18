from urllib.parse import unquote_plus
from munch import Munch
import sysconfig
import pybind11
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

def construct( Environment, VariantDir, ARGLIST, name, used_arg_names, files ):
    # build directory
    cwd = os.getcwd()
    bad = pdir( cwd, 3 )

    VariantDir( 'build', bad )

    # common args
    args = {}
    for k, v in ARGLIST:
        if k in [ 'module_name', 'suffix' ]:
            args[ k ] = v
        else:
            args[ k ] = unquote_plus( v )

    module_name = args[ 'module_name' ]
    suffix = args[ 'suffix' ]

    # scalar_type = args[ 'scalar_type' ]
    # nb_dims = args[ 'nb_dims' ]
    # arch = args[ 'arch' ]

    # includes
    CPPPATH = [
        # src
        os.path.join( bad, 'src', 'cpp' ),

        # ext
        os.path.join( bad, 'ext', 'tl20', 'src', 'cpp' ),
        os.path.join( bad, 'ext', 'Catch2', 'src' ),
        os.path.join( bad, 'ext', 'asimd', 'src' ),
        os.path.join( bad, 'ext', 'eigen' ),

        # systelm
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
        "build/ext/tl20/src/cpp/tl/support/display/DisplayItem_Pointer.cpp",
        "build/ext/tl20/src/cpp/tl/support/display/DisplayItem_Number.cpp",
        "build/ext/tl20/src/cpp/tl/support/display/DisplayItem_String.cpp",
        "build/ext/tl20/src/cpp/tl/support/display/DisplayItem_List.cpp",

        "build/ext/tl20/src/cpp/tl/support/display/DisplayParameters.cpp",
        "build/ext/tl20/src/cpp/tl/support/display/DisplayContext.cpp",
        "build/ext/tl20/src/cpp/tl/support/display/DisplayItem.cpp",

        "build/ext/tl20/src/cpp/tl/support/string/CompactReprWriter.cpp",
        "build/ext/tl20/src/cpp/tl/support/string/CompactReprReader.cpp",
        "build/ext/tl20/src/cpp/tl/support/string/read_arg_name.cpp",
        "build/ext/tl20/src/cpp/tl/support/string/va_string.cpp",
        "build/ext/tl20/src/cpp/tl/support/Displayer.cpp",

        'build/src/cpp/sdot/support/BigRational.cpp',
        "build/src/cpp/sdot/support/VtkOutput.cpp",
        "build/src/cpp/sdot/support/Mpi.cpp",

        'build/src/cpp/sdot/symbolic/Symbol.cpp',
        'build/src/cpp/sdot/symbolic/Value.cpp',
        'build/src/cpp/sdot/symbolic/Expr.cpp',
        'build/src/cpp/sdot/symbolic/Func.cpp',
        'build/src/cpp/sdot/symbolic/Inst.cpp',
    ]


    # make the library
    env = Environment( CPPPATH = CPPPATH, CXXFLAGS = CXXFLAGS, LIBS = LIBS, LIBPATH = LIBPATH, SHLIBPREFIX = '' )
    env.SharedLibrary( module_name + env[ 'SHLIBSUFFIX' ], sources )
