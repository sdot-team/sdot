from urllib.parse import quote_plus
from pathlib import Path
from munch import Munch
import archspec.cpu
import subprocess
import platform
import numpy
import sys
import os

# needed to share the symbols
if os.name != 'nt':
    sys.setdlopenflags( os.RTLD_GLOBAL | os.RTLD_LAZY )

# ------------------------------------------------------------------------------------------------------------------------------------------
def get_global_directory_for( name ):
    """
      find a directory to 
        * copy .cpp/.h
        * install ext libraries
        * store .so/.dylib/... files
    """
    
    # Try in source/build
    sources = Path( __file__ ).parent.parent
    res = sources / name
    if can_use_dir( res ):
        return res
    
    # Else, try in $HOME/sdot_build
    local = Path.home() / ".local" / "lib" / "sdot"
    res = local / name
    if can_use_dir( res ):
        return res    
    
    raise RuntimeError( f"found no way to make a { name } dir (tried in { sources } and in { local }" )

def can_use_dir( dir ):
    try:
        os.makedirs( dir, exist_ok = True )
        return os.access( dir, os.W_OK )
    except OSError:
        pass
    return False

# ------------------------------------------------------------------------------------------------------------------------------------------
def push_activity_log( name ):
    activities.append( name )
    print( ' -> '.join( activities ), end = '' )
def pop_activity_log():
    s = ' -> '.join( activities )
    print( '\r' + ' ' * len( s ), end = '\r' )
    activities.pop()


def set_auto_rebuild( value ):
    global auto_rebuild
    auto_rebuild = value

def normalized_dtype( dtype, raise_exception_if_not_found = True ):
    if dtype in [ "FP64", "double", numpy.float64, numpy.int32, numpy.int64 ]:
        return "FP64"
    if dtype in [ "FP128", numpy.float128 ]:
        return "FP128"
    if dtype in [ "FP32", "float", numpy.float32 ]:
        return "FP32"
    if raise_exception_if_not_found:
        raise RuntimeError( f"unknown dtype `{ dtype }`" )
    return None


def numpy_dtype_for( dtype ):
    if dtype in [ "FP128" ]:
        return numpy.float128
    if dtype in [ "FP64" ]:
        return numpy.float64
    if dtype in [ "FP32" ]:
        return numpy.float32
    return None


def type_score( dtype: str ):
    if dtype.startswith( "FP" ):
        return int( dtype[ 2: ] )
    raise RuntimeError( "TODO" )


def type_promote( dtypes ):
    if len( dtypes ) == 0:
        return None

    res = normalized_dtype( dtypes[ 0 ] )
    for dtype in map( normalized_dtype, dtypes[ 1: ] ):
        if type_score( dtype ) > type_score( res ):
            res = dtype

    return res


def get_local_build_directory( global_build_directory, *kargs ):
    res = global_build_directory
    for arg in kargs:
        res = res / arg

    os.makedirs( res, exist_ok = True )
    if os.access( res, os.W_OK ):
        return res
    
    raise RuntimeError( f"failed to make the local build dir '{ res }'" )


def module_for( name, dir_of_the_SConstruct_file = Path( __file__ ).parent, use_arch = False, **kwargs ):
    # ensure that the std objects are loaded
    if name != 'generic_objects':
        module_for( 'generic_objects', dir_of_the_SConstruct_file, False )

    # sorted list of parameters
    plist = [ ( x, str( y ) ) for x, y in kwargs.items() ]
    plist.sort()

    # append 'os'
    plist.append( ( 'os', platform.system() ) )
    # if use_arch:
    #     plist.append( ( 'arch', archspec.cpu.generic_microarchitecture() ) )

    # other presentation for the parameters
    ilist = [ f"{ k }={ quote_plus( str( v ) ) }" for k, v in plist ]
    vlist = [ quote_plus( str( v ) ) for _, v in plist ]

    # names
    suffix = "_".join( vlist ).replace( "%", '_' )
    module_name = f'{ name }_bindings_for_{ suffix }'

    # in the cache
    if module_name in loader_cache:
        return loader_cache[ module_name ]

    # where to find/put the files
    local_build_directory = get_local_build_directory( global_build_directory, name, suffix )
    if local_build_directory not in sys.path:
        sys.path.insert( 0, str( local_build_directory ) )

    # if allowed, try to load directly the library
    module = None
    if not auto_rebuild:
        try:
            module = __import__( module_name )
        except ModuleNotFoundError:
            pass

    # else, build and load it
    if module is None:
        if global_verbosity_level:
            if len( kwargs ):
                push_activity_log( f"compilation of { name } for { kwargs }" )
            else:
                push_activity_log( f"compilation of { name }" )

        #
        source_directory = Path( __file__ ).parent

        args = [ 'scons', '-j8', f"--sconstruct={ dir_of_the_SConstruct_file / ( name + '.SConstruct' ) }", # '-s',
            f"source_directory={ source_directory }", 
            f"ext_directory={ global_ext_directory }", 
            f"module_name={ module_name }", 
            f"suffix={ suffix }",
        ] + ilist

        # print( " ".join( args ) )

        # call scons
        ret_code = subprocess.call( args, cwd = str( local_build_directory ), shell = False )
        
        if ret_code:
            raise RuntimeError( f"Failed to compile the C++ { name } binding" )

        if global_verbosity_level:
            pop_activity_log()

        # import
        print( '\n'.join(sys.path) )

        module = __import__( module_name )

    # change names
    res = Munch()
    for n, v in module.__dict__.items():
        if n.endswith( '_' + suffix ):
            n = n[ : -1 - len( suffix ) ]
        res[ n ] = v

    # store the module in the cache
    loader_cache[ module_name ] = res

    return res

# --------------------------------------- global variable ---------------------------------------
global_build_directory = get_global_directory_for( "build" )
global_ext_directory = get_global_directory_for( "ext" )
global_verbosity_level = 1
auto_rebuild = False
loader_cache = {}
activities = []
