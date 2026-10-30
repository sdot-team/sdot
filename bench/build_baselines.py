"""THE PROGRAMS THE COMPARISONS RUN, BUILT ON THE MACHINE: CGAL's ( `bench_cgal.py` ) and geogram's ( `bench_geogram.py` ).

    errand run --env n22-numpy -x sdot-ffi/bench/build_baselines.py              # what is missing or out of date
    errand run --env n22-numpy -x sdot-ffi/bench/build_baselines.py force=1      # rebuild both
    errand run --env n22-numpy -x sdot-ffi/bench/build_baselines.py only=cgal cgal_ref=v6.1
    errand run --env lmo-numpy -x sdot-ffi/bench/build_baselines.py cache=/tmp/trial force=1   # a trial, elsewhere

`page.py` runs it before the jobs that need them. Linux only ( geogram's platform is `Linux64-gcc` ).

A baseline is UP TO DATE when the first directory its bench looks in that holds its programs ( `bench/<x>/`,
`~/.cache/sdot/<x>-bench`, then `/data/<x>-bench` ) has a `SOURCES` file equal to the hash of what they are built from ( the drivers, `build.sh` ): changing a
driver rebuilds it. Otherwise it is built into `~/.cache/sdot/<x>-bench` by its `build.sh`, the sources cloned next to it
( CGAL's main, geogram's and exploragram's default branches ).

The libraries ( gmp, mpfr, boost's headers, TBB ) come from a micromamba environment of their own, `~/.cache/sdot/baselines-env`,
created at the first build ( conda-forge ), with cmake / make / git / a compiler when the machine has none.

Without internet ( the image `containers/jax-cuda.def` ): `SDOT_BASELINES_LIBS` is a prefix that already holds the libraries,
`SDOT_BASELINES_SRC` a directory that already holds the clones ( `cgal/`, `geogram-src/` ), built as they are. `SDOT_CACHE`
replaces `~/.cache/sdot` ( where $HOME is small ).
"""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE  = Path( __file__ ).resolve().parent
CACHE = Path( os.path.expanduser( os.environ.get( "SDOT_CACHE", "~/.cache/sdot" ) ) )

#: name -> ( the programs, what they are built from, relative to HERE )
BASELINES = {
    "cgal":    ( [ "power_2d", "power_3d" ], [ "cgal/build.sh", "cgal/power_2d.cpp", "cgal/power_3d.cpp", "cgal/nsdot-src" ] ),
    "geogram": ( [ "geogram_ot" ],           [ "geogram/build.sh", "geogram/geogram_ot.cpp" ] ),
}


def sources_hash( paths ):
    h = hashlib.sha256()
    for rel in paths:
        p = HERE / rel
        for f in sorted( p.rglob( "*" ) ) if p.is_dir() else [ p ]:
            if f.is_file():
                h.update( str( f.relative_to( HERE ) ).encode() + b"\0" + f.read_bytes() )
    return h.hexdigest()[ :16 ]


def up_to_date( name ):
    progs, srcs = BASELINES[ name ]
    want = sources_hash( srcs )
    for d in ( HERE / name, CACHE / f"{ name }-bench", Path( f"/data/{ name }-bench" ) ):     # the bench's order ( `bench_<x>.py` )
        if all( os.access( d / p, os.X_OK ) for p in progs ):        # the one the bench will run
            stamp = d / "SOURCES"
            return d if stamp.exists() and stamp.read_text().strip() == want else None
    return None


def libraries_env():
    """the prefix holding gmp / mpfr / boost / TBB ( and the tools the machine lacks ), created once"""
    if given := os.environ.get( "SDOT_BASELINES_LIBS" ):
        return Path( given )
    prefix = CACHE / "baselines-env"
    if ( prefix / "include" / "gmp.h" ).exists():
        return prefix
    mamba = os.environ.get( "MAMBA_EXE" ) or shutil.which( "micromamba" )
    if not mamba:
        sys.exit( "build_baselines: no micromamba ( MAMBA_EXE or on PATH ) to install gmp / mpfr / boost / TBB" )
    pkgs = [ "gmp", "mpfr", "libboost-headers", "tbb-devel" ]
    pkgs += [ t for t in ( "cmake", "make", "git" ) if not shutil.which( t ) ]
    if not shutil.which( "g++" ):
        pkgs += [ "gxx", "gcc" ]
    print( f"== the libraries: { prefix } ( { ' '.join( pkgs ) } )", flush = True )
    subprocess.run( [ mamba, "create", "-y", "-q", "-p", str( prefix ), "-c", "conda-forge", *pkgs ], check = True )
    return prefix


def build( name, libs, args ):
    progs, srcs = BASELINES[ name ]
    out = CACHE / f"{ name }-bench"
    env = dict( os.environ, LIBS = str( libs ), PATH = f"{ os.environ[ 'PATH' ] }:{ libs }/bin" )
    if not shutil.which( "g++" ):                         # the compiler of the env ( conda's names it after the target )
        cxx = next( iter( sorted( ( libs / "bin" ).glob( "*-g++" ) ) ), None )
        cc  = next( iter( sorted( ( libs / "bin" ).glob( "*-gcc" ) ) ), None )
        env.update( CXX = str( cxx ), CC = str( cc ) )
    src = Path( os.environ.get( "SDOT_BASELINES_SRC", CACHE ) )   # the clones, given ( offline ) or made in the cache
    if name == "cgal":
        env.update( CGAL_DIR = str( src / "cgal" ), CGAL_REF = args.get( "cgal_ref", "main" ) )
    else:
        env.update( GEOGRAM_SRC = str( src / "geogram-src" ) )
    print( f"== { name }: building into { out }", flush = True )
    subprocess.run( [ "bash", str( HERE / name / "build.sh" ), str( out ) ], env = env, check = True )
    ( out / "SOURCES" ).write_text( sources_hash( srcs ) + "\n" )


def main():
    global CACHE
    args = dict( a.split( "=", 1 ) for a in sys.argv[ 1: ] if "=" in a )
    if args.get( "cache" ):                               # another place than ~/.cache/sdot ( a trial build )
        CACHE = Path( os.path.expanduser( args[ "cache" ] ) )
    names = [ n for n in args.get( "only", ",".join( BASELINES ) ).split( "," ) if n ]
    if bad := [ n for n in names if n not in BASELINES ]:
        sys.exit( f"build_baselines: unknown { bad } ( { ', '.join( BASELINES ) } )" )
    todo = [ ]
    for n in names:
        d = None if args.get( "force" ) == "1" else up_to_date( n )
        print( f"== { n }: { f'up to date in { d }' if d else 'to build' }", flush = True )
        if d is None:
            todo.append( n )
    if todo:
        libs = libraries_env()
        for n in todo:
            build( n, libs, args )


if __name__ == "__main__":
    main()
