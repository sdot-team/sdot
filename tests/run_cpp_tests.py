#!/usr/bin/env python3
"""Build & run the C++ tests via xmake, reusing the bindings' compilation toolchain.

  micromamba run -n vfs python tests/run_cpp_tests.py            # all tests
  micromamba run -n vfs python tests/run_cpp_tests.py Run Index  # only matching names

CPU tests always run. If a CUDA toolkit (nvcc) is present, the same sources are also
compiled with CUDA (so __CUDACC__ is defined and the GPU execution space is exercised);
they are only *run* when a GPU device is available, otherwise the run is skipped (the
build is still validated). Exit code is non-zero if any build or run failed.
"""
import subprocess
import shutil
import sys
from pathlib import Path

ROOT = Path( __file__ ).resolve().parents[ 1 ]
sys.path.insert( 0, str( ROOT / "src" / "python" ) )

from sdot.compilation.make_executable_from_files import make_executable_from_files
from sdot.drivers.Device import Device


def discover( filters ):
    files = sorted( ( ROOT / "tests" / "cpp" ).glob( "test_*.cpp" ) )
    if filters:
        files = [ f for f in files if any( k.lower() in f.stem.lower() for k in filters ) ]
    return files


def gpu_present():
    if not shutil.which( "nvidia-smi" ):
        return False
    r = subprocess.run( [ "nvidia-smi", "-L" ], capture_output = True, text = True )
    return r.returncode == 0 and "GPU" in r.stdout


def build_and_run( files, device, run ):
    label    = "cuda" if device.is_cuda_gpu else "cpu"
    failures = []
    for f in files:
        print( f"\n=== [{ label }] { f.stem } ===", flush = True )
        try:
            exe = make_executable_from_files( f"{ f.stem }_{ label }", [ f ], device )
        except Exception as e:
            print( f"  BUILD-FAIL: { e }" )
            failures.append( ( label, f.stem, "build" ) )
            continue
        if not run:
            print( "  BUILT (run skipped: no GPU device)" )
            continue
        if subprocess.run( [ str( exe ) ] ).returncode:
            failures.append( ( label, f.stem, "run" ) )
    return failures


def main():
    files = discover( sys.argv[ 1: ] )
    if not files:
        print( "no matching tests" )
        return 1

    failures = build_and_run( files, Device.factory( "cpu" ), run = True )

    if shutil.which( "nvcc" ):
        try:
            failures += build_and_run( files, Device.factory( "cuda" ), run = gpu_present() )
        except Exception as e:
            print( f"\n[cuda] skipped: { e }" )
    else:
        print( "\n[cuda] skipped: nvcc not found" )

    print( "\n================ summary ================" )
    if failures:
        for label, name, phase in failures:
            print( f"  [{ label }] { name }: { phase } FAILED" )
        return 1
    print( f"  all good ({ len( files ) } test(s))" )
    return 0


if __name__ == "__main__":
    sys.exit( main() )
