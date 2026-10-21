"""Where the wheels are built -- one environment per operating system:

    errand run --env mac_wheels,lin_wheels,win_wheels --pull dist make local_wheels      # or: make wheels

Each runs `make local_wheels` in its own copy of the tree ( the kernels of its frameworks, then the wheels ) and what
`--pull` names comes back here. Nothing in this file is particular to a machine: the hosts and paths are in
`errand-vars.py`, which is not versioned and is read first. With none, only `mac_wheels` is declared.

    # errand-vars.py
    linux_host    = "a machine with a GPU, Apptainer and a CUDA 13 driver"
    linux_root    = "a directory over there"       # the project is copied into it, loom beside it
    linux_binds   = [ "/data" ]                     # when that directory is a link to another disk
    linux_scratch = "a directory with room"         # to build the image
    linux_cache   = "a directory with room"         # the compilation cache
    mamba_root    = "~/.mamba"                      # where the `mac` environment goes
    win_host      = "an x86-64 machine with KVM and QEMU"   # where the Windows virtual machine boots
    win_root      = "a directory over there"
"""
import os
import errand

# `errand run` starts errand again inside each environment ( except a virtual machine ): the checkout this one comes from
# goes on the path there, pushed beside the project on another machine.
ERRAND = os.path.dirname( os.path.dirname( os.path.abspath( errand.__file__ ) ) )

# Kernels are compiled for every framework the catalogue covers: one interpreter that has them all.
FRAMEWORKS = "jax numpy torch"

# The Jax kernels declare the XLA FFI API version of the jaxlib they are compiled with, and a jaxlib refuses a NEWER one
# ( loom's `compilation/catalogue.py` then compiles them instead ): the wheels are built with a jaxlib old enough for
# about everyone. 0.10.2 = API 0.3, which jaxlib 0.10 to 0.11.0 have and any later one accepts. Only that check depends
# on it: the kernels' code does not. The same pin is in scripts/manylinux.def.
JAX = "jax==0.10.2"

errand.envs[ "mac_wheels" ] = errand.Env( [
    # The derivative kernels need P2280 ( unknown references in constant expressions ), which Apple's clang does not have.
    errand.Micromamba( "mac_wheels", python = "3.13", channels = [ "conda-forge" ], packages = [ "clangxx" ],
                       root_prefix = errand.value( "mamba_root" ),
                       pip = [ JAX, "numpy", "scipy", "torch", "pyyaml", "devtools", "typing_extensions", "ninja",
                               "pytest", "wheel", "twine" ] ),
    errand.Vars( { "LOOM_CXX": "clang++", "LOOM_CXXFLAGS": "-std=c++23", "FRAMEWORKS": FRAMEWORKS, "PYTHONPATH": ERRAND } ),
] )

host = errand.value( "linux_host" )
if host:
    # manylinux_2_28 ( glibc 2.28 ) with clang, GCC 14 for nvcc, and the frameworks: scripts/manylinux.def
    errand.envs[ "lin_wheels" ] = errand.Env( [
        errand.Ssh( host = host, root = errand.value( "linux_root", "errand/sdot-ffi" ), also = [ "../loom", "../errand" ] ),
        errand.Apptainer( image = "manylinux.sif", recipe = "scripts/manylinux.def", fakeroot = True,
                          scratch = errand.value( "linux_scratch" ),
                          flags = [ "--nv", *[ f for b in errand.value( "linux_binds", [ ] ) for f in ( "--bind", b ) ] ] ),
        errand.Vars( { "FRAMEWORKS": FRAMEWORKS + " cupy", "PYTHONPATH": "../errand",
                       **( { "LOOM_CACHE_DIR": errand.value( "linux_cache" ) } if errand.value( "linux_cache" ) else { } ) } ),
    ], cuda = True )

win_host = errand.value( "win_host" )
if win_host:
    # Windows Server under QEMU, booted on `win_host`. clang ( llvm-mingw ) builds the kernels as DLLs with the C++ runtime
    # linked in, and GNU make drives the build: both installed by scripts/win_tools.ps1 when the image is made.
    TOOLS = r"C:\tools"
    errand.envs[ "win_wheels" ] = errand.Env( [
        errand.Ssh( host = win_host, root = errand.value( "win_root", "errand/sdot-ffi" ), also = [ "../loom" ] ),
        errand.Windows( python = "3.13.16", cpus = 8, ram = "16G", also = [ "../loom" ],
                        provision = [ "& 'C:\\Program Files\\Python313\\python.exe' -m pip install --no-warn-script-location "
                                      f"numpy scipy pyyaml devtools typing_extensions ninja pytest wheel twine certifi { JAX } torch",
                                      "scripts/win_tools.ps1" ] ),
        errand.Vars( { "PATH": TOOLS + r"\llvm-mingw-20261006-ucrt-x86_64\bin;" + TOOLS + r"\mingw64\bin;%PATH%",
                       "LOOM_CXX": "clang++", "LOOM_LDFLAGS": "-static",
                       # Eigen 3.4.0 does not compile its AVX-512 kernels with this clang ( pfirst on __m128 ): no vectorization there
                       "LOOM_CXXFLAGS": "-std=c++23 -DEIGEN_DONT_VECTORIZE",
                       "LOOM_CACHE_DIR": r"E:\loomcache",     # the disk that is kept: C: starts clean at every boot
                       "SSL_CERT_FILE": r"C:\Program Files\Python313\Lib\site-packages\certifi\cacert.pem",
                       "FRAMEWORKS": FRAMEWORKS } ),
    ] )
