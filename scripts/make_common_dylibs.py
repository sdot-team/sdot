from sdot.generated_files.compilation_directories import dylib_dir
from pathlib import Path
import numpy
import sdot

def for_each_driver_comb( cb ):
    for framework in [ "jax" ]: # "torch",
        for dtype in [ "FP64", "FP32" ]: #
            for device in [ "cpu" ]: #, "cuda"
                sdot.driver.framework = framework
                sdot.driver.device = device
                sdot.driver.dtype = dtype
                cb()

def mklibs():
    g = sdot.SumOfWeightedDiracs( numpy.random.random( [ 5, 2 ] ) )
    f = sdot.PolynomialGrid( [ [ [ 1 ] ] ] )
    # print( g.positions.dtype )
    p = sdot.optimal_transport_plan( f, g )


#
for_each_driver_comb( mklibs )

# move the .so/.dll files
dst = Path( __file__ ).absolute().parents[ 1 ] / "src" / "python" / "sdot" / "generated_files"
src = dylib_dir()

for pattern in [ "*.so", "*.dll" ]:
    for f in dst.glob( pattern ):
        f.unlink()
    for f in src.glob( pattern ):
        f.rename( dst / f.name )

