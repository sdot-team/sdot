import numpy
import sdot

def for_each_driver_comb( cb ):
    for framework in [ "torch", "jax" ]: #
        for dtype in [ "FP64", "FP32" ]: #
            for device in [ "cpu" ]: #, "cuda"
                sdot.driver.framework = framework
                sdot.driver.device = device
                sdot.driver.dtype = dtype
                cb()


def test_basic():
    plan = sdot.OtPlan(
        sdot.SumOfWeightedDiracs( numpy.random.random( [ 50, 2 ] ) ),
        sdot.PiecewiseConstantGrid( [ [ 1 ] ] ),
    )
    info( plan.distance )


if __name__ == "__main__":
    test_basic()
    print( "All good" )
