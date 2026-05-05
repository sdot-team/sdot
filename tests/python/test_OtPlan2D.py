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
    # plan = sdot.opt(
    # )
    # info( plan.distance )
    g = sdot.SumOfWeightedDiracs( numpy.random.random( [ 5, 2 ] ), numpy.ones( 5 ) / 5 )
    f = sdot.PolynomialGrid( [ [ [ 1 ] ] ] )
    p = sdot.optimal_transport_plan( f, g )


if __name__ == "__main__":
    test_basic()
    print( "All good" )
