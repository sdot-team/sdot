import numpy
import sdot

def for_each_driver_comb( cb ):
    for framework in [ "jax" ]: # "torch",
        sdot.driver.framework = framework
        print( sdot.driver )
        for dtype in [ "FP64", "FP32" ]: #
            for device in [ "cpu" ]: #, "cuda"
                sdot.driver.device = device
                sdot.driver.dtype = dtype
                cb()


def test_basic():
    g = sdot.SumOfWeightedDiracs( numpy.random.random( [ 5, 2 ] ) )
    f = sdot.PolynomialGrid( [ [ [ 1 ] ] ] )
    p = sdot.optimal_transport_plan( f, g )

    info( p.distance )
    p.plot()


if __name__ == "__main__":
    test_basic()
    print( "All good" )
