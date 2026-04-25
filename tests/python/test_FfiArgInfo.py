# from sdot.drivers.compilation.collect_attributes import collect_attributes
# from sdot.drivers.compilation.FfiArgInfo import FfiArgInfo
from sdot.object_with_tensors import object_with_tensors, TensorField
import numpy
import sdot

@object_with_tensors
class Pouet:
    positions = TensorField( "dim" )


def test_base():
    # Rq: ça serait bien que la backward version construise les entrées
    # fai = FfiArgInfo( {
    #      "a": sdot.Return( Pouet, dim = 2 ),
    #      "b": sdot.Return( sdot.Tensor, [ 10, 20 ] ),
    # }, sdot.driver ) # , "b": numpy.array( [ 13 ] )

    # info( fai.backward_version( sdot.driver ) )

    # pouet = Pouet( [ 53 ] )
    # res = sdot.driver.call( "test_alac", "sdot/cell/test_alac.h",
    #     b = numpy.array( [ 13 ] ),
    #     ret = sdot.Mutable( pouet ),
    # )
    # info( pouet )
    # info( res )

    def f( x ):
        return sdot.driver.call( "test_alac", "sdot/cell/test_alac.h",
            o0 = sdot.Return( sdot.Tensor, [] ),
            o1 = sdot.Return( sdot.Tensor, [] ),
            i0 = x,
            i1 = x,
        )[ 0 ]

    from jax._src import test_util as jtu

    x = sdot.driver.t0( 3.0 )
    jtu.check_grads( f, ( x, ), order = 1, modes = [ "rev" ] )

if __name__ == "__main__":
    test_base()
