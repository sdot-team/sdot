# from sdot.drivers.compilation.collect_attributes import collect_attributes
from sdot.object_with_tensors import object_with_tensors, TensorField
import numpy
import sdot

@object_with_tensors
class Pouet:
    positions = TensorField( "dim" )


def test_base():
    # fai = FfiArgInfo( { "a": 17, "b": numpy.array( [ 13 ] ) }, sdot.driver )
    # info( fai )

    res = sdot.driver.call( "test_alac", "sdot/cell/test_alac.h",
        b = numpy.array( [ 13 ] ),
        ret = sdot.Return( Pouet, dim = 3 ),
        inp = Pouet( [ 32 ] ),
        _no_grad = True
    )
    info( res )

if __name__ == "__main__":
    test_base()
