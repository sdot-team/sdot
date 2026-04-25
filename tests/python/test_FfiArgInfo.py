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

    pouet = Pouet( [ 53 ] )
    res = sdot.driver.call( "test_alac", "sdot/cell/test_alac.h",
        b = numpy.array( [ 13 ] ),
        ret = sdot.Mutable( pouet ),
        no_grad = True
    )
    info( pouet )
    info( res )

if __name__ == "__main__":
    test_base()
