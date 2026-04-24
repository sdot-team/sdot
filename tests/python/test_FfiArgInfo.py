# from sdot.drivers.compilation.FfiArgInfo import FfiArgInfo
import numpy
import sdot

def test_base():
    # fai = FfiArgInfo( { "a": 17, "b": numpy.array( [ 13 ] ) }, sdot.driver )
    # info( fai )

    sdot.driver.call( "test_alac", "sdot/cell/test_alac.h", b = numpy.array( [ 13 ] ), ret = sdot.Return( sdot.Tensor, [] ), _no_grad = True )

if __name__ == "__main__":
    test_base()
