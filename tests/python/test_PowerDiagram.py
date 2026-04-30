import numpy
import sdot

def test_bsp():
    # positions = sdot.driver.t2( numpy.random.random( [ 50, 2 ] ) )
    # weights = sdot.driver.t1( [ 3, 4 ] )
    # pd = sdot.PowerDiagram( positions, weights )

    # pd.plot()
    from sdot.compilation.CallArg_MainList import CallArg_MainList
    args = {
        "out": sdot.driver.array( [ 1, 2 ] ),
        "ret": sdot.Return( sdot.Tensor, "dim" ),
    }
    axis_values = {
        "dim": 2
    }

    info( CallArg_MainList.factory( args, axis_values ) )

if __name__ == "__main__":
    test_bsp()
