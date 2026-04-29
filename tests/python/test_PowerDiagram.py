import numpy
import sdot

def test_bsp():
    # positions = sdot.driver.t2( numpy.random.random( [ 50, 2 ] ) )
    # weights = sdot.driver.t1( [ 3, 4 ] )
    # pd = sdot.PowerDiagram( positions, weights )

    # pd.plot()
    from sdot.aggregate.ShapeItem import ShapeItem

    info( ShapeItem( "2 * index + 1 < dim" ) )

if __name__ == "__main__":
    test_bsp()
