import numpy
import sdot

def test_bsp():
    positions = sdot.driver.t2( numpy.random.random( [ 50, 2 ] ) )
    weights = sdot.driver.t1( [ 3, 4 ] )
    pd = sdot.PowerDiagram( positions, weights )

    pd.plot()

if __name__ == "__main__":
    test_bsp()
