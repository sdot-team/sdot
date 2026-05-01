import numpy
import sdot

def test_bsp():
    positions = numpy.random.random( [ 50, 2 ] )
    weights = [ 3, 4 ]
    pd = sdot.PowerDiagram( positions, weights )

    pd.plot()

if __name__ == "__main__":
    test_bsp()
