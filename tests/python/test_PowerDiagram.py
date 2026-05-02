import numpy
import sdot

def test_bsp():
    n = 20
    d = 3

    positions = numpy.random.random( [ n, d ] ) * 0.5
    weights = numpy.full( [ n ], 0 )

    pd = sdot.PowerDiagram( positions, weights )
    pd.plot()

if __name__ == "__main__":
    test_bsp()
