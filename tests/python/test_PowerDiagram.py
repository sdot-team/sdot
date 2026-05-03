import numpy
import sdot

def test_bsp():
    # n = 16
    # d = 2

    image = sdot.PolynomialGrid( values = [ [ [ 1,0,0,0 ], [ 2,0,0,0 ] ] ], knots = [], frame = [] )
    info( image.dim )
    info( image.shape )
    info( image.nb_coeffs )
    # info( image.max_of_nb_knots )

    # positions = numpy.random.random( [ n, d ] ) * 1.5
    # # positions = [ [ 10, 0 ], [ 11, 0 ] ]
    # # positions = [ [ 0, 0 ], [ 1, 0 ] ]
    # weights = numpy.full( [ n ], 0 )

    # pd = sdot.PowerDiagram( positions, weights )
    # pd.plot()

if __name__ == "__main__":
    test_bsp()
