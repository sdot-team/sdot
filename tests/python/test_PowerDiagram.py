import numpy
import sdot

import jax
jax.config.update( 'jax_platform_name', 'cpu' )

numpy.random.seed( 0 )

def test_bsp():
    n = 100
    d = 2

    image = sdot.PolynomialGrid( values = [ [ [ 1 ] ] ] )
    image = image.normalized_version( mass = 1 )

    positions = numpy.random.random( [ n, d ] ) * 0.01 + 10 # + numpy.array( [ 0.5, 0 ] )
    # # positions = [ [ 10, 0 ], [ 11, 0 ] ]
    # # positions = [ [ 0, 0 ], [ 1, 0 ] ]
    weights = numpy.full( [ n ], 0 )

    pd = sdot.PowerDiagram( positions, weights )
    pd.adjust_weights( dirac_masses = numpy.full( [ n ], 1 / n ), target_distribution = image )
    pd.plot( target_distribution = image )

if __name__ == "__main__":
    test_bsp()
