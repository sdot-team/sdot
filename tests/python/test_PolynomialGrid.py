# import numpy
import sdot

# import jax
# jax.config.update( 'jax_platform_name', 'cpu' )

# numpy.random.seed( 0 )

def test_PolynomialGrid():
    grid = sdot.PolynomialGrid( values = [ [ [ 1, 2, 3 ] ] ] )
    info( grid.mass )
    # info( grid.dim )
    # image = image.normalized_version( mass = 1 )

    # info( image )
    # info( image._aggregate_items() )

    # info( image.dim )
    # info( image.shape )
    # info( image.nb_coeffs )

if __name__ == "__main__":
    test_PolynomialGrid()
