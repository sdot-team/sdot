import numpy as np
import ot

# from ot.datasets import make_1D_gauss as gauss
# import ot.plot
n = 10000  # nb bins
e = 1e-8

def dist_dirac( positions ):
    d = int( n / len( positions ) )
    r = []
    for x in positions:
        for _ in range( d ):
            r.append( x )
    return np.array( r, dtype = np.float64 )

def cost( x, y, a = None, b = None ):
    # loss matrix
    M = ot.dist( x.reshape((n, 1)), y.reshape((n, 1)), 'sqeuclidean' )

    # weights
    if a is None:
        a = np.ones( ( n, ) ) / n
    if b is None:
        b = np.ones( ( n, ) ) / n
    a /= np.sum( a )
    b /= np.sum( b )

    # use fast 1D solver
    return ot.solve( M, a, b )

print( " 1 à 0.0 vers 1", cost( dist_dirac( [ 1.0 ] ), np.linspace( 0, 1, n ) ) )
print( " 1 à 0.5 vers 1", cost( dist_dirac( [ 0.5 ] ), np.linspace( 0, 1, n ) ) )
print( " 1 à 0.5 vers x", cost( dist_dirac( [ 0.5 ] ), np.linspace( 0, 1, n ), b = np.linspace( 0, 1, n ) ) )

dist_10 = np.linspace( 0.05, 0.95, 10 )
print( "10 lsp vers 1", cost( dist_dirac( dist_10 ), np.linspace( 0, 1, n ) ) )
print( "10 lsp vers x", cost( dist_dirac( dist_10 ), np.linspace( 0, 1, n ), b = np.linspace( 0, 1, n ) ) )
