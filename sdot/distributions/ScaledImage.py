# from .bindings.loader import sdot_module_for
from .Distribution import Distribution
import numpy as np
import itertools
import math

class ScaledImage( Distribution ):
    """ Piecewise (grid) polynomial interpolation.

        The last dimension in the array is the number of coefficients for each pixel/voxel/... For instance, for a 1D grid with piecewise affine function (i.e. with 2 coefficients for each pixel), one can use a `[n, 2]` array where `n` is the number of pixels.

        By default the grid is located on a [0,1]^ndim box, but it is possible to describe another origin and axes.

        There are static methods to ease the construction of ScaledImages. For instance
        * ScaledImage.interpolation enable to construct a function interpolation using a function of the coordinates
    """

    def __init__( self, array, origin = None, axes = None ):
        """ 
            `array` contains the coefficients of the polynomials. For instance, a piecewise contant 10x10 image with have a [10,10,1] shape
            
            by default, 
            * origin is `(0,)^ndim` (`ndim` is defined by array)
            * axes are `[ [1,0,...], [0,1,...], ... ]`
        """

        self.array = np.ascontiguousarray( array )

        if origin is None:
            origin = np.zeros( [ self.ndim ] )
        self.origin = np.ascontiguousarray( origin )
        
        if axes is None:
            axes = np.identity( self.ndim )
        self.axes = np.ascontiguousarray( axes )

        print( self.array, self.origin, self.axes )
    
    def boundary_split( self, ndim = None ):
        if ndim is None:
            if self.ndim is None:
                raise RuntimeError( 'found no way to guess the dimensionality' )
            ndim = self.ndim

        # TODO: include transformations
        bnds = []
        for d in range( ndim ):
            bnds.append( [ + ( i == d ) for i in range( ndim ) ] + [ 1 ] )
            bnds.append( [ - ( i == d ) for i in range( ndim ) ] + [ 0 ] )

        return bnds, self
    
    @property
    def ndim( self ):
        return self.array.ndim - 1

    @staticmethod
    def interpolation( min_pos, max_pos, nb_divs, func, degree = 0 ):
        """
            create a new Scaled image aligned with the axes

            min_pos, max_pos and nb_divs must be arrays of the same size, and they define the number of dimensions

            func is called with positions at the centers of the cells
        """
        ndim = len( min_pos )
        assert( ndim == len( max_pos ) )
        assert( ndim == len( nb_divs ) )

        shape = list( nb_divs ) + [ math.comb( ndim + degree, ndim ) ]

        lpos = []
        for ( mi, ma, nb ) in zip( min_pos, max_pos, nb_divs ):
            de = ( ma - mi ) / nb / 2
            lpos.append( np.linspace( mi + de, ma - de, nb, endpoint = True ) )

        array = []
        for pos in itertools.product( *lpos ):
            array.append( func( np.array( pos ) ) )
            if degree != 0:
                raise RuntimeError( "TODO: higher degrees for interpolation" )
        array = np.resize( array, shape )

        de = np.array( max_pos ) - np.array( min_pos )

        return ScaledImage( array, origin = min_pos, axes = de @ np.identity( ndim ) )


