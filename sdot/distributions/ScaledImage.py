# from .bindings.loader import sdot_module_for
from ..TransformationMatrix import TransformationMatrix
from .Distribution import Distribution
import numpy as np

class ScaledImage( Distribution ):
    """ The mass (integral) of this function is prescribed and equal to 1 by default (can be changed via the `mass` parameter)

        By default it is on a [0,1]^ndim box, but one can apply an affine transformation to move it to another places.
    """

    def __init__( self, array, transformation = None, mass = 1 ):
        """ 
            `transformation` is a way to move the [0,1]^ndim basic box. It must be compatible with TransformationMatrix.
            `mass` is the prescribed total mass (after the transformation)
        """

        self.transformation = TransformationMatrix( transformation )
        self.array = np.ascontiguousarray( array )
        self.mass = mass

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
        return self.array.ndim


