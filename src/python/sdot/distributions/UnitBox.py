# from .bindings.loader import sdot_module_for
from ..TransformationMatrix import TransformationMatrix
from .Distribution import Distribution
import numpy as np

class UnitBox( Distribution ):
    """ The mass (integral) of this box is prescribed and equal to 1 by default (hence the "Unit" in the name), even if the box is scaled.

        The default it is a [0,1]^ndim box, but one can apply an affine transformation to move it to another places.
    """

    def __init__( self, transformation = None, ndim = None, mass = 1 ):
        """ 
            `transformation` is a way to move the [0,1]^ndim basic box. It must be compatible with TransformationMatrix.
            `ndim` si optional (it can be useful if there aren't already other ways to find it in calling procedures)
            `mass` is the prescribed total mass (after the transformation)
        """

        self.transformation = TransformationMatrix( transformation )
        self.ndim = ndim
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

        return bnds, 1



