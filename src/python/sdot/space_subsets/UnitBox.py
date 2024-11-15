# from .bindings.loader import sdot_module_for
from .SpaceSubset import SpaceSubset
import numpy as np

class UnitBox( SpaceSubset ):
    """ 
    """

    def __init__( self, ndim = None ):
        """ 
            `ndim` si optional (can be useful if there aren't already other ways to find it in calling procedures)
        """

        self.ndim = ndim

    def convex_boundaries( self, ndim ):
        if ndim is None:
            if self.ndim is None:
                return None
            ndim = self.ndim

        res = []
        for d in range( ndim ):
            res.append( [ + ( i == d ) for i in range( ndim ) ] + [ 1 ] )
            res.append( [ - ( i == d ) for i in range( ndim ) ] + [ 0 ] )
        return res
    
    def indicator_binding( self, base_cell, binding_module ):
        # we can use a constant value, with modified boundaries
        ndim = binding_module.ndim()
        for d in range( ndim ):
            base_cell.cut( [ + ( i == d ) for i in range( ndim ) ], 1 )
            base_cell.cut( [ - ( i == d ) for i in range( ndim ) ], 0 )
            
        return binding_module.ConstantValue( 1 )


