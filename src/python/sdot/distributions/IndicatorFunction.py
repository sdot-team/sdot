from .Distribution import Distribution

class IndicatorFunction( Distribution ):
    """ function with value `coeff` when x is in `space_subset`

        `space_subset` can be for instance a MeshedSpaceSubset, a SquareSpaceSubset, ...
    """

    def __init__( self, space_subset, coeff = 1 ):
        """ 
        """
        self.space_subset = space_subset

    # def __repr__( self ):
    #     return self._cell.__repr__()

    def convex_boundaries( self, ndim = None ):
        return self.space_subset.convex_boundaries( ndim )
    
