# from .bindings.loader import sdot_module_for
# import numpy as np

class IndicatorFunction:
    """ function with value `coeff` when x is in `space_subset`

        `space_subset` can be for instance a MeshedSpaceSubset, a SquareSpaceSubset, ...
    """

    def __init__( self, space_subset, coeff = 1 ):
        """ 
        """
        self.space_subset = space_subset

    # def __repr__( self ):
    #     return self._cell.__repr__()

