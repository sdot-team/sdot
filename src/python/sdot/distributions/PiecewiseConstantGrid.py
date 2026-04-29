# from sdot.aggregate import aggregate
from .SplineGrid import SplineGrid

# @aggregate
class PiecewiseConstantGrid( SplineGrid ):
    """
    SplineGrid with order = 0
    """

    def __init__( self, values = None, knots = None, bounds = None, order = None ):
        SplineGrid.__init__( self, values = values, knots = knots, bounds = bounds, order = 0 )

        if order is not None:
            assert( order == 0 )


