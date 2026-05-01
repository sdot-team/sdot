# from sdot.aggregate._methods import aggregate
from .SplineGrid import SplineGrid

# @aggregate
class PiecewiseAffineGrid( SplineGrid ):
    """
    SplineGrid with order = 1
    """

    # def __init__( self, values = None, bounds_or_knots = None, knots = None, bounds = None, order = None ):
    #     SplineGrid.__init__( self, values = values, bounds_or_knots = bounds_or_knots, knots = knots, bounds = bounds, order = 1 )

    #     if order is not None:
    #         assert( order == 1 )


