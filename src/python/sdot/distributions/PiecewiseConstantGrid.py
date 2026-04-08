from sdot.distributions.helpers.distribution_methods import generate_distribution_methods
from .SplineGrid import SplineGrid

@generate_distribution_methods
class PiecewiseConstantGrid( SplineGrid ):
    """
    SplineGrid with order = 0
    """

    def __init__( self, values = None, knots = None, bounds = None, order = None ):
        SplineGrid.__init__( self, values = values, knots = knots, bounds = bounds, order = 0 )

        if order is not None:
            assert( order == 0 )


