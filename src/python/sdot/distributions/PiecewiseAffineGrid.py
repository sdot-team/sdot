from sdot.distributions.helpers.distribution_methods import generate_distribution_methods
from .SplineGrid import SplineGrid

@generate_distribution_methods
class PiecewiseAffineGrid( SplineGrid ):
    """
    SplineGrid with order = 1
    """

    def __init__( self, values = None, bounds_or_knots = None, knots = None, bounds = None ):
        SplineGrid.__init__( self, values = values, bounds_or_knots = bounds_or_knots, knots = knots, bounds = bounds, order = 1 )


