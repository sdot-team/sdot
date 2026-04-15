from sdot.object_with_tensors._methods import object_with_tensors
from .SplineGrid import SplineGrid

@object_with_tensors
class PiecewiseConstantGrid( SplineGrid ):
    """
    SplineGrid with order = 0
    """

    def __init__( self, values = None, knots = None, bounds = None, order = None ):
        SplineGrid.__init__( self, values = values, knots = knots, bounds = bounds, order = 0 )

        if order is not None:
            assert( order == 0 )


