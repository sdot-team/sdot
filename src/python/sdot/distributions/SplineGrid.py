from sdot.distributions.helpers.distribution_methods import TensorField, ListOfTensorFields, generate_distribution_methods, Distribution, driver


@generate_distribution_methods
class SplineGrid( Distribution ):
    """
    values : tensor[ x, ..., z ]
    bounds : tensor[ num_bound, dim ]. [ [ 0, 0, ... ], [ 1, 0, ... ], [ 0, 1, ... ] ] by default. nb_bounds must be equal to dim + 1.
    knots  : [ tensor[ num_point ] for axis in axes ],

    If ``knots`` is defined, there's no need to define ``bounds`` (because ``bounds`` correspondes to extrema of knots).

    By default ``knots`` is a linspace between bounds for each axis.

    If ``bounds`` and ``knots`` are defined, ``knots`` are "normalized" to fit into the bounds (it's a way to define a rotated/skewed grid for instance)

    Of course
    """

    values    = TensorField( "shape * dim" )
    bounds    = TensorField( "dim + 1", "dim" )
    knots     = ListOfTensorFields( "dim", [ "shape[ index ]" ] )
    order     = 1

    nb_points : list[ int ] # for each dim
    nb_bounds : int
    shape     : list[ int ]

    def __init__( self, values = None, bounds_or_knots = None, knots = None, bounds = None, order = 1 ):
        if bounds_or_knots is not None:
            assert bounds_or_knots.ndim == 2
            assert values is not None
            dim = values.ndim
            if bounds_or_knots.shape[ 0 ] == dim + 1:
                assert bounds is None
                bounds = bounds_or_knots
            else:
                assert knots is None
                knots = bounds_or_knots

        self.values = values
        self.bounds = bounds
        self.knots = knots
        self.order = order


