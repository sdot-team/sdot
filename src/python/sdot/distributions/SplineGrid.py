from sdot.distributions.helpers.distribution_methods import TensorField, ListOfTensorFields, generate_distribution_methods, Distribution, driver
from .PolynomialGrid import PolynomialGrid
from typing import TYPE_CHECKING


@generate_distribution_methods
class SplineGrid( Distribution ):
    """
    values : tensor[ x, y, z, ... ] (shape[ 0 ] -> size for the x direction, ...)
    frame  : tensor[ nb_vec, dim ]. [ [ 0, 0, ... ], [ 1, 0, ... ], [ 0, 1, ... ] ] by default. nb_vec must be equal to dim + 1.
    knots  : [ tensor[ num_point ] for axis in axes ],

    By default ``knots`` is a linspace( 0, 1, nb_points ) for each axis.
    """

    values    = TensorField( "shape * dim" )
    frame     = TensorField( "dim + 1", "dim" )
    knots     = ListOfTensorFields( "dim", [ "shape[ index ]" ] )
    order     = 1

    if TYPE_CHECKING:
        def __init__( self, values = None, frame = None, knots = None ): ...
        shape : list[ int ] # corresponds to values.shape[ d ] (values.shape[ d + 1 ] in the Batch version)

    def normalized_version( self ):
        new_values = None
        if self.values is not None:
            if self.order == 0:
                new_values = self.values[ None, ... ]
            else:
                raise NotImplementedError

        return PolynomialGrid( new_values, self.frame, self.knots )
