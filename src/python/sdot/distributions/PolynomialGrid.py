from sdot.distributions.helpers.distribution_methods import TensorField, ListOfTensorFields, generate_distribution_methods, Distribution, driver
from typing import TYPE_CHECKING


@generate_distribution_methods
class PolynomialGrid( Distribution ):
    """
    values : tensor[ x, y, z, ... ] (shape[ 0 ] -> size for the x direction, ...)
    frame  : tensor[ nb_vec, dim ]. [ [ 0, 0, ... ], [ 1, 0, ... ], [ 0, 1, ... ] ] by default. nb_vec must be equal to dim + 1.
    knots  : [ tensor[ num_point ] for axis in axes ],

    By default ``knots`` is a linspace( 0, 1, nb_points ) for each axis.
    """

    values    = TensorField( "shape * dim", "nb_coeffs" )
    frame     = TensorField( "dim + 1", "dim" )
    knots     = ListOfTensorFields( "dim", [ "shape[ index ]" ] )

    if TYPE_CHECKING:
        def __init__( self, values = None, frame = None, knots = None ): ...
        nb_coeffs : int # nb coeffs in each polynomial
        shape     : list[ int ] # for each dim

    @property
    def order( self ):
        n = self.nb_coeffs
        d = self.dim

        # order 0 ?
        if n <= 1:
            if n != 1:
                raise IndexError( "nb_coeffs error" )
            return 0

        # order 1 ?
        if n <= 1 + d:
            if n != 1 + d:
                raise IndexError( "nb_coeffs error" )
            return 1

        # order 2 ?
        if n <= 1 + d + d * ( d + 1 ) // 2:
            if n != 1 + d + d * ( d + 1 ) // 2:
                raise IndexError( "nb_coeffs error" )
            return 2

        raise NotImplementedError

    def primitive_function( self, batch_version: bool ):
        prim = f"{ "batch_of_" if batch_version else "" }polynomial_grid( CtInt<{ self.order }>(), g_values, g_frame, g_knots, true )"
        incl = [ f"sdot/{ "BatchOf" if batch_version else "" }PolynomialGrid.h" ]
        return prim, incl
