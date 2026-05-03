from ..aggregate import aggregate, Tensor, Return
from ..driver import driver
from .Distribution import Distribution
from typing import TYPE_CHECKING


@aggregate
class SplineGrid( Distribution ):
    """
    values      : tensor[ x, y, z, ... ]
    frame       : tensor[ nb_vec, dim ] — canonical frame by default
    knots       : [ tensor[ num_point ] for each axis ] — `range( shape( axis_size ) )` by default
    continuity  : int — minimum continuity order (default 0 = C0)

    continuity=0  →  Q_1 bilinear per cell, local Vandermonde solve
    continuity=1  →  natural cubic spline (C2), tensor-product 1D solve per axis
    """

    values      : Tensor( "shape( dim )" )
    frame       : Tensor( "dim + 1", "dim" )
    knots       : Tensor( "nb_knots[ dim ]" ) # TODO: ListOfTensors( "index < dim", [ "shape( index )" ] )
    continuity  : int

    if TYPE_CHECKING:
        shape : list[ int ]
        dim : int

    def __init__( self, values, frame = None, knots = None, continuity = 1 ):
        self.continuity = continuity
        self.values = values
        self.frame = frame
        self.knots = knots

    @property
    def polynomial_order( self ):
        """ polynomial order implied by the continuity requirement:
            C0 → Q_1 (degree 1),  C1 → cubic (degree 3),  Ck → degree 2k+1
        """
        return 2 * self.continuity + 1

    def normalized_version( self ):
        from .PolynomialGrid import PolynomialGrid
        if self.values is None:
            return PolynomialGrid( None, self.frame, self.knots )

        return driver.call( "polynomial_grid_from_spline_grid", "sdot/Distributions/polynomial_grid_from_spline_grid.h",
            polynomial_grid = Return( PolynomialGrid, max_of_nb_knots = self.max_of_nb_knots, shape = self.shape, dim = self.dim ),
            spline_grid = self,
            grad = False
        )
