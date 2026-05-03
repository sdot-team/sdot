from ..aggregate import Tensor, aggregate
from ..driver import driver
from .Distribution import Distribution
from typing import TYPE_CHECKING

# from itertools import product as iproduct
# from bisect import bisect_right
# from math import prod # , floor

@aggregate
class PolynomialGrid( Distribution ):
    """
    Q_k polynomial grid: coefficients follow the Q_k (tensor-product) basis,
    ordered by multi-index (p0, p1, ...) in lex order with each p_d in 0..order.

    Coefficient ordering example (order=1, dim=2):
      c[0]: 1   (p=(0,0))
      c[1]: y   (p=(0,1))
      c[2]: x   (p=(1,0))
      c[3]: xy  (p=(1,1))

    values : tensor[ x, y, z, ... , nb_coeffs ]
    frame  : tensor[ nb_vec, dim ]. Canonical frame by default.
    knots  : [ tensor[ num_point ] for axis in axes ]. linspace(0,1) by default.
    """

    values : Tensor( "shape( dim )", "nb_coeffs" )
    frame  : Tensor( "dim + 1", "dim" )
    knots  : Tensor( "nb_knots[ dim ]" ) # ListOfTensors( "dim", [ "shape[ index ]" ] )

    if TYPE_CHECKING:
        max_of_nb_knots : int  #
        nb_coeffs       : int  # (order+1)^dim
        shape           : list[ int ]

    @property
    def order( self ):
        n = self.nb_coeffs
        d = self.dim
        # Q_k: nb_coeffs == (order+1)^dim
        for k in range( 8 ):
            if ( k + 1 ) ** d == n:
                return k
        raise IndexError( f"nb_coeffs={n} is not (k+1)^{d} for any k in 0..7" )

    @staticmethod
    def nb_coeffs_for( dim, order ):
        return ( order + 1 ) ** dim

