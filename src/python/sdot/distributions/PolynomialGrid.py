from ..drivers.driver import driver, FfiCodeParallel
from ..aggregate import Tensor, Return, aggregate
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

    values : Tensor( "shape( dim )", "nb_coeffs", ct_variables = [ "nb_coeffs" ] )
    frame  : Tensor( "dim + 1", "dim" )
    knots  : Tensor( "dim", "nb_knots[ dim ]" ) # ListOfTensors( "dim", [ "shape[ index ]" ] )

    if TYPE_CHECKING:
        def __init__( self, values = None, frame = None, knots = None ): ...

        batch_axes_dict : dict[ str, int ]

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

    @property
    def mass( self ):
        return driver.call(
            FfiCodeParallel( name = "PolynomialGrid",
                parallel_over = [ "polynomial_grid" ],
                fwd_body = "p.output( batch_index ) = p.polynomial_grid( batch_index ).mass();",
                bwd_body = "p.polynomial_grid( batch_index ).mass_bwd( p, batch_index );",
            ),
            output = Return( Tensor( *self.batch_axes_dict.keys() ), **self.batch_axes_dict ),
            polynomial_grid = self
        )

    def normalized_version( self, **kwargs ):
        if "mass" in kwargs:
            target_mass = kwargs[ "mass" ]
            current_mass = self.mass()
            if abs( current_mass - target_mass ) > 1e-6 * target_mass:
                return PolynomialGrid(
                    values = target_mass / current_mass * self.values,
                    frame = self.frame,
                    knots = self.knots
                )

        return self
