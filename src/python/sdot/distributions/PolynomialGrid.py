from sdot.distributions.helpers.distribution_methods import TensorField, ListOfTensorFields, generate_distribution_methods, Distribution, driver
from itertools import product as iproduct
from typing import TYPE_CHECKING, Self
from bisect import bisect_right
from math import prod, floor


@generate_distribution_methods
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

    values    = TensorField( "shape * dim", "nb_coeffs" )
    frame     = TensorField( "dim + 1", "dim" )
    knots     = ListOfTensorFields( "dim", [ "shape[ index ]" ] )

    if TYPE_CHECKING:
        def __init__( self, values = None, frame = None, knots = None ): ...
        nb_coeffs : int  # (order+1)^dim
        shape     : list[ int ]

    def __getitem__( self, key ):
        if self.frame is not None:
            raise NotImplementedError
        if isinstance( key, ( int, float ) ):
            return self.__getitem__( [ key ] )
        assert len( key ) == self.dim

        dim = self.values.ndim - 1

        if self.knots is not None:
            knots = [ list( driver.to_numpy( self.knots[ d ] ) ) for d in range( dim ) ]
            ind = [ max( 0, min( bisect_right( knots[ d ], key[ d ] ) - 1, len( knots[ d ] ) - 2 ) ) for d in range( dim ) ]
        else:
            ind = [ max( 0, min( self.values.shape[ d ] - 1, int( key[ d ] ) ) ) for d in range( dim ) ]

        res = 0
        for cpt, powers in enumerate( iproduct( range( self.order + 1 ), repeat = dim ) ):
            monomial = prod( key[ d ] ** powers[ d ] for d in range( dim ) )
            res += float( self.values[ *ind, cpt ] ) * monomial
        return res

    def with_new_frame( self, transformation ) -> Self:
        raise NotImplementedError

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

    def primitive_function( self, batch_version: bool ):
        prim = f"{ 'batch_of_' if batch_version else '' }polynomial_grid( CtInt<{ self.order }>(), g_values, g_frame, g_knots, true )"
        incl = [ f"sdot/{ 'BatchOf' if batch_version else '' }PolynomialGrid.h" ]
        return prim, incl
