from sdot.distributions.helpers.distribution_methods import TensorField, ListOfTensorFields, generate_distribution_methods, Distribution, driver
from itertools import product as iproduct
from .PolynomialGrid import PolynomialGrid
from typing import TYPE_CHECKING
import numpy as np


@generate_distribution_methods
class SplineGrid( Distribution ):
    """
    values      : tensor[ x, y, z, ... ]
    frame       : tensor[ nb_vec, dim ] — canonical frame by default
    knots       : [ tensor[ num_point ] for axis ] — linspace(0,1) by default
    continuity  : int — minimum continuity order (default 0 = C0)

    continuity=0  →  Q_1 bilinear per cell, local Vandermonde solve
    continuity=1  →  natural cubic spline (C2), tensor-product 1D solve per axis
    """

    values      = TensorField( "shape * dim" )
    frame       = TensorField( "dim + 1", "dim" )
    knots       = ListOfTensorFields( "dim", [ "shape[ index ]" ] )
    continuity  = 0

    if TYPE_CHECKING:
        def __init__( self, values = None, frame = None, knots = None, continuity = 0 ): ...
        shape : list[ int ]

    # polynomial order implied by the continuity requirement:
    # C0 → Q_1 (degree 1),  C1 → cubic (degree 3),  Ck → degree 2k+1
    @property
    def order( self ):
        return 2 * self.continuity + 1

    def normalized_version( self ):
        if self.values is None:
            return PolynomialGrid( None, self.frame, self.knots )

        values = self.values
        dim    = values.ndim
        knots  = self.defined_knots()

        # for d in range( dim ):
        #     if values.shape[ d ] < self.order + 1:
        #         raise ValueError( f"Dimension {d}: need >= {self.order + 1} nodes for continuity {self.continuity}" )

        if self.continuity == 0:
            return self._c0_normalized( values, dim, knots )
        elif self.continuity == 1:
            return self._c1_normalized( values, dim, knots )
        else:
            raise NotImplementedError( f"continuity={self.continuity} not yet implemented" )

    # ── C0: Q_1 local Vandermonde per cell ────────────────────────────────────

    def _c0_normalized( self, values, dim, knots ):
        order     = 1
        n_cells   = [ s - 1 for s in values.shape ]
        all_offsets = list( iproduct( range( order + 1 ), repeat=dim ) )

        starts_np = [ np.minimum( np.arange( n_cells[ d ] ), values.shape[ d ] - order - 1 )
                      for d in range( dim ) ]

        def bcast( d, t ):
            return t.reshape( [ -1 if dd == d else 1 for dd in range( dim ) ] )

        gathered = [ values[ tuple( bcast( d, starts_np[ d ] + off[ d ] ) for d in range( dim ) ) ]
                     for off in all_offsets ]
        b    = driver.stack( gathered, axis=-1 )
        ones = driver.ones( n_cells )

        V_rows = []
        for off in all_offsets:
            node_x   = [ bcast( d, knots[ d ][ starts_np[ d ] + off[ d ] ] ) * ones for d in range( dim ) ]
            row_cols = []
            for powers in all_offsets:
                monomial = ones
                for d in range( dim ):
                    for _ in range( powers[ d ] ):
                        monomial = monomial * node_x[ d ]
                row_cols.append( monomial )
            V_rows.append( driver.stack( row_cols, axis=-1 ) )

        V = driver.stack( V_rows, axis=-2 )
        c = driver.linalg_solve( V, driver.expand_dims( b, -1 ) )[ ..., 0 ]
        return PolynomialGrid( c, self.frame, self.knots )

    # ── C1: natural cubic spline, tensor-product along each axis ─────────────
    # Gives C2 continuity (natural cubic = C2 ⊃ C1).
    # Each axis is solved independently; operations compose by linearity.
    # The output Q_3 coefficients are in global x, matching PolynomialGrid ordering.

    def _c1_normalized( self, values, dim, knots ):
        # Start with shape [n_0, n_1, ..., n_{d-1}].
        # After processing axis d we get shape [..., n_d-1, ..., 4]
        # (n_d replaced by n_d-1, and 4 Q_3 coefficients appended).
        # After all axes: [n_0-1, ..., n_{d-1}-1, 4, 4, ..., 4]
        # Reshape trailing 4^dim into one dim for PolynomialGrid.

        result = values
        for d in range( dim ):
            n    = result.shape[ d ]
            kp   = knots[ d ]                    # [n_d] knot positions
            h    = kp[ 1: ] - kp[ :-1 ]          # [n_d-1] cell widths

            # ── tridiagonal system for interior second derivatives M ────────
            # A[i,i]   = 2*(h[i]+h[i+1])
            # A[i,i±1] = h[i+1] / h[i]  (symmetric)
            # RHS[i]   = 6*((y[i+2]-y[i+1])/h[i+1] - (y[i+1]-y[i])/h[i])
            # for i = 0 ... n-3  (interior nodes)
            # Boundary: M[0] = M[n-1] = 0  (natural spline)

            m = n - 2   # number of interior nodes
            if m <= 0:
                raise ValueError( f"Axis {d}: need >= 3 nodes for C1 spline" )

            # Build A as full matrix — size (m×m), depends on h (differentiable)
            # Use driver.stack to keep differentiability w.r.t. knots
            diag  = 2 * ( h[ :-1 ] + h[ 1: ] )                  # [m]
            off_d = h[ 1:-1 ]                                     # [m-1]

            # Build rows of A: each row has at most 3 non-zero entries
            A_rows = []
            for i in range( m ):
                row_parts = [ driver.zeros( [ 1 ] ) ] * m
                row_parts[ i ] = diag[ i:i+1 ]
                if i > 0:
                    row_parts[ i - 1 ] = off_d[ i-1:i ]
                if i < m - 1:
                    row_parts[ i + 1 ] = off_d[ i:i+1 ]
                A_rows.append( driver.stack( row_parts, axis=0 ) )   # [m]
            A = driver.stack( A_rows, axis=0 )                        # [m, m]

            # ── RHS: shape is [..., m, ...] where d is the axis ─────────────
            # Move axis d to the front for easier slicing, then move back
            result_t = driver.moveaxis( result, d, 0 )          # [n, ...]
            y        = result_t

            rhs_t = 6 * (   ( y[ 2: ] - y[ 1:-1 ] ) / _expand_for( h[ 1: ], y.ndim - 1 )
                          - ( y[ 1:-1 ] - y[ :-2 ] ) / _expand_for( h[ :-1 ], y.ndim - 1 ) )  # [m, ...]

            # ── Solve A @ M_int = rhs  (batched over all other axes) ────────
            # A: [m, m],  rhs_t: [m, *rest]  →  M_int: [m, *rest]
            rest_shape = list( rhs_t.shape[ 1: ] )
            n_rest = 1
            for s in rest_shape:
                n_rest *= s
            rhs_flat = rhs_t.reshape( m, n_rest )                 # [m, n_rest]
            M_int    = driver.linalg_solve(
                driver.expand_dims( A, -1 ).broadcast_to( m, m, n_rest ),  # won't work
                driver.expand_dims( rhs_flat, -1 )
            )[ ..., 0 ]                                            # [m, n_rest]
            M_int = M_int.reshape( [ m ] + rest_shape )           # [m, *rest]

            # Pad boundary zeros:  M = [0, M_int[0], ..., M_int[m-1], 0]
            zeros  = driver.zeros( [ 1 ] + rest_shape )
            M      = driver.stack( [ zeros, M_int, zeros ], axis=0 )    # [n, *rest]

            # ── Convert to Q_3 coefficients in global x ─────────────────────
            # On cell i: S(t) = A*(h-t)^3 + B*t^3 + C*(h-t) + D*t,  t = x - kp[i]
            # A = M[i]/(6h),  B = M[i+1]/(6h),  C = y[i]/h - M[i]*h/6,  D = y[i+1]/h - M[i+1]*h/6
            # In Q_3 global x:
            #   c3 = (B-A)          = (M[i+1]-M[i]) / (6h)
            #   c2 = 3*A*h          = M[i] / 2
            #   c1 = C - 3*A*h^2 - D = (y[i+1]-y[i])/h - M[i+1]*h/6 - M[i]*h/3
            #   c0 = A*h^3 + D*h    = y[i]

            hi  = _expand_for( h,       y.ndim - 1 )   # [n-1, 1, ...]
            Mi  = M[ :-1 ]                               # [n-1, *rest]
            Mi1 = M[ 1:  ]                               # [n-1, *rest]
            yi  = y[ :-1 ]
            yi1 = y[ 1:  ]

            c0 = yi
            c1 = ( yi1 - yi ) / hi - Mi1 * hi / 6 - Mi * hi / 3
            c2 = Mi / 2
            c3 = ( Mi1 - Mi ) / ( 6 * hi )

            # Stack as [..., n_d-1, ..., 4] then move axis back
            cell_coeffs = driver.stack( [ c0, c1, c2, c3 ], axis=-1 )  # [n-1, *rest, 4]
            # Move the n-1 axis back to position d, and append the 4-axis at end
            result = driver.moveaxis( cell_coeffs, 0, d )               # [*, n-1, *, 4]

        # result shape: [n_0-1, ..., n_{d-1}-1, 4, 4, ..., 4]
        # Reshape trailing 4^dim into one dim
        cell_shape = list( result.shape[ :dim ] )
        result = result.reshape( cell_shape + [ 4 ** dim ] )
        return PolynomialGrid( result, self.frame, self.knots )

    def defined_knots( self ):
        if self.knots is None:
            return [ driver.linspace( 0, 1, self.values.shape[ d ] ) for d in range( self.values.ndim ) ]
        return self.knots


def _expand_for( t, ndim ):
    """ Reshape a 1-D tensor to [n, 1, 1, ...] with ndim trailing ones. """
    return t.reshape( [ -1 ] + [ 1 ] * ndim )
