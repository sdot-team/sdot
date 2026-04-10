import itertools
import numpy


class Polynomial:
    def __init__( self, nb_variables, order, values ):
        self.nb_variables = nb_variables
        self.values = values
        self.order = order

    @staticmethod
    def size_for( nb_variables, order ):
        return sum( nb_variables ** k for k in range( order + 1 ) )

    def _degree_offset( self, d ):
        """Index in values where degree-d coefficients start."""
        n = self.nb_variables
        return sum( n ** k for k in range( d ) )

    def truncate( self, max_order ):
        """Return a copy with coefficients above max_order zeroed."""
        values = self.values.copy()
        values[ self._degree_offset( max_order + 1 ) : ] = 0
        return Polynomial( self.nb_variables, min( self.order, max_order ), values )

    def gradient( self, x ):
        x = numpy.asarray( x )
        n = self.nb_variables
        g = numpy.zeros( n )
        letters = 'abcdefghijklmnop'
        offset = 1  # degree-0 has no gradient
        for d in range( 1, self.order + 1 ):
            C = self.values[ offset : offset + n ** d ].reshape( [n] * d )
            c_idx = letters[ 1 : d + 1 ]  # d letters for C's indices
            for p in range( d ):
                free_c_idx = c_idx[ :p ] + 'a' + c_idx[ p + 1: ]
                x_parts    = list( c_idx[ :p ] + c_idx[ p + 1: ] )  # d-1 letters
                lhs = free_c_idx + ( ',' + ','.join( x_parts ) if x_parts else '' )
                g  += numpy.einsum( lhs + '->a', C, *( [ x ] * ( d - 1 ) ) )
            offset += n ** d
        return g

    def cut_positive_region( self, cell, n_cuts = 24 ):
        """Cut `cell` to the region where this polynomial (truncated at order 2) is positive.

        The order-2 zero-set is an ellipsoid around the quadratic minimiser x*.
        For each tangent plane of that ellipsoid on the side facing x=0, we add a
        half-plane cut to `cell` (convention: cell.cut(dir,dot) keeps dir·x ≤ dot).
        Only planes that do not remove x=0 are added (safety check: dot_val ≥ 0).
        """
        n  = self.nb_variables
        c0 = float( self.values[ 0 ] )
        c1 = self.values[ 1 : 1 + n ].astype( float )
        C2 = self.values[ 1 + n : 1 + n + n * n ].reshape( n, n ).astype( float )
        H  = C2 + C2.T  # full Hessian of the quadratic part

        x_star = numpy.linalg.solve( H, -c1 )
        f_star = c0 + 0.5 * c1 @ x_star   # f(x*) via c0 + ½ c1·x* (Hx*=-c1)

        if f_star >= 0:
            return  # polynomial is always positive — nothing to cut

        r = float( numpy.sqrt( -2.0 * f_star ) )
        L = numpy.linalg.cholesky( H )    # H = L @ L.T,  L lower-triangular

        # --- generate candidate unit directions on the n-sphere ----------------
        if n == 1:
            candidates = [ numpy.array( [1.0] ), numpy.array( [-1.0] ) ]
        elif n == 2:
            angles = numpy.linspace( 0, 2 * numpy.pi, n_cuts, endpoint=False )
            candidates = [ numpy.array( [ numpy.cos(a), numpy.sin(a) ] ) for a in angles ]
        else:
            # Fibonacci lattice on S^{n-1} (approximate uniform coverage)
            rng = numpy.random.default_rng( 0 )
            raw = rng.standard_normal( ( n_cuts * 4, n ) )
            norms = numpy.linalg.norm( raw, axis=1, keepdims=True )
            candidates = list( raw / norms )

        # --- add one cut per safe direction ------------------------------------
        for u in candidates:
            u = numpy.asarray( u, dtype=float )
            Lu = L @ u                          # outward normal of ellipsoid at x(u)
            dir_cut = -Lu                       # cell.cut keeps dir·x ≤ dot
            dot_val = float( dir_cut @ x_star ) - r   # = -(Lu)·x* - r

            if dot_val < 0:
                continue   # this plane would cut away x=0; skip

            cell.cut( dir_cut, dot_val )

    def argmin( self ):
        n  = self.nb_variables
        c1 = self.values[ 1 : 1 + n ]
        c2 = self.values[ 1 + n : 1 + n + n * n ].reshape( n, n )
        H  = c2 + c2.T  # Hessian of the quadratic part
        x0 = numpy.linalg.solve( H, -c1 )

        # exact quadratic solve when all higher-order coefficients are negligible
        if numpy.allclose( self.values[ self._degree_offset( 3 ) : ], 0 ):
            return x0

        from scipy.optimize import minimize
        res = minimize(
            fun    = lambda x: self[ x ],
            jac    = self.gradient,
            x0     = x0,
            method = 'BFGS',
            tol    = 1e-12,
        )

        return res.x

    def __getitem__( self, key ):
        res = self.values[ 0 ]
        off = 1

        n = self.nb_variables
        for d in range( 1, self.order + 1 ):
            for idx in itertools.product( range( n ), repeat = d ):
                monomial = self.values[ off ]
                for i in idx:
                    monomial *= key[ i ]
                res += monomial
                off += 1
        return res
