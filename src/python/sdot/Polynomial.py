class Polynomial:
    """

    """
    def __init__( self, order, nb_variables, values ):
        self.nb_variables = nb_variables
        self.values = values
        self.order = order

    def truncate( self, max_order ):
        """Return a copy with coefficients above max_order zeroed."""
        n = self.nb_variables
        values = self.values.copy()
        if max_order < 3:
            values[ 1 + n + n * n : ] = 0
        if max_order < 2:
            values[ 1 + n : 1 + n + n * n ] = 0
        if max_order < 1:
            values[ 1 : 1 + n ] = 0
        return Polynomial( min( self.order, max_order ), n, values )

    def gradient( self, x ):
        import numpy
        x  = numpy.asarray( x )
        n  = self.nb_variables
        c1 = self.values[ 1 : 1 + n ]
        c2 = self.values[ 1 + n : 1 + n + n * n ].reshape( n, n )
        c3 = self.values[ 1 + n + n * n : ].reshape( n, n, n )
        g  = c1 + ( c2 + c2.T ) @ x
        g += numpy.einsum( 'ajk,j,k->a', c3 + c3.transpose(1,0,2) + c3.transpose(1,2,0), x, x )
        return g

    def argmin( self ):
        import numpy
        n  = self.nb_variables
        c1 = self.values[ 1 : 1 + n ]
        c2 = self.values[ 1 + n : 1 + n + n * n ].reshape( n, n )
        H  = c2 + c2.T  # Hessian of the quadratic part

        # exact quadratic solve if order-3 coefficients are negligible
        c3 = self.values[ 1 + n + n * n : ]

        x0 = numpy.linalg.solve( H, -c1 )
        if numpy.allclose( c3, 0 ):
            return x0

        # otherwise: BFGS from 0 (gradient descent toward the nearby minimum)
        from scipy.optimize import minimize
        res = minimize(
            fun = lambda x: self[ x ],
            jac = self.gradient,
            x0  = x0,
            method = 'BFGS',
            tol = 1e-12,
        )
        # ic( res )
        return res.x

    def __getitem__( self, key ):
        cpt = 0

        res = self.values[ cpt ]
        cpt += 1

        for i in range( self.nb_variables ):
            res += self.values[ cpt ] * key[ i ]
            cpt += 1

        for i in range( self.nb_variables ):
            for j in range( self.nb_variables ):
                res += self.values[ cpt ] * key[ i ] * key[ j ]
                cpt += 1

        for i in range( self.nb_variables ):
            for j in range( self.nb_variables ):
                for k in range( self.nb_variables ):
                    res += self.values[ cpt ] * key[ i ] * key[ j ] * key[ k ]
                    cpt += 1

        return res
