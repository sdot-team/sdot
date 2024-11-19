from .PowerDiagram import PowerDiagram, rt_int
import numpy as np
import vfs

class ConvexFunction:
    """ 
        Function =
          max( 
            -inf, 
            scalar_product( m_dirs[ 0 ], y ) - m_offs[ 0 ], 
            scalar_product( m_dirs[ 1 ], y ) - m_offs[ 1 ], 
            ...
          ) 
          for y such that 
            scalar_product( b_dirs[ 0 ], y ) - b_offs[ 0 ] <= 0,
            scalar_product( b_dirs[ 1 ], y ) - b_offs[ 1 ] <= 0,
            ...
    """

    def __init__( self, m_dirs, m_offs, b_dirs = None, b_offs = None, dtype = np.double ) -> None:
        # m_...
        self.m_dirs = np.asarray( m_dirs, dtype = dtype )
        self.m_offs = np.asarray( m_offs, dtype = dtype )

        assert( self.m_dirs.ndim == 2 )
        assert( self.m_offs.ndim == 1 )
        assert( self.m_offs.shape[ 0 ] == self.m_dirs.shape[ 0 ] )

        # b_...
        if b_dirs is None or not len( b_dirs ):
            assert( b_offs is None or not len( b_offs ) )
            b_dirs = np.zeros( [ 0, self.m_dirs.shape[ 1 ] ], dtype = dtype )
            b_offs = np.zeros( [ 0 ], dtype = dtype )
        self.b_dirs = np.asarray( b_dirs, dtype = dtype )
        self.b_offs = np.asarray( b_offs, dtype = dtype )

        assert( self.b_dirs.ndim == 2 )
        assert( self.b_offs.ndim == 1 )
        assert( self.b_offs.shape[ 0 ] == self.b_dirs.shape[ 0 ] )
        assert( self.b_dirs.shape[ 1 ] == self.m_dirs.shape[ 1 ] )

        # attributes by deduction
        self.nb_dims = self.m_dirs.shape[ 1 ]

    def make_approx_from_values_and_derivatives( sample_coords, f_val, f_der, b_dirs = None, b_offs = None ):
        m_dirs = []
        m_offs = []
        for num_point in range( sample_coords.shape[ 1 ] ):
            point = sample_coords[ :, num_point ]
            val = f_val( point )
            der = f_der( point )

            m_dirs.append( der )
            m_offs.append( np.dot( der, point ) - val )
        return ConvexFunction( m_dirs, m_offs, b_dirs, b_offs )

    def to_power_diagram( self ):
        weights = np.sum( self.m_dirs * self.m_dirs, axis = 1 ) - 2 * self.m_offs
        return PowerDiagram( self.m_dirs, weights, self.b_dirs, self.b_offs )

    def write_vtk( self, filename, fit_boundary = 1.0 ):
        pd = self.to_power_diagram()
        pd.write_vtk( filename, fit_boundary )

    def summary( self ):
        from sympy import Symbol, tensorproduct

        y = [ Symbol( f'y_{ d }' ) for d in range( self.nb_dims ) ]

        repr_max = lambda i: str( sum( tensorproduct( self.m_dirs[ i, : ], y ) ) -  self.m_offs[ i ] )
        repr_bnd = lambda i: str( sum( tensorproduct( self.b_dirs[ i, : ], y ) ) <= self.b_offs[ i ] )

        lst_max = [ repr_max( i ) for i in range( self.m_dirs.shape[ 0 ] ) ]
        lst_bnd = [ repr_bnd( i ) for i in range( self.b_dirs.shape[ 0 ] ) ]
        lst_max.sort()
        lst_bnd.sort()

        res = f"max({ ', '.join( lst_max ) })"
        if len( lst_bnd ):
            res += f" for { ', '.join( lst_bnd ) }"
        return res

    def legendre_transform( self ):
        # call the legendre_transform func
        func = vfs.function( 'legendre_transform', [ f'inc_file:sdot/legendre_transform/legendre_transform.h' ] )
        new_m_dirs, new_m_offs, new_b_dirs, new_b_offs = func( self.m_dirs, self.m_offs, self.b_dirs, self.b_offs, rt_int( self.nb_dims ) )
        new_m_dirs = np.reshape( new_m_dirs, [ -1, self.nb_dims ] )
        new_b_dirs = np.reshape( new_b_dirs, [ -1, self.nb_dims ] )

        # make a new ConvexFunction
        return ConvexFunction( new_m_dirs, new_m_offs, new_b_dirs, new_b_offs )

    def __repr__( self ):
        return self.summary()

    def is_inside_boundaries( self, x_k ):
        if isinstance( x_k, ( int, float ) ):
            x_k = [ x_k ]
        x_k = np.asarray( x_k )

        return self.b_dirs.shape[ 0 ] == 0 or ( self.b_dirs @ x_k - self.b_offs ).max() <= 0

    def __call__( self, x_k ):
        return self.val( x_k )
    
    def val( self, x_k ):
        if isinstance( x_k, ( int, float ) ):
            x_k = [ x_k ]
        x_k = np.asarray( x_k )

        if not self.is_inside_boundaries( x_k ):
            return float('inf')
        
        return ( self.m_dirs @ x_k - self.m_offs ).max()

    def grad( self, x_k ):
        if isinstance( x_k, ( int, float ) ):
            x_k = [ x_k ]
        x_k = np.asarray( x_k )

        if not self.is_inside_boundaries( x_k ):
            return float('inf')

        k = ( self.m_dirs @ x_k - self.m_offs ).argmax()     
        return self.m_dirs[ k ]

    def __add__( self, u2 ):
        u1 = self
        if isinstance( u2, ConvexFunction ): # addition of two convex functions
            if u1.nb_dims == u2.nb_dims:
                nb_dims = u1.nb_dims
            else:
                print('Dimensions do not match.')

            new_m_dirs = ( u1.m_dirs[ :, None, : ] + u2.m_dirs[ None, :, : ] ).reshape( ( -1, nb_dims ) )
            new_m_offs = ( u1.m_dirs[ :, None ] + u2.m_dirs[ None, : ] ).flatten()
            new_b_dirs = np.block( [ [ u1.b_dirs ], [ u2.b_dirs ] ] )
            new_b_offs = np.concatenate( [ u1.b_offs, u2.b_offs ] )

            return ConvexFunction( new_m_dirs, new_m_offs, new_b_dirs, new_b_offs )
            
        elif isinstance( u2, ( int, float ) ): # addition of a polyhedral function and a scalar
            return ConvexFunction( u1.m_dirs, u1.m_offs - u2, u1.b_dirs, u1.b_offs )
        
        else:
            raise NotImplementedError( "Can't add Polyhedral and {}".format( type( u2 ) ) )
        
    def __radd__( self, scal ):
        return self.__add__( scal )
    
    def __sub__( self, scal ):
        if isinstance( scal,( int, float ) ): 
            return ConvexFunction( self.m_dirs, self.m_offs + scal, self.b_dirs, self.b_offs )
        else:
            raise NotImplementedError( "Can't substract ConvexFunction from {}".format( type( scal ) ) )
    
    def __mul__( self, scal ):
        if isinstance( scal, ( int, float ) ): 
            assert( scal >= 0 )
            return ConvexFunction( scal * self.m_dirs, scal * self.m_offs, self.b_dirs, self.b_offs )
        else:
            raise NotImplementedError( "Can't multiply ConvexFunction and {}".format( type( scal ) ) )
            
    def __rmul__( self, scal ):
        return self.__mul__( scal )
        
    def __truediv__( self, scal ):
        if isinstance( scal, ( int, float ) ): 
            assert( scal > 0 )
            return ConvexFunction( self.m_dirs / scal, self.m_offs / scal, self.b_dirs, self.b_offs )
        else:
            raise NotImplementedError( "Can't divide {} by ConvexFunction.".format( type( scal ) ) )


    def __or__(self,u2): 
        u1 = self
        if isinstance( u2, ConvexFunction ): # maximum of two PHC functions
            new_m_dirs = np.block( [ [ u1.m_dirs ], [ u2.m_dirs ] ] )
            new_m_offs = np.concatenate( [ u1.m_offs, u2.m_offs ] )
            new_b_dirs = np.block( [ [ u1.b_dirs ], [ u2.b_dirs ] ] )
            new_b_offs = np.concatenate( [ u1.b_offs,u2.b_offs ] )
            return ConvexFunction( new_m_dirs, new_m_offs, new_b_dirs, new_b_offs )
            
        elif isinstance(u2,(int,float)): # addition of a polyhedral function and a scalar
            return u1.__or__( ConvexFunction( np.zero( [ 1, u1.nb_dims ] ), [ -u2 ] ) )
        
        else:
            raise NotImplementedError( "Can't take the maximum of ConvexFunction and {}".format( type( u2 ) ) )
        
    def __ror__( self, scal ):
        return self.__or__( scal )


    def __xor__( self, u2 ): # inf-convolution of two PHC functions
        if isinstance( u2, ConvexFunction ):
            return ( self.legendre_transform() + u2.legendre_transform() ).legendre_transform()
            
        else:
            raise NotImplementedError("Can't take the inf-convolution of ConvexFunction and {}".format(type(u2)))
