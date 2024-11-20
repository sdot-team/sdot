import numpy as np

class TransformationMatrix:
    def __init__( self, value = None, ndim = None ):
        if isinstance( value, TransformationMatrix ):
            self.value = value.value
        elif value is None and ndim is not None:
            self.value = np.eye( ndim + 1 )
        else:
            self.value = value

    def scale( self, s ):
        if isinstance( s, float ) or isinstance( s, int ):
            self.value = self.get()
            for i in range( self.value.shape[ 0 ] ):
                self.value[ i, i ] *= s
        else:
            raise RuntimeError( "TODO" )

    def translate( self, v ):
        self.value = self.get( v.size )
        self.value[ :-1, -1 ] = v

    def dir( self, d ):
        M = self.get( d.size )
        return M[ :-1, :-1 ] @ d

    def pos( self, p ):
        n = np.array( list( p ) + [ 1 ] )
        M = self.get()
        return M[ :-1, : ] @ n
    
    def get( self, ndim = None ):
        # None
        if self.value is None:
            return np.eye( ndim + 1 )

        # ( M, V )
        if isinstance( self.value, tuple ):
            assert( len( self.value ) == 2 )
            M = np.array( self.value[ 0 ] )
            V = np.array( self.value[ 1 ] )
            
            assert( M.shape[ 0 ] == M.shape[ 1 ] )
            assert( M.shape[ 0 ] == V.shape[ 0 ] )
            s = M.shape[ 0 ]

            res = np.eye( s + 1 )
            res[ :s, :s ] = M
            res[ :s, s ] = V
            return res
        
        # full matrix
        array = np.array( self.value )
        if len( array.shape ) == 2:
            assert( array.shape[ 0 ] == array.shape[ 1 ] )
            if ndim is not None:
                assert( array.shape[ 0 ] == ndim + 1 )
            return array
        
        # vector
        if len( array.shape ) == 1:
            s = array.size

            res = np.eye( s + 1 )
            res[ :s, s ] = array
            return res

        print( self.value )        
        raise ValueError( "invalid transformation" )
