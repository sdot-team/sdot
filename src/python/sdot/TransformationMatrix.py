import numpy as np

class TransformationMatrix:
    def __init__( self, value = None ):
        if isinstance( value, TransformationMatrix ):
            self.value = value.value
        else:
            self.value = value

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
        
        raise ValueError( "invalid transformation" )
