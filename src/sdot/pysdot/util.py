"""`pysdot.util`: `FastMarching`, the geodesic distance to targets in a domain ( for the crowd motion examples ). Needs `scikit-fmm`."""
import numpy as np
import math


class FastMarching:
    def __init__( self, domain, targets, space_step ):
        import skfmm
        self.mi = np.asarray( domain.min_position(), dtype = float ) - space_step
        self.ma = np.asarray( domain.max_position(), dtype = float ) + space_step
        self.ss = space_step

        nd = self.mi.shape[ 0 ]
        if nd != 2:
            raise NotImplementedError( "FastMarching: 2D only" )

        # a grid
        n = [ math.ceil( ( self.ma[ d ] - self.mi[ d ] ) / space_step ) for d in range( nd ) ]
        s = [ np.linspace( self.mi[ d ], self.ma[ d ], n[ d ] ) for d in range( nd ) ]
        g = np.meshgrid( *s, indexing = 'ij' )
        c = np.hstack( [ g[ d ].reshape( ( -1, 1 ) ) for d in range( nd ) ] )

        # the forbidden zones
        mask = np.array( [ not domain.contains( c[ i, : ] ) for i in range( c.shape[ 0 ] ) ] ).reshape( n )
        phi = - 1.0 * np.ones_like( mask, dtype = float )
        for t in targets:
            c = np.floor( ( np.asarray( t ) - self.mi ) / space_step ).astype( int )
            phi[ c[ 0 ], c[ 1 ] ] = 0

        self.dist = skfmm.distance( np.ma.MaskedArray( phi, mask ), dx = [ space_step for d in range( nd ) ] )

        # an integer spiral ( for `grad` )
        self._offsets = [ ( x, y ) for y in range( -75, 76 ) for x in range( -75, 76 ) ]
        self._offsets.sort( key = lambda p: p[ 0 ]**2 + p[ 1 ]**2 )

    def grad( self, position, min_dist = 0 ):
        ib = np.floor( ( position - self.mi ) / self.ss ).astype( int )
        msk = np.ma.getmaskarray( self.dist )
        for dx, dy in self._offsets:
            ix = ib[ 0 ] + dx
            iy = ib[ 1 ] + dy
            if ix >= 0 and iy >= 0 and ix < msk.shape[ 0 ] - 1 and iy < msk.shape[ 1 ] - 1 and not msk[ ix, iy ] and not msk[ ix + 1, iy ] and not msk[ ix, iy + 1 ]:
                cd = self.dist[ ix, iy ]
                re = np.array( [ self.dist[ ix + 1, iy ] - cd, self.dist[ ix, iy + 1 ] - cd ] )
                no = np.linalg.norm( re )
                if abs( cd ) < min_dist:
                    return re / ( no + ( no == 0 ) ) * abs( cd ) / min_dist
                return re / ( no + ( no == 0 ) )
        return np.array( [ 0, 1 ] )
