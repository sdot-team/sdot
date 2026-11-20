"""Fixed-connectivity power cells for seeds on a lattice ( square or hexagonal ), the scheme of grid_stencil/proto/stencil2d.py
made generic: offsets of a ring, merged per direction, tightened to the support function by a fixed list of pairs, then
one vertex per pair of consecutive directions.

Box faces are directions of their own ( +-x, +-y ) when the lattice does not already have them: on the hexagonal lattice
( rows along x ) ring 1 has +-x but not +-y, ring 2 has both.
"""
import math
import numpy as np
import lattice2d as L


def lattice_basis( kind ):
    if kind == "square": return np.array( [ [ 1.0, 0.0 ], [ 0.0, 1.0 ] ] )
    return np.array( [ [ 1.0, 0.0 ], [ 0.5, L.S3 / 2 ] ] ) # axial q, r


def ring_offsets( kind, r ):
    offs = []
    for a in range( -r, r + 1 ):
        for b in range( -r, r + 1 ):
            if ( a, b ) == ( 0, 0 ): continue
            d = max( abs( a ), abs( b ) ) if kind == "square" else ( abs( a ) + abs( b ) + abs( a + b ) ) // 2
            if d <= r: offs.append( ( a, b ) )
    return offs


class StencilKernel:
    def __init__( self, kind, r, box_dirs = True ):
        self.kind, self.r = kind, r
        Bm = lattice_basis( kind )
        offs = ring_offsets( kind, r )
        vec = np.array( offs, float ) @ Bm
        # directions = primitive offsets, multiples merged
        dirs, self.offsets = [], [] # offsets: ( lattice offset, direction index, multiple )
        for o, v in zip( offs, vec ):
            g = math.gcd( *o )
            prim = ( o[ 0 ] // g, o[ 1 ] // g )
            if prim not in [ d[ 0 ] for d in dirs ]:
                dirs.append( ( prim, np.array( prim, float ) @ Bm ) )
        if box_dirs:
            for b in [ ( 1, 0 ), ( -1, 0 ), ( 0, 1 ), ( 0, -1 ) ]:
                bv = np.array( b, float )
                if not any( np.allclose( d[ 1 ] / np.linalg.norm( d[ 1 ] ), bv ) for d in dirs ):
                    dirs.append( ( None, bv ) )
        dirs.sort( key = lambda d: math.atan2( d[ 1 ][ 1 ], d[ 1 ][ 0 ] ) )
        self.dir_off = [ d[ 0 ] for d in dirs ]
        self.dirs = np.array( [ d[ 1 ] for d in dirs ] )
        for o in offs:
            g = math.gcd( *o )
            prim = ( o[ 0 ] // g, o[ 1 ] // g )
            self.offsets.append( ( o, self.dir_off.index( prim ), g ) )
        D = len( self.dirs )
        pairs = []
        for k in range( D ):
            for j in range( D ):
                for l in range( j + 1, D ):
                    if j == k or l == k: continue
                    M = np.array( [ self.dirs[ j ], self.dirs[ l ] ] ).T
                    if abs( np.linalg.det( M ) ) < 1e-12: continue
                    lam, mu = np.linalg.solve( M, self.dirs[ k ] )
                    if lam > 1e-12 and mu > 1e-12:
                        pairs.append( ( k, j, l, lam, mu ) )
        pairs.sort()
        self.pairs = pairs
        self.pk, self.pj, self.pl = ( np.array( [ p[ i ] for p in pairs ] ) for i in range( 3 ) )
        self.plam, self.pmu = ( np.array( [ p[ i ] for p in pairs ] ) for i in ( 3, 4 ) )
        self.vinv = np.array( [ np.linalg.inv( np.array( [ self.dirs[ k ], self.dirs[ ( k + 1 ) % D ] ] ) ) for k in range( D ) ] )

    def describe( self ):
        return f"{self.kind} ring {self.r}: {len( self.offsets )} neighbours, {len( self.dirs )} directions, {len( self.pairs )} pairs"

    def raw_h( self, lat, psi ):
        n, D = lat.n, len( self.dirs )
        lut = { tuple( k ): i for i, k in enumerate( lat.idx.tolist() ) }
        h = np.full( ( n, D ), np.inf )
        owner = np.full( ( n, D ), -1 )
        Bm = lattice_basis( lat.kind )
        for o, d, m in self.offsets:
            j = np.array( [ lut.get( ( a + o[ 0 ], b + o[ 1 ] ), -1 ) for a, b in lat.idx.tolist() ] )
            ok = j >= 0
            v = np.array( o, float ) @ Bm
            hk = np.where( ok, ( v @ v + psi - psi[ np.maximum( j, 0 ) ] ) / 2 / m, np.inf )
            better = hk < h[ :, d ]
            h[ :, d ] = np.where( better, hk, h[ :, d ] )
            owner[ :, d ] = np.where( better, j, owner[ :, d ] )
        x0, x1, y0, y1 = lat.box
        px, py = lat.pos[ :, 0 ], lat.pos[ :, 1 ]
        nrm = np.linalg.norm( self.dirs, axis = 1 )
        for d, u in enumerate( self.dirs / nrm[ :, None ] ):
            # box face along +-x / +-y, in the units of the direction ( n.y <= h, |n| = nrm )
            if np.allclose( u, ( 1, 0 ) ): hb = ( x1 - px ) * nrm[ d ]
            elif np.allclose( u, ( -1, 0 ) ): hb = ( px - x0 ) * nrm[ d ]
            elif np.allclose( u, ( 0, 1 ) ): hb = ( y1 - py ) * nrm[ d ]
            elif np.allclose( u, ( 0, -1 ) ): hb = ( py - y0 ) * nrm[ d ]
            else: continue
            better = hb <= h[ :, d ]
            h[ :, d ] = np.where( better, hb, h[ :, d ] )
            owner[ :, d ] = np.where( better, -1, owner[ :, d ] )
        return h, owner

    def tighten( self, h ):
        cand = h[ :, self.pj ] * self.plam + h[ :, self.pl ] * self.pmu
        cand = np.where( np.isnan( cand ), np.inf, cand )
        ht = h.copy()
        for k in range( h.shape[ 1 ] ):
            sel = self.pk == k
            if sel.any(): ht[ :, k ] = np.minimum( ht[ :, k ], cand[ :, sel ].min( axis = 1 ) )
        return ht

    def cells( self, lat, psi ):
        h, owner = self.raw_h( lat, psi )
        ht = self.tighten( h )
        D = len( self.dirs )
        hh = np.stack( [ ht, np.roll( ht, -1, axis = 1 ) ], axis = 2 )
        v = np.einsum( "kab,nkb->nka", self.vinv, hh )
        vp = np.roll( v, 1, axis = 1 )
        area = 0.5 * np.sum( vp[ :, :, 0 ] * v[ :, :, 1 ] - vp[ :, :, 1 ] * v[ :, :, 0 ], axis = 1 )
        t = np.stack( [ -self.dirs[ :, 1 ], self.dirs[ :, 0 ] ], axis = 1 ) / np.linalg.norm( self.dirs, axis = 1 )[ :, None ]
        Ln = np.einsum( "nka,ka->nk", v - vp, t )
        return dict( h = h, ht = ht, owner = owner, v = v, area = area, L = Ln )


if __name__ == "__main__":
    for kind in [ "square", "hex" ]:
        for r in [ 1, 2 ]:
            print( StencilKernel( kind, r ).describe() )
