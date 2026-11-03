"""Fixed-connectivity power cells for seeds on a regular 2D grid.

Seeds x_i = (a, b) for 0 <= a, b < N, domain = [-1/2, N-1/2]^2, cell of i:
    { y = x - x_i : k.y <= h_k = ( |k|^2 + psi_i - psi_{i+k} ) / 2 }.

`StencilKernel( r )` keeps the neighbors with Chebyshev offset <= r, merges
offsets that share a direction, tightens the offsets to the support function
with a fixed list of pairs (a min per pair, no test), then reads the fixed
polygon (one vertex per pair of consecutive directions, zero-length edges for
redundant constraints).

`reference_cells` clips the box by every neighbor that can matter (pure
python, labels on edges) to get the exact cells.
"""
import math
import numpy as np


def primitive_directions( r ):
    dirs = [ ( a, b ) for a in range( -r, r + 1 ) for b in range( -r, r + 1 ) if ( a, b ) != ( 0, 0 ) and math.gcd( a, b ) == 1 ]
    dirs.sort( key = lambda d: math.atan2( d[ 1 ], d[ 0 ] ) )
    return np.array( dirs, dtype = float )


class StencilKernel:
    def __init__( self, r, pair_mode = "all" ):
        self.r = r
        self.dirs = primitive_directions( r )
        D = len( self.dirs )

        # offsets merged on each direction: h_d = min_m h_{m d} / m
        self.offsets = [] # ( direction index, multiple )
        for d, ( a, b ) in enumerate( self.dirs.astype( int ) ):
            m = 1
            while max( abs( m * a ), abs( m * b ) ) <= r:
                self.offsets.append( ( d, m ) )
                m += 1

        # pairs ( k, j, l, lam, mu ) with n_k = lam n_j + mu n_l, lam, mu > 0
        pairs = []
        for k in range( D ):
            cand = range( D ) if pair_mode == "all" else [ ( k - 1 ) % D, ( k + 1 ) % D ]
            for j in cand:
                for l in cand:
                    if j >= l or j == k or l == k:
                        continue
                    M = np.array( [ self.dirs[ j ], self.dirs[ l ] ] ).T
                    if abs( np.linalg.det( M ) ) < 1e-12:
                        continue
                    lam, mu = np.linalg.solve( M, self.dirs[ k ] )
                    if lam > 1e-12 and mu > 1e-12:
                        pairs.append( ( k, j, l, lam, mu ) )
        pairs.sort()
        self.pairs = pairs
        self.pk = np.array( [ p[ 0 ] for p in pairs ] )
        self.pj = np.array( [ p[ 1 ] for p in pairs ] )
        self.pl = np.array( [ p[ 2 ] for p in pairs ] )
        self.plam = np.array( [ p[ 3 ] for p in pairs ] )
        self.pmu = np.array( [ p[ 4 ] for p in pairs ] )

        # vertex k = lines k and k + 1
        self.vinv = np.array( [ np.linalg.inv( np.array( [ self.dirs[ k ], self.dirs[ ( k + 1 ) % D ] ] ) ) for k in range( D ) ] )

    def raw_h( self, psi ):
        """ ( n, D ) offsets and ( n, D ) index of the neighbor that owns each direction ( -1 for the box ) """
        N = psi.shape[ 0 ]
        A, B = np.meshgrid( np.arange( N ), np.arange( N ), indexing = "ij" )
        A, B = A.ravel(), B.ravel()
        flat = psi.ravel()
        D = len( self.dirs )
        h = np.full( ( N * N, D ), np.inf )
        owner = np.full( ( N * N, D ), -1 )
        for d, m in self.offsets:
            ka, kb = ( m * self.dirs[ d ] ).astype( int )
            na, nb = A + ka, B + kb
            ok = ( na >= 0 ) & ( na < N ) & ( nb >= 0 ) & ( nb < N )
            j = np.where( ok, na * N + nb, 0 )
            hk = np.where( ok, ( ka * ka + kb * kb + flat - flat[ j ] ) / 2 / m, np.inf )
            better = hk < h[ :, d ]
            h[ :, d ] = np.where( better, hk, h[ :, d ] )
            owner[ :, d ] = np.where( better, j, owner[ :, d ] )
        # box ( axis directions are always in the stencil )
        for d, ( a, b ) in enumerate( self.dirs.astype( int ) ):
            if ( a, b ) == ( 1, 0 ): hb = N - 0.5 - A
            elif ( a, b ) == ( -1, 0 ): hb = A + 0.5
            elif ( a, b ) == ( 0, 1 ): hb = N - 0.5 - B
            elif ( a, b ) == ( 0, -1 ): hb = B + 0.5
            else: continue
            better = hb <= h[ :, d ]
            h[ :, d ] = np.where( better, hb, h[ :, d ] )
            owner[ :, d ] = np.where( better, -1, owner[ :, d ] )
        return h, owner

    def tighten( self, h, sweeps = 1 ):
        for _ in range( sweeps ):
            cand = h[ :, self.pj ] * self.plam + h[ :, self.pl ] * self.pmu
            cand = np.where( np.isnan( cand ), np.inf, cand )
            red = np.full_like( h, np.inf )
            for k in range( h.shape[ 1 ] ):
                sel = self.pk == k
                if sel.any():
                    red[ :, k ] = cand[ :, sel ].min( axis = 1 )
            h = np.minimum( h, red )
        return h

    def cells( self, psi, sweeps = 1 ):
        h, owner = self.raw_h( psi )
        ht = self.tighten( h, sweeps )
        D = len( self.dirs )
        hh = np.stack( [ ht, np.roll( ht, -1, axis = 1 ) ], axis = 2 ) # ( n, D, 2 )
        v = np.einsum( "kab,nkb->nka", self.vinv, hh ) # vertex k = lines k, k + 1
        vp = np.roll( v, 1, axis = 1 ) # vertex k - 1
        area = 0.5 * np.sum( vp[ :, :, 0 ] * v[ :, :, 1 ] - vp[ :, :, 1 ] * v[ :, :, 0 ], axis = 1 )
        t = np.stack( [ -self.dirs[ :, 1 ], self.dirs[ :, 0 ] ], axis = 1 ) / np.linalg.norm( self.dirs, axis = 1 )[ :, None ]
        L = np.einsum( "nka,ka->nk", v - vp, t ) # signed length of edge k
        R = np.linalg.norm( v, axis = 2 ).max( axis = 1 )
        return dict( h = h, ht = ht, owner = owner, v = v, area = area, L = L, R = R )


# ------------------------------------------------------------------ reference
def _clip( poly, labels, n, c, lab ):
    """ clip polygon by n.y <= c; edge e goes from poly[ e ] to poly[ e + 1 ] with label labels[ e ] """
    m = len( poly )
    s = [ n[ 0 ] * p[ 0 ] + n[ 1 ] * p[ 1 ] - c for p in poly ]
    if max( s ) <= 0:
        return poly, labels, False
    out, outl = [], []
    for e in range( m ):
        p, q = poly[ e ], poly[ ( e + 1 ) % m ]
        sp, sq = s[ e ], s[ ( e + 1 ) % m ]
        if sp <= 0:
            out.append( p )
            outl.append( labels[ e ] )
            if sq > 0:
                t = sp / ( sp - sq )
                out.append( ( p[ 0 ] + t * ( q[ 0 ] - p[ 0 ] ), p[ 1 ] + t * ( q[ 1 ] - p[ 1 ] ) ) )
                outl.append( lab )
        elif sq <= 0:
            t = sp / ( sp - sq )
            out.append( ( p[ 0 ] + t * ( q[ 0 ] - p[ 0 ] ), p[ 1 ] + t * ( q[ 1 ] - p[ 1 ] ) ) )
            outl.append( labels[ e ] )
    return out, outl, True


def reference_cells( psi, tol = 1e-10 ):
    """ exact cells: area, faces { j: length }, max Chebyshev ring of active neighbors """
    N = psi.shape[ 0 ]
    pmax = psi.max()
    offs = [ ( a, b ) for a in range( -N, N + 1 ) for b in range( -N, N + 1 ) if ( a, b ) != ( 0, 0 ) ]
    offs.sort( key = lambda k: k[ 0 ] ** 2 + k[ 1 ] ** 2 )
    areas = np.zeros( N * N )
    faces = []
    rings = np.zeros( N * N, dtype = int )
    for a in range( N ):
        for b in range( N ):
            i = a * N + b
            poly = [ ( -a - 0.5, -b - 0.5 ), ( N - 0.5 - a, -b - 0.5 ), ( N - 0.5 - a, N - 0.5 - b ), ( -a - 0.5, N - 0.5 - b ) ]
            labels = [ -1, -1, -1, -1 ]
            pi = psi[ a, b ]
            for ka, kb in offs:
                nk = math.sqrt( ka * ka + kb * kb )
                R = max( math.hypot( *p ) for p in poly ) if poly else 0
                # no neighbor at distance >= nk can cut when nk^2 - 2 nk R + psi_i - psi_max > 0
                if nk >= R and nk * nk - 2 * nk * R + pi - pmax > 0:
                    break
                na, nb = a + ka, b + kb
                if not ( 0 <= na < N and 0 <= nb < N ):
                    continue
                c = ( ka * ka + kb * kb + pi - psi[ na, nb ] ) / 2
                poly, labels, _ = _clip( poly, labels, ( ka, kb ), c, na * N + nb )
                if not poly:
                    break
            m = len( poly )
            ar = 0.0
            f = {}
            for e in range( m ):
                p, q = poly[ e ], poly[ ( e + 1 ) % m ]
                ar += p[ 0 ] * q[ 1 ] - p[ 1 ] * q[ 0 ]
                ln = math.hypot( q[ 0 ] - p[ 0 ], q[ 1 ] - p[ 1 ] )
                if labels[ e ] >= 0 and ln > tol:
                    f[ labels[ e ] ] = f.get( labels[ e ], 0 ) + ln
            areas[ i ] = ar / 2
            faces.append( f )
            rings[ i ] = max( [ max( abs( j // N - a ), abs( j % N - b ) ) for j in f ], default = 0 )
    return areas, faces, rings
