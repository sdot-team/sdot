"""Better filters: work in coordinates centred on the ring-r cell, so that the
affine part of psi ( a translation of the diagram ) costs nothing."""
import sys, math
import numpy as np
sys.path.insert( 0, __file__.rsplit( "/", 1 )[ 0 ] )
import ot_stats as O
from stencil2d import StencilKernel

N = O.N


def filters( K, psi, c, W ):
    """ returns ( ball, vert ) safe masks.
    ball: with centre m and radius rho of the ring-r cell, neighbor k is inactive if
          2 k.m + 2 |k| rho <= |k|^2 + psi_i - psi_j   ( checked for r < |k|_inf <= W ),
          beyond W: |k|^2 - |k| ( 2 rho + 2 |m| ) + psi_i - max psi >= 0 at |k| = W + 1
    vert: same window, but exact test max_v k.v <= h_k on the vertices """
    v = c[ "v" ]
    lo, hi = v.min( axis = 1 ), v.max( axis = 1 )
    m = ( lo + hi ) / 2
    rho = np.linalg.norm( v - m[ :, None, : ], axis = 2 ).max( axis = 1 )
    a, b = np.divmod( np.arange( N * N ), N )
    p = psi.ravel()
    ball = np.ones( N * N, bool )
    vert = np.ones( N * N, bool )
    for ka in range( -W, W + 1 ):
        for kb in range( -W, W + 1 ):
            if max( abs( ka ), abs( kb ) ) <= K.r: continue
            na, nb = a + ka, b + kb
            ok = ( na >= 0 ) & ( na < N ) & ( nb >= 0 ) & ( nb < N )
            j = np.where( ok, na * N + nb, 0 )
            h = ( ka * ka + kb * kb + p - p[ j ] ) / 2
            nk = math.hypot( ka, kb )
            ball &= ~ok | ( ka * m[ :, 0 ] + kb * m[ :, 1 ] + nk * rho <= h )
            vert &= ~ok | ( ( ka * v[ :, :, 0 ] + kb * v[ :, :, 1 ] ).max( axis = 1 ) <= h )
    s = W + 1
    far = s * s - s * ( 2 * rho + 2 * np.linalg.norm( m, axis = 1 ) ) + p - psi.max() >= 0
    return ball & far, vert & far


print( f"N={N}  ( flagged fraction, for ring r kernel; 'bad' = truly needs > r; miss must be 0 )" )
for kind in [ "sines", "bands", "noise" ]:
    for cst in [ 10, 100, 1000 ]:
        its = O.newton( O.masses( kind, cst ) )
        line = f"{kind:6s} c={cst:<5d}"
        for r in [ 1, 2 ]:
            K = O.Ks[ r ]
            worst = None
            for psi, ar, rings in its:
                cc = K.cells( psi )
                bad = rings > r
                for W in [ r + 2 ]:
                    ball, vert = filters( K, psi, cc, W )
                    miss = ( ( ball | vert ) & bad ).sum()
                    st = ( bad.mean(), ( ~ball ).mean(), ( ~vert ).mean(), miss )
                    worst = st if worst is None or st[ 2 ] > worst[ 2 ] else worst
            psi, ar, rings = its[ -1 ]
            cc = K.cells( psi )
            ball, vert = filters( K, psi, cc, r + 2 )
            line += f" | r{r} final bad {( rings > r ).mean():.3f} ball {( ~ball ).mean():.3f} vert {( ~vert ).mean():.3f}" \
                    f" (worst it: bad {worst[ 0 ]:.3f} vert {worst[ 2 ]:.3f}) miss {worst[ 3 ]}"
        print( line )
