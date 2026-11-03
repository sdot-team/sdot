"""Cheaper tightening: consecutive pairs only, repeated sweeps, vs all pairs."""
import sys
import numpy as np
sys.path.insert( 0, __file__.rsplit( "/", 1 )[ 0 ] )
import ot_stats as O
from stencil2d import StencilKernel

N = O.N
psis = []
for kind in [ "sines", "bands", "noise" ]:
    for cst in [ 10, 1000 ]:
        psis += [ p for p, _, _ in O.newton( O.masses( kind, cst ) ) ]
rng = np.random.default_rng( 3 )
# psis += [ s * rng.standard_normal( ( N, N ) ) for s in [ 0.1, 0.3 ] ]
for r in [ 1, 2 ]:
    Kall = StencilKernel( r )
    Kc = StencilKernel( r, pair_mode = "consecutive" )
    used = np.zeros( len( Kall.pairs ), bool )
    errs = { s: 0.0 for s in [ 1, 2, 3, 4, 6 ] }
    for psi in psis:
        h, _ = Kall.raw_h( psi )
        ref = Kall.tighten( h )
        cand = h[ :, Kall.pj ] * Kall.plam + h[ :, Kall.pl ] * Kall.pmu
        # pairs that reach the minimum ( strictly below h_k ) somewhere
        tk = ref[ :, Kall.pk ]
        used |= ( ( np.abs( cand - tk ) < 1e-12 ) & ( cand < h[ :, Kall.pk ] - 1e-12 ) ).any( axis = 0 )
        a, b = np.divmod( np.arange( N * N ), N ); inner = ( a >= r ) & ( a < N - r ) & ( b >= r ) & ( b < N - r )
        fin = np.isfinite( ref ) & inner[ :, None ]
        for s in errs:
            e = np.abs( Kc.tighten( h, sweeps = s ) - ref )
            errs[ s ] = max( errs[ s ], e[ fin ].max() )
    print( f"ring {r}: all pairs {len( Kall.pairs )}, ever useful {used.sum()}, consecutive {len( Kc.pairs )}; "
           f"max |h - h_all| after sweeps: " + ", ".join( f"{s}: {e:.1e}" for s, e in errs.items() ) )
