"""Fixed-stencil cells vs exact clipping on synthetic psi families."""
import sys, time
import numpy as np
sys.path.insert( 0, __file__.rsplit( "/", 1 )[ 0 ] )
from stencil2d import StencilKernel, reference_cells

N = 24
rng = np.random.default_rng( 0 )
A, B = np.meshgrid( np.arange( N ), np.arange( N ), indexing = "ij" )

fams = {
    "constant": np.zeros( ( N, N ) ),
    "quadratic": 0.3 * A * A - 0.2 * A * B + 0.1 * B * B,
    "noise 0.1": 0.1 * rng.standard_normal( ( N, N ) ),
    "noise 0.5": 0.5 * rng.standard_normal( ( N, N ) ),
    "noise 1.5": 1.5 * rng.standard_normal( ( N, N ) ),
}

for r in [ 1, 2 ]:
    K = StencilKernel( r )
    print( f"ring {r}: {len( K.dirs )} directions, {len( K.offsets )} neighbors, {len( K.pairs )} pairs" )

for name, psi in fams.items():
    t0 = time.time()
    ar, faces, rings = reference_cells( psi )
    tref = time.time() - t0
    line = f"{name:10s} ref {tref:5.1f}s sum={ar.sum():.6f} rings={np.bincount( rings ).tolist()} empty={( ar <= 1e-12 ).sum()}"
    for r in [ 1, 2 ]:
        K = StencilKernel( r )
        c = K.cells( psi )
        covered = ( rings <= r ) & ( ar > 1e-12 )
        err = np.abs( c[ "area" ] - ar )
        line += f" | r{r}: covered {covered.mean():.3f} err_cov {err[ covered ].max() if covered.any() else 0:.1e} minL {c[ 'L' ].min():.1e}"
        # faces of covered cells
        ferr = 0.0
        for i in np.flatnonzero( covered ):
            got = {}
            for d in range( len( K.dirs ) ):
                j = c[ "owner" ][ i, d ]
                if j >= 0 and c[ "L" ][ i, d ] > 1e-10:
                    got[ j ] = got.get( j, 0 ) + c[ "L" ][ i, d ]
            keys = set( got ) | set( faces[ i ] )
            for j in keys:
                ferr = max( ferr, abs( got.get( j, 0 ) - faces[ i ].get( j, 0 ) ) )
        line += f" ferr {ferr:.1e}"
    print( line )
