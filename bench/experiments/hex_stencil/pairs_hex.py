"""Which tightening pairs ( k - a, k + b ) of the hexagonal ring 2 ever set ht_k, on the psi of OT solves ( N = 16, iterates
included ) and on noisy psi: counts per ( a, b ). A pair is useful at a cell when it is strictly below h_k and equal to ht_k."""
import sys, math
import numpy as np
sys.path.insert( 0, __file__.rsplit( "/", 1 )[ 0 ] )
import lattice2d as L
import stencil_hex as S

lat = L.hexagonal( 15, 17 )
K = S.StencilKernel( "hex", 2 )
nrm = np.linalg.norm( K.dirs, axis = 1 )
ang = np.round( np.degrees( np.arctan2( K.dirs[ :, 1 ], K.dirs[ :, 0 ] ) ) ) % 360
psis = []
x, y = lat.unit_coords()
for f in [ np.sin( 4 * np.pi * x ) * np.sin( 6 * np.pi * y ), np.tanh( 40 * np.sin( 2 * np.pi * 2.5 * ( x + 0.3 * y ) ) ), np.random.default_rng( 1 ).random( lat.n ) ]:
    for c in [ 2, 10, 100 ]:
        m = 1 + ( c - 1 ) * 0.5 * ( 1 + f )
        m *= lat.area() / m.sum()
        psi = np.zeros( lat.n )
        for it in range( 12 ): # Newton iterates, kept
            cells = L.reference_cells( lat, psi )
            psis.append( ( psi.copy(), cells.ring ) )
            err = np.abs( cells.area - m ).max()
            if err < 1e-9: break
            Hm = L.hessian( lat.pos, cells ); g = cells.area - m
            d = np.zeros( lat.n ); import scipy.sparse.linalg as spla
            d[ 1: ] = spla.spsolve( Hm[ 1:, 1: ].tocsc(), -g[ 1: ] )
            tau = 1.0
            while True:
                c2 = L.reference_cells( lat, psi + tau * d )
                if c2.area.min() > 0.5 * min( cells.area.min(), m.min() ) and np.abs( c2.area - m ).max() <= ( 1 - tau / 2 ) * err: break
                tau /= 2
            psi = psi + tau * d
# interior: the cell is at least 2 seeds from the box ( no box direction can be active )
col = np.arange( lat.n ) % 15; row = lat.idx[ :, 1 ]
interior = ( col >= 2 ) & ( col <= 12 ) & ( row >= 2 ) & ( row <= 14 )
cnt = {}; cnti = {}
tot = 0
for psi, ring in psis:
    h, _ = K.raw_h( lat, psi )
    ht = K.tighten( h )
    ok = ring <= 2
    cand = h[ :, K.pj ] * K.plam + h[ :, K.pl ] * K.pmu
    tk, hk = ht[ :, K.pk ], h[ :, K.pk ]
    use = ( np.abs( cand - tk ) <= 1e-12 * ( 1 + np.abs( tk ) ) ) & ( cand < hk - 1e-12 * ( 1 + np.abs( hk ) ) ) & ok[ :, None ]
    tot += ok.sum()
    for p, ( k, j, l, _, _ ) in enumerate( K.pairs ):
        a = round( ( ang[ k ] - ang[ j ] ) % 360 / 30 ); b = round( ( ang[ l ] - ang[ k ] ) % 360 / 30 )
        if a > 6: a, b = round( ( ang[ k ] - ang[ l ] ) % 360 / 30 ), round( ( ang[ j ] - ang[ k ] ) % 360 / 30 )
        cnt[ ( a, b ) ] = cnt.get( ( a, b ), 0 ) + int( use[ :, p ].sum() )
        cnti[ ( a, b ) ] = cnti.get( ( a, b ), 0 ) + int( ( use[ :, p ] & interior ).sum() )
print( f"{len( psis )} psi, {tot} cells within ring 2; cells where pair ( a, b ) sets ht_k, summed over the 12 directions:" )
for ab in sorted( cnt ): print( f"  a={ab[ 0 ]} b={ab[ 1 ]} span {ab[ 0 ] + ab[ 1 ]}: {cnt[ ab ]:6d}  ( interior cells: {cnti[ ab ]} )" )

# cells ( within ring 2 ) whose ht changes when only some spans are kept
def span_of( p ):
    k, j, l = K.pairs[ p ][ :3 ]
    a = round( ( ang[ k ] - ang[ j ] ) % 360 / 30 ); b = round( ( ang[ l ] - ang[ k ] ) % 360 / 30 )
    return a + b if a <= 6 else round( ( ang[ k ] - ang[ l ] ) % 360 / 30 ) + round( ( ang[ j ] - ang[ k ] ) % 360 / 30 )
spans = np.array( [ span_of( p ) for p in range( len( K.pairs ) ) ] )
print( "cells whose ht differs from the 120 pairs, by kept spans ( all cells within ring 2 / interior ):" )
for keep in [ ( 2, ), ( 2, 4 ), ( 2, 3, 4 ), ( 2, 4, 5 ) ]:
    sel = np.isin( spans, keep )
    bad = badi = n = ni = 0
    for psi, ring in psis:
        h, _ = K.raw_h( lat, psi )
        ht = K.tighten( h )
        cand = np.where( np.isnan( h[ :, K.pj ] * K.plam + h[ :, K.pl ] * K.pmu ), np.inf, h[ :, K.pj ] * K.plam + h[ :, K.pl ] * K.pmu )
        hs = h.copy()
        for k in range( h.shape[ 1 ] ):
            s = ( K.pk == k ) & sel
            if s.any(): hs[ :, k ] = np.minimum( hs[ :, k ], cand[ :, s ].min( 1 ) )
        diff = ( np.abs( np.where( np.isfinite( ht ), hs - ht, 0 ) ) > 1e-12 * ( 1 + np.abs( np.where( np.isfinite( ht ), ht, 0 ) ) ) ).any( 1 ) & ( ring <= 2 )
        bad += diff.sum(); badi += ( diff & interior ).sum(); n += ( ring <= 2 ).sum(); ni += ( ( ring <= 2 ) & interior ).sum()
    print( f"  spans {keep} ( {sel.sum()} pairs ): {bad / n:.4f} / {badi / ni:.4f}" )
