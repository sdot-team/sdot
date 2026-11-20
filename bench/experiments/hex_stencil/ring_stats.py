"""Square vs hexagonal lattice, ~ the same number of seeds: which rings the exact cells need along real OT solves.

Source = Lebesgue on the box, target = the lattice diracs with masses from an image-like field of contrast c ( the
cases of grid_stencil/proto/ot_stats.py ). Newton ( damped ) on exact cells; every iterate is recorded.
Usage: python ring_stats.py [ N ]   ( square N x N, hexagonal of as many seeds in a box as square )
"""
import sys, math, time
import numpy as np
sys.path.insert( 0, __file__.rsplit( "/", 1 )[ 0 ] )
import lattice2d as L

N = int( sys.argv[ 1 ] ) if len( sys.argv ) > 1 else 32
# quick by default: no c = 1000, no sharp bands at c >= 100 ( 60 Newton steps of pure python: 20 min each at N = 32 )
CS = [ int( c ) for c in ( sys.argv[ 2 ] if len( sys.argv ) > 2 else "2,10,100" ).split( "," ) ]
MAXIT = int( sys.argv[ 3 ] ) if len( sys.argv ) > 3 else 20
W = round( N * math.sqrt( math.sqrt( 3 ) / 2 ) )
R = round( N * N / W )
lats = [ L.square( N ), L.hexagonal( W, R ) ]
rng = np.random.default_rng( 1 )
noise = {}


def field( lat, kind ):
    x, y = lat.unit_coords()
    if kind == "sines": return 0.5 * ( 1 + np.sin( 2 * np.pi * 2 * x ) * np.sin( 2 * np.pi * 3 * y ) )
    if kind == "bands": return 0.5 * ( 1 + np.tanh( 40 * np.sin( 2 * np.pi * 2.5 * ( x + 0.3 * y ) ) ) )
    if kind == "noise": return np.random.default_rng( 1 ).random( lat.n )


def masses( lat, kind, c ):
    m = 1 + ( c - 1 ) * field( lat, kind )
    return m * lat.area() / m.sum()


print( f"square {N}x{N} = {N * N} seeds, hexagonal {W}x{R} = {W * R} seeds" )
print( f"{'case':14s} {'lattice':7s} {'its':>3s} | {'final ring hist':28s} | worst iterate: frac > r1 / > r2 / max ring" )
for kind in [ "sines", "bands", "noise" ]:
    for c in CS:
        if kind == "bands" and c >= 100 and MAXIT <= 20: continue
        for lat in lats:
            t0 = time.time()
            psi, cells, its = L.newton( lat, masses( lat, kind, c ), maxit = MAXIT )
            hist = np.bincount( cells.ring, minlength = 4 ).tolist()
            w1 = max( ( r > 1 ).mean() for r in its )
            w2 = max( ( r > 2 ).mean() for r in its )
            mx = max( r.max() for r in its )
            print( f"{kind:6s} c={c:<5d} {lat.kind:7s} {len( its ):3d} | {str( hist ):28s} | {w1:6.3f} / {w2:6.3f} / {mx}"
                   f"   ({time.time() - t0:.0f} s)", flush = True )
