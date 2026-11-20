"""Optimal quantization of a density on the box, from a square, a hexagonal or a random start.

- `lloyd`: Voronoi cells ( free masses ), x <- barycentres;
- `ccvt`: equal masses ( semi-discrete OT, Newton from psi = 0: the previous psi can leave an empty cell on moved seeds ), x <- barycentres of the power cells.
Per iteration: n E ( E = sum_i int_cell |y - x_i|^2 rho, n ~ the same for all starts ) and the share of cells whose active
neighbours are not all in ring 1 / 2 of their START lattice ( would a fixed stencil from the lattice indices still be exact ).
Usage: python quantize.py algo dens N iters
"""
import os, sys, math, time
import numpy as np
sys.path.insert( 0, __file__.rsplit( "/", 1 )[ 0 ] )
import lattice2d as L

algo = sys.argv[ 1 ] if len( sys.argv ) > 1 else "lloyd"
assert algo in ( "lloyd", "ccvt" ), algo
dname = sys.argv[ 2 ] if len( sys.argv ) > 2 else "gauss"
N = int( sys.argv[ 3 ] ) if len( sys.argv ) > 3 else 32
iters = int( sys.argv[ 4 ] ) if len( sys.argv ) > 4 else 30

W = round( N * math.sqrt( math.sqrt( 3 ) / 2 ) )
R = round( N * N / W )
starts = { "square": L.square( N ), "hex": L.hexagonal( W, R ) }
# random start: the square's labels ( rings meaningless ), positions uniform in the square's box
rnd = L.square( N )
rng = np.random.default_rng( 0 )
x0, x1, y0, y1 = rnd.box
rnd.pos = np.stack( [ rng.uniform( x0, x1, rnd.n ), rng.uniform( y0, y1, rnd.n ) ], 1 )
starts[ "random" ] = rnd


def make_dens( lat ):
    x0, x1, y0, y1 = lat.box
    s = max( x1 - x0, y1 - y0 )
    def f( X, Y ):
        u, v = ( X - x0 ) / s, ( Y - y0 ) / s
        if dname == "gauss": return 0.1 + np.exp( -( ( u - 0.4 ) ** 2 + ( v - 0.55 ) ** 2 ) / ( 2 * 0.18 ** 2 ) )
        if dname == "sines": return 1 + 0.7 * np.sin( 2 * np.pi * 1.5 * u ) * np.sin( 2 * np.pi * v )
        return np.ones_like( X )
    return f, s


print( f"{algo} on {dname}, square {N}x{N} = {N * N}, hex {W}x{R} = {W * R}; columns: it, n E / s^2 x 1e3, frac ring > 1, > 2 ( vs start lattice )" )
res = {}
for name, lat in starts.items():
    dens, s = make_dens( lat )
    pos = lat.pos.copy()
    psi = np.zeros( lat.n )
    t0 = time.time()
    hist = []
    for it in range( iters + 1 ):
        if algo == "lloyd":
            c = L.reference_cells( lat, psi, pos, dens = dens )
        else:
            mass = L.reference_cells( lat, np.zeros( lat.n ), pos, dens = dens ).area.sum()
            psi, c, _ = L.newton( lat, np.full( lat.n, mass / lat.n ), pos = pos, psi0 = None, tol = 1e-5, record = False, dens = dens )
        # energy around the CURRENT seeds: int |y - x_i|^2 rho
        E = c.m2.sum()
        hist.append( ( it, lat.n * E / s ** 4 * 1e3, ( c.ring > 1 ).mean(), ( c.ring > 2 ).mean() ) )
        pos = c.cent.copy()
    res[ name ] = hist
    print( f"{name:7s} ({time.time() - t0:.0f} s): " + "  ".join( f"{h[ 0 ]}:{h[ 1 ]:.4f}/{h[ 2 ]:.2f}/{h[ 3 ]:.2f}" for h in hist if h[ 0 ] in ( 0, 1, 2, 3, 5, 10, 20, 30, 50, 100 ) or h[ 0 ] == iters ), flush = True )
np.save( os.environ.get( "ERRAND_OUT_DIR", "." ) + f"/quantize_{algo}_{dname}_{N}.npy", np.array( [ res[ k ] for k in starts ] ), allow_pickle = True )
