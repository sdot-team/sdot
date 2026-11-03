"""Ring statistics and a priori filter on psi from real semi-discrete OT solves.

Source = Lebesgue on the box, target = Diracs on the grid with masses from an
image-like field of contrast c. Newton (KMT damping) on the exact cells; at
every iterate we record which cells need neighbors beyond ring r, and whether
the filter flags them.
"""
import sys, math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import maximum_filter
sys.path.insert( 0, __file__.rsplit( "/", 1 )[ 0 ] )
from stencil2d import StencilKernel, reference_cells

N = int( sys.argv[ 1 ] ) if len( sys.argv ) > 1 else 32
A, B = np.meshgrid( np.arange( N ), np.arange( N ), indexing = "ij" )
x, y = ( A + 0.5 ) / N, ( B + 0.5 ) / N
rng = np.random.default_rng( 1 )


def field( kind ):
    if kind == "sines": return 0.5 * ( 1 + np.sin( 2 * np.pi * 2 * x ) * np.sin( 2 * np.pi * 3 * y ) )
    if kind == "bands": return 0.5 * ( 1 + np.tanh( 40 * np.sin( 2 * np.pi * 2.5 * ( x + 0.3 * y ) ) ) )
    if kind == "noise": return rng.random( ( N, N ) )


def masses( kind, c ):
    m = 1 + ( c - 1 ) * field( kind )
    return m * N * N / m.sum()


def newton( m, tol = 1e-9, maxit = 60 ):
    psi = np.zeros( ( N, N ) )
    iterates = []
    ar, faces, rings = reference_cells( psi )
    for it in range( maxit ):
        iterates.append( ( psi.copy(), ar, rings ) )
        err = np.abs( ar - m.ravel() ).max()
        if err < tol: break
        rows, cols, vals = [], [], []
        for i, f in enumerate( faces ):
            for j, L in f.items():
                k = math.hypot( j // N - i // N, j % N - i % N )
                w = L / ( 2 * k )
                rows += [ i, i ]; cols += [ j, i ]; vals += [ -w, w ]
        H = sp.csr_matrix( ( vals, ( rows, cols ) ), shape = ( N * N, N * N ) )
        g = ar - m.ravel()
        d = np.zeros( N * N )
        d[ 1: ] = spla.spsolve( H[ 1:, 1: ].tocsc(), -g[ 1: ] )
        eps0 = 0.5 * min( ar.min(), m.min() )
        tau = 1.0
        while True:
            p2 = psi + tau * d.reshape( N, N )
            a2, f2, r2 = reference_cells( p2 )
            if a2.min() > eps0 and np.abs( a2 - m.ravel() ).max() <= ( 1 - tau / 2 ) * err: break
            tau /= 2
        psi, ar, faces, rings = p2, a2, f2, r2
    return iterates


def filter_safe( K, psi, c, S = None ):
    """ cell i provably needs no neighbor beyond ring r: for every shell s > r,
        s >= R_i and s^2 - 2 s R_i + psi_i - max_{square radius s} psi > 0 """
    r = K.r
    S = S or r + 3
    R = c[ "R" ]
    p = psi.ravel()
    safe = np.ones( N * N, dtype = bool )
    for s in range( r + 1, S + 1 ):
        M = maximum_filter( psi, size = 2 * s + 1, mode = "nearest" ).ravel()
        safe &= ( s >= R ) & ( s * s - 2 * s * R + p - M > 0 )
    s = S + 1
    safe &= ( s >= R ) & ( s * s - 2 * s * R + p - psi.max() > 0 )
    return safe


if __name__ != "__main__":
    Ks = { r: StencilKernel( r ) for r in [ 1, 2 ] }
else:
  Ks = { r: StencilKernel( r ) for r in [ 1, 2 ] }
  print( f"N={N}" )
  print( f"{'case':14s} {'its':>3s} | {'final: ring hist':24s} | worst iterate beyond r1 / r2 | filter r2 at solution: flagged / truly bad" )
  for kind in [ "sines", "bands", "noise" ]:
      for c in [ 2, 10, 100, 1000 ]:
          its = newton( masses( kind, c ) )
          psi, ar, rings = its[ -1 ]
          hist = np.bincount( rings, minlength = 5 ).tolist()
          worst1 = max( ( rr > 1 ).mean() for _, _, rr in its )
          worst2 = max( ( rr > 2 ).mean() for _, _, rr in its )
          K = Ks[ 2 ]
          cc = K.cells( psi )
          safe = filter_safe( K, psi, cc )
          bad = rings > 2
          miss = ( safe & bad ).sum() # must be 0
          print( f"{kind:6s} c={c:<5d} {len( its ):3d} | {str( hist ):24s} | {worst1:6.3f} / {worst2:6.3f}"
                 f"          | {( ~safe ).mean():6.3f} / {bad.mean():6.3f}  miss={miss}" )
