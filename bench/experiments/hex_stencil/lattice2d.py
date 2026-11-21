"""Diracs on a 2D lattice ( square or hexagonal ) in a box: exact power cells, Newton on the masses, ring statistics.

Lattices, spacing 1:
- `square( N )`: seeds ( a, b ), box [ -1/2, N - 1/2 ]^2;
- `hexagonal( W, R )`: R rows of W seeds, row r at y = r sqrt( 3 ) / 2, odd rows shifted by 1/2,
  box [ -1/4, W - 1/4 ] x [ -sqrt( 3 ) / 4, R sqrt( 3 ) / 2 - sqrt( 3 ) / 4 ].
Each lattice gives `ring( i, j )`: the graph distance between seeds i and j in the lattice ( Chebyshev on the square,
hexagonal distance on the honeycomb ), so that "ring r" means the same thing as in grid_stencil.

Cells: cell i = { y : |y - x_i|^2 - psi_i <= |y - x_j|^2 - psi_j }, clipped by the box. `reference_cells` clips the box by
every seed that can cut ( KD-tree by distance, stopped by |d|^2 - 2 |d| R + psi_i - psi_max > 0 ), and works for ANY
positions: the lattice only labels the neighbours ( `ring` ), which is how moved seeds are compared with their lattice.
"""
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

S3 = math.sqrt( 3.0 )


class Lattice:
    def __init__( self, kind, pos, idx, box ):
        self.kind = kind
        self.pos = pos         # ( n, 2 )
        self.idx = idx         # ( n, 2 ) integer lattice coordinates ( a, b ) or axial ( q, r )
        self.box = box         # ( x0, x1, y0, y1 )
        self.n = len( pos )

    def ring( self, i, j ):
        d = self.idx[ j ] - self.idx[ i ]
        if self.kind == "square":
            return np.maximum( np.abs( d[ ..., 0 ] ), np.abs( d[ ..., 1 ] ) )
        dq, dr = d[ ..., 0 ], d[ ..., 1 ]
        return ( np.abs( dq ) + np.abs( dr ) + np.abs( dq + dr ) ) // 2

    def unit_coords( self ):
        x0, x1, y0, y1 = self.box
        return ( self.pos[ :, 0 ] - x0 ) / ( x1 - x0 ), ( self.pos[ :, 1 ] - y0 ) / ( y1 - y0 )

    def area( self ):
        x0, x1, y0, y1 = self.box
        return ( x1 - x0 ) * ( y1 - y0 )


def square( N ):
    A, B = np.meshgrid( np.arange( N ), np.arange( N ), indexing = "ij" )
    pos = np.stack( [ A.ravel(), B.ravel() ], 1 ).astype( float )
    return Lattice( "square", pos, np.stack( [ A.ravel(), B.ravel() ], 1 ), ( -0.5, N - 0.5, -0.5, N - 0.5 ) )


def hexagonal( W, R ):
    rr, cc = np.meshgrid( np.arange( R ), np.arange( W ), indexing = "ij" )
    rr, cc = rr.ravel(), cc.ravel()
    pos = np.stack( [ cc + 0.5 * ( rr % 2 ), rr * S3 / 2 ], 1 ).astype( float )
    q = cc - ( rr - ( rr % 2 ) ) // 2 # offset rows -> axial coordinates ( odd rows shifted right )
    return Lattice( "hex", pos, np.stack( [ q, rr ], 1 ), ( -0.25, W - 0.25, -S3 / 4, R * S3 / 2 - S3 / 4 ) )


# ------------------------------------------------------------------ exact cells
def _clip( poly, labels, n, c, lab ):
    """ clip polygon by n.y <= c; edge e goes from poly[ e ] to poly[ e + 1 ] with label labels[ e ] """
    m = len( poly )
    s = [ n[ 0 ] * p[ 0 ] + n[ 1 ] * p[ 1 ] - c for p in poly ]
    if max( s ) <= 0:
        return poly, labels
    out, outl = [], []
    for e in range( m ):
        p, q = poly[ e ], poly[ ( e + 1 ) % m ]
        sp_, sq = s[ e ], s[ ( e + 1 ) % m ]
        if sp_ <= 0:
            out.append( p )
            outl.append( labels[ e ] )
            if sq > 0:
                t = sp_ / ( sp_ - sq )
                out.append( ( p[ 0 ] + t * ( q[ 0 ] - p[ 0 ] ), p[ 1 ] + t * ( q[ 1 ] - p[ 1 ] ) ) )
                outl.append( lab )
        elif sq <= 0:
            t = sp_ / ( sp_ - sq )
            out.append( ( p[ 0 ] + t * ( q[ 0 ] - p[ 0 ] ), p[ 1 ] + t * ( q[ 1 ] - p[ 1 ] ) ) )
            outl.append( labels[ e ] )
    return out, outl


class Cells:
    """ area, centroid, second moment ( int |y - x_i|^2 ), faces { j: length }, max ring of the active neighbours """
    pass


# Dunavant degree 5, 7 points: barycentric coordinates and weights ( sum 1 )
_DA = np.array( [ [ 1 / 3, 1 / 3, 1 / 3 ],
                  [ 0.797426985353087, 0.101286507323456, 0.101286507323456 ], [ 0.101286507323456, 0.797426985353087, 0.101286507323456 ], [ 0.101286507323456, 0.101286507323456, 0.797426985353087 ],
                  [ 0.059715871789770, 0.470142064105115, 0.470142064105115 ], [ 0.470142064105115, 0.059715871789770, 0.470142064105115 ], [ 0.470142064105115, 0.470142064105115, 0.059715871789770 ] ] )
_DW = np.array( [ 0.225 ] + [ 0.125939180544827 ] * 3 + [ 0.132394152788506 ] * 3 )


def _dens_poly( dens, px, py, poly, labels, faces ):
    """ mass, first moment ( relative to the seed ), second moment of the density on poly ( relative coordinates ); adds
        the face integrals of the density to faces ( Simpson ) """
    P = np.array( poly )
    m = len( P )
    tri = np.stack( [ np.zeros( ( m, 2 ) ), P, np.roll( P, -1, axis = 0 ) ], 1 ) # ( m, 3, 2 ) fan from the seed
    det = tri[ :, 1, 0 ] * tri[ :, 2, 1 ] - tri[ :, 1, 1 ] * tri[ :, 2, 0 ]
    q = np.einsum( "pk,tkd->tpd", _DA, tri ) # ( m, 7, 2 )
    f = dens( q[ ..., 0 ] + px, q[ ..., 1 ] + py ) * _DW[ None, : ] * ( 0.5 * det )[ :, None ]
    mass = f.sum()
    mom = np.einsum( "tp,tpd->d", f, q )
    m2 = ( f * ( q ** 2 ).sum( -1 ) ).sum()
    for e in range( m ):
        if labels[ e ] >= 0 and labels[ e ] in faces:
            a, b = P[ e ], P[ ( e + 1 ) % m ]
            c = 0.5 * ( a + b )
            fv = dens( np.array( [ a[ 0 ], c[ 0 ], b[ 0 ] ] ) + px, np.array( [ a[ 1 ], c[ 1 ], b[ 1 ] ] ) + py )
            faces[ labels[ e ] ] += ( fv[ 0 ] + 4 * fv[ 1 ] + fv[ 2 ] ) / 6 * math.hypot( *( b - a ) ) - math.hypot( *( b - a ) )
    return mass, mom, m2


def reference_cells( lat, psi, pos = None, tol = 1e-10, dens = None ):
    """ with `dens` ( vectorized f( x, y ) ): area, cent, m2 are the mass, barycentre, second moment of the density and
        faces hold the integrals of the density on the faces """
    pos = lat.pos if pos is None else pos
    n = len( pos )
    x0, x1, y0, y1 = lat.box
    tree = cKDTree( pos )
    pmax = psi.max()
    # ( all the neighbours at once below 6000 seeds: a second, larger query does not return ties in the same order, and the
    # walk skipped a seed at the distance where the first query stopped )
    K = n if n <= 6000 else 48
    dd, jj = tree.query( pos, k = K )
    out = Cells()
    out.area = np.zeros( n )
    out.cent = np.zeros( ( n, 2 ) )
    out.m2 = np.zeros( n )
    out.faces = []
    out.ring = np.zeros( n, dtype = int )
    for i in range( n ):
        px, py = pos[ i ]
        poly = [ ( x0 - px, y0 - py ), ( x1 - px, y0 - py ), ( x1 - px, y1 - py ), ( x0 - px, y1 - py ) ]
        labels = [ -1, -1, -1, -1 ]
        pi = psi[ i ]
        cand_d, cand_j, k = dd[ i ], jj[ i ], K
        t = 1
        while True:
            if t >= len( cand_j ):
                if k >= n: break
                k = min( n, 2 * k )
                cand_d, cand_j = tree.query( pos[ i ], k = k )
                continue
            j = cand_j[ t ]; nk = cand_d[ t ]; t += 1
            if not poly: break
            R = max( math.hypot( *p ) for p in poly )
            if nk >= R and nk * nk - 2 * nk * R + pi - pmax > 0:
                break
            dx, dy = pos[ j, 0 ] - px, pos[ j, 1 ] - py
            c = ( dx * dx + dy * dy + pi - psi[ j ] ) / 2
            poly, labels = _clip( poly, labels, ( dx, dy ), c, j )
        m = len( poly )
        ar = 0.0; cx = 0.0; cy = 0.0; m2 = 0.0
        f = {}
        for e in range( m ):
            p, q = poly[ e ], poly[ ( e + 1 ) % m ]
            cr = p[ 0 ] * q[ 1 ] - p[ 1 ] * q[ 0 ]
            ar += cr
            cx += ( p[ 0 ] + q[ 0 ] ) * cr
            cy += ( p[ 1 ] + q[ 1 ] ) * cr
            m2 += cr * ( p[ 0 ] ** 2 + p[ 0 ] * q[ 0 ] + q[ 0 ] ** 2 + p[ 1 ] ** 2 + p[ 1 ] * q[ 1 ] + q[ 1 ] ** 2 )
            ln = math.hypot( q[ 0 ] - p[ 0 ], q[ 1 ] - p[ 1 ] )
            if labels[ e ] >= 0 and ln > tol:
                f[ labels[ e ] ] = f.get( labels[ e ], 0 ) + ln
        ar /= 2
        out.area[ i ] = ar
        if ar > 0:
            out.cent[ i ] = ( px + cx / ( 6 * ar ), py + cy / ( 6 * ar ) )
        else:
            out.cent[ i ] = ( px, py )
        out.m2[ i ] = m2 / 12
        if dens is not None and len( poly ) >= 3:
            ms, mo, m2d = _dens_poly( dens, px, py, poly, labels, f )
            out.area[ i ] = ms
            out.cent[ i ] = ( px + mo[ 0 ] / ms, py + mo[ 1 ] / ms ) if ms > 0 else ( px, py )
            out.m2[ i ] = m2d
        out.faces.append( f )
        out.ring[ i ] = max( [ int( lat.ring( i, j ) ) for j in f ], default = 0 )
    return out


def hessian( pos, cells ):
    n = len( pos )
    rows, cols, vals = [], [], []
    for i, f in enumerate( cells.faces ):
        for j, L in f.items():
            w = L / ( 2 * math.hypot( *( pos[ j ] - pos[ i ] ) ) )
            rows += [ i, i ]; cols += [ j, i ]; vals += [ -w, w ]
    return sp.csr_matrix( ( vals, ( rows, cols ) ), shape = ( n, n ) )


def newton( lat, m, pos = None, psi0 = None, tol = 1e-9, maxit = 60, record = True, dens = None ):
    """ psi such that the cells have the masses m ( sum m = box area ); returns ( psi, cells, iterates ) """
    pos = lat.pos if pos is None else pos
    n = len( pos )
    psi = np.zeros( n ) if psi0 is None else psi0.copy()
    cells = reference_cells( lat, psi, pos, dens = dens )
    iterates = []
    for it in range( maxit ):
        if record: iterates.append( cells.ring.copy() )
        err = np.abs( cells.area - m ).max()
        if err < tol * m.max(): break
        H = hessian( pos, cells )
        g = cells.area - m
        d = np.zeros( n )
        d[ 1: ] = spla.spsolve( H[ 1:, 1: ].tocsc(), -g[ 1: ] )
        eps0 = 0.5 * min( cells.area.min(), m.min() )
        tau = 1.0
        while True:
            p2 = psi + tau * d
            c2 = reference_cells( lat, p2, pos, dens = dens )
            if c2.area.min() > eps0 and np.abs( c2.area - m ).max() <= ( 1 - tau / 2 ) * err: break
            tau /= 2
            if tau < 1e-4: # quadrature floor ( masses by Dunavant, hessian by Simpson on the faces: ~1e-5 relative )
                if err < 1e-3 * m.max(): return psi, cells, iterates
                raise RuntimeError( f"newton: line search failed at it {it}, err {err:.2e}" )
        psi, cells = p2, c2
    return psi, cells, iterates
