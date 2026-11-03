"""Local a posteriori certificate: every computed cell contains the true one;
flag cell i when one of its vertices v ( between faces owned by j and l ) is
not also a vertex of the computed cells of j and l. Then flag i and the
neighbor. No flag anywhere => the computed diagram is the exact one."""
import sys
import numpy as np
sys.path.insert( 0, __file__.rsplit( "/", 1 )[ 0 ] )
import ot_stats as O

N = O.N


def certify( K, psi, tol = 1e-9 ):
    c = K.cells( psi )
    n = N * N
    a, b = np.divmod( np.arange( n ), N )
    V = c[ "v" ] + np.stack( [ a, b ], 1 )[ :, None, : ] # absolute vertices, vertex k between edges k and k + 1
    L, own = c[ "L" ], c[ "owner" ]
    D = L.shape[ 1 ]
    flag = np.zeros( n, bool )
    for i in range( n ):
        pos = np.flatnonzero( L[ i ] > tol )
        if len( pos ) < 3:
            flag[ i ] = True
            continue
        for e, f in zip( pos, np.roll( pos, -1 ) ):
            v = V[ i, ( f - 1 ) % D ] # vertex between positive edges e and f
            for j in ( own[ i, e ], own[ i, f ] ):
                if j < 0: continue
                if np.abs( V[ j ] - v ).max( axis = 1 ).min() > 1e-7:
                    flag[ i ] = flag[ j ] = True
    return c, flag


print( f"N={N}  ( certificate on ring-r cells; bad = truly needs > r; miss must be 0 )" )
for kind in [ "sines", "bands", "noise" ]:
    for cst in [ 10, 100, 1000 ]:
        its = O.newton( O.masses( kind, cst ) )
        line = f"{kind:6s} c={cst:<5d}"
        for r in [ 1, 2 ]:
            K = O.Ks[ r ]
            tot_bad = tot_flag = miss = 0
            gl_ok = gl_bad = 0
            for psi, ar, rings in its:
                c, flag = certify( K, psi )
                bad = rings > r
                tot_bad += bad.sum(); tot_flag += flag.sum(); miss += ( bad & ~flag ).sum()
                gl_ok += ( not flag.any() ) and ( not bad.any() )
                gl_bad += ( not flag.any() ) and bad.any()
            m = len( its ) * N * N
            line += f" | r{r}: bad {tot_bad / m:.3f} flag {tot_flag / m:.3f} miss {miss}  iterates certified {gl_ok}/{len( its )} wrongly {gl_bad}"
        print( line )
