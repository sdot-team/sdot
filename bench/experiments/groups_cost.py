# The cost of `groups.find_groups` on a uniform cloud ( no aggregate ), piece by piece.
#   errand run --env n20-numpy --no-queue python sdot-ffi/bench/experiments/groups_cost.py 1000000 36
import os, sys, time
sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import cases
from sdot import groups as grp
from sdot.SdotPlanNd import exact_duplicates

n = int( sys.argv[ 1 ] ) if len( sys.argv ) > 1 else 1_000_000
T = int( sys.argv[ 2 ] ) if len( sys.argv ) > 2 else 36
P, _, d = cases.case( "uniform2d", n = n, seed = 0 )
P = np.ascontiguousarray( P, dtype = np.float64 )
print( f"n {n}, cpus {os.cpu_count()}" )

def tm( label, f, reps = 2 ):
    best = None
    for _ in range( reps ):
        t = time.perf_counter(); r = f(); dt = time.perf_counter() - t
        best = dt if best is None else min( best, dt )
    print( f"  {label:40s} {best * 1e3:9.1f} ms", flush = True )
    return r

tm( "find_groups", lambda: grp.find_groups( P ) )
tm( "exact_duplicates", lambda: exact_duplicates( P ) )
tree = tm( "cKDTree build", lambda: cKDTree( P ) )
tm( "cKDTree build unbalanced", lambda: cKDTree( P, balanced_tree = False ) )
tm( "cKDTree build unbalanced, not compact", lambda: cKDTree( P, balanced_tree = False, compact_nodes = False ) )
t2 = cKDTree( P, balanced_tree = False, compact_nodes = False )
tm( f"query unbalanced workers={T}", lambda: t2.query( P, k = 2 * d + 2, workers = T ) )
K = 2 * d + 1
for w in ( 1, 8, T, -1 ):
    dist, idx = tm( f"query k={K + 1} workers={w}", lambda: tree.query( P, k = K + 1, workers = w ) )
a = np.repeat( np.arange( n, dtype = np.int64 ), K )
b = idx[ :, 1: ].reshape( -1 ).astype( np.int64 )
l = dist[ :, 1: ].reshape( -1 )
tm( "_connect", lambda: grp._connect( P, tree, a, b, l, K ) )
g0 = csr_matrix( ( np.ones( len( a ), dtype = np.int8 ), b, np.arange( 0, len( a ) + 1, K ) ), shape = ( n, n ) )
nc, comp = connected_components( g0, directed = True, connection = "weak" )
print( "  components of the kNN graph:", nc, np.bincount( np.bincount( comp ) )[ :20 ].nonzero() )
g = csr_matrix( ( np.ones( len( a ), dtype = np.int8 ), b, np.arange( 0, len( a ) + 1, K ) ), shape = ( n, n ) )
tm( "cc weak", lambda: connected_components( g, directed = True, connection = "weak" ) )
tm( "cc strong", lambda: connected_components( g, directed = True, connection = "strong" ) )
g32 = csr_matrix( ( np.ones( len( a ), dtype = np.int8 ), b.astype( np.int32 ), np.arange( 0, len( a ) + 1, K, dtype = np.int32 ) ), shape = ( n, n ) )
tm( "cc weak int32", lambda: connected_components( g32, directed = True, connection = "weak" ) )
tm( "_separated_nodes", lambda: grp._separated_nodes( P, grp.KAPPA ) )
