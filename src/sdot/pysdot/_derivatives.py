"""`der_centroids_and_integrals_wrt_weight_and_positions`, by the automatic differentiation of sdot.

The centroid and the integral of cell `i` depend only on the seeds of its closed neighborhood `N[ i ]`. Cells whose
neighborhoods do not meet ( a distance-2 coloring of the Laguerre graph ) can then share a reverse sweep: the cotangent that
sums their rows gives a gradient whose pieces do not overlap. `d + 1` sweeps per color, a few tens of colors in 2D.
"""
import numpy as np

# the frameworks tried for the automatic differentiation, in this order
FRAMEWORKS = ( "jax", "torch" )


def der_centroids_and_integrals( cpd ):
    from .PowerDiagram import DerResult
    n, d = cpd.positions.shape
    nbrs = _neighbors( cpd )
    colors = _distance2_coloring( nbrs )
    mass = cpd._mass()
    vjp_of, values = _vjp( cpd )               # values: [ n, d + 1 ] ( centroids, integral )

    # one sweep per ( color, row of the block ): the columns of each cell are its neighborhood
    blocks = {}                                # ( i, j ) -> [ d + 1, d + 1 ]
    for c in range( colors.max() + 1 ):
        members = np.nonzero( colors == c )[ 0 ]
        for r in range( d + 1 ):
            seed = np.zeros( ( n, d + 1 ) )
            seed[ members, r ] = 1
            gp, gw = vjp_of( seed )            # [ n, d ], [ n ]
            for i in members:
                for j in nbrs[ i ]:
                    b = blocks.setdefault( ( i, j ), np.zeros( ( d + 1, d + 1 ) ) )
                    b[ r, :d ] = gp[ j ]
                    b[ r, d ] = gw[ j ]

    # the CSR matrix, rows `( d + 1 ) i + r`, columns `( d + 1 ) j + c`
    rows, cols, vals = [], [], []
    for ( i, j ), b in blocks.items():
        for r in range( d + 1 ):
            for c in range( d + 1 ):
                rows.append( ( d + 1 ) * i + r )
                cols.append( ( d + 1 ) * j + c )
                vals.append( b[ r, c ] * ( mass if r == d else 1.0 ) )
    from scipy.sparse import csr_matrix
    N = n * ( d + 1 )
    M = csr_matrix( ( vals, ( rows, cols ) ), shape = ( N, N ) )
    M.sum_duplicates()
    M.sort_indices()
    v = values.copy()
    v[ :, d ] *= mass
    return DerResult( M.indptr.astype( np.uint64 ), M.indices.astype( np.uint64 ), M.data.astype( float ), v.reshape( -1 ), 0 )


def _neighbors( cpd ):
    """the closed neighborhood of each cell ( itself and the cells it shares a facet with )"""
    n = cpd.positions.shape[ 0 ]
    rows, cols, _ = cpd._jacobian_entries()
    res = [ { i } for i in range( n ) ]
    for i, j in zip( rows, cols ):
        res[ i ].add( int( j ) )
        res[ int( j ) ].add( int( i ) )
    return [ np.array( sorted( s ) ) for s in res ]


def _distance2_coloring( nbrs ):
    """a color per cell, two cells whose neighborhoods meet having different ones ( greedy )"""
    n = len( nbrs )
    colors = np.full( n, -1 )
    for i in range( n ):
        taken = set()
        for j in nbrs[ i ]:
            for k in nbrs[ j ]:
                if colors[ k ] >= 0:
                    taken.add( colors[ k ] )
        c = 0
        while c in taken:
            c += 1
        colors[ i ] = c
    return colors


def _vjp( cpd ):
    """`( vjp, values )`: the centroids and the normalized integrals `[ n, d + 1 ]` as a function of the positions and the weights,
    and the product of a cotangent `[ n, d + 1 ]` with their jacobian, `( d positions [ n, d ], d weights [ n ] )`"""
    import sdot
    from .radial_funcs import sdot_radial_args
    pos0, w0 = cpd.positions, cpd.weights if cpd.weights is not None else np.ones( cpd.positions.shape[ 0 ] )
    d = pos0.shape[ 1 ]
    sink_weight, radial = sdot_radial_args( cpd.radial_func )
    dist = cpd.domain.to_sdot()
    per = cpd.sdot_periodicity()

    n = pos0.shape[ 0 ]
    cpd.sdot_diagram()                                    # ( decides whether the replications are a periodicity or explicit images )
    if cpd._images:
        per = None

    def values_of( pos, w, stack, where, concatenate ):
        pe, we = cpd.extended_seeds( pos, w, concatenate )
        pd = sdot.PowerDiagram( pe, we, distribution = dist, kernel_dtype = "FP64", periodicity = per, sink_weight = sink_weight, radial_func = radial )
        m = pd.measures[ :n ]
        safe = where( m > 0, m, 1.0 )
        cols = [ where( m > 0, pd.moment( k )[ :n ] / safe, pos[ :, k ] ) for k in range( d ) ]
        return stack( cols + [ m ], 1 )

    for fw in FRAMEWORKS:
        try:
            if fw == "jax":
                return _vjp_jax( values_of, pos0, w0 )
            if fw == "torch":
                return _vjp_torch( values_of, pos0, w0 )
        except ImportError:
            continue
    raise ImportError( "der_centroids_and_integrals_wrt_weight_and_positions differentiates through sdot: install jax or torch" )


def _vjp_jax( values_of, pos0, w0 ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    import jax.numpy as jnp
    f = lambda p, w: values_of( p, w, jnp.stack, jnp.where, jnp.concatenate )
    out, vjp = jax.vjp( f, jnp.asarray( pos0 ), jnp.asarray( w0 ) )
    return ( lambda seed: tuple( np.asarray( g ) for g in vjp( jnp.asarray( seed ) ) ) ), np.asarray( out )


def _vjp_torch( values_of, pos0, w0 ):
    import torch
    p = torch.tensor( pos0, dtype = torch.float64, requires_grad = True )
    w = torch.tensor( w0, dtype = torch.float64, requires_grad = True )
    out = values_of( p, w, torch.stack, torch.where, torch.cat )

    def vjp( seed ):
        gp, gw = torch.autograd.grad( out, ( p, w ), torch.as_tensor( seed, dtype = out.dtype ), retain_graph = True, allow_unused = True )
        return ( np.zeros( pos0.shape ) if gp is None else gp.detach().numpy(),
                 np.zeros( w0.shape ) if gw is None else gw.detach().numpy() )
    return vjp, out.detach().numpy()
