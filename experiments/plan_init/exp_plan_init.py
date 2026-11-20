"""PLAN-BASED STARTS for regular diracs -> contrasted image ( prototype, not committed ).

The target: an image on the unit square ( bands at an angle, a contrasted sine, a white star on black ), a floor of 1e-3 of the
max. The source: N x N diracs at the centres of a regular grid, equal masses. Weights `w`: the cell of `i` is where
`|x - g_i|^2 - w_i` wins ( sdot's convention, `w` = psi ).

Starts compared on the fine grid ( coarse grid = N / 2, solved by sdot with its defaults ):

    zero      w = 0
    cubic     coarse w interpolated: cubic spline of w - |g|^2, |g|^2 added back
    local     plan-based: each coarse cell split among its 4 children by a Knothe split on the real image ( x cut at half
              mass, then each half along y ), sibling faces read from the cuts, cousin faces = the coarse face; least squares
              on all the grid edges ( a Poisson solve )
    global    global Knothe map: column faces at the quantiles of the x marginal, row faces at the conditional quantiles in
              each column strip; least squares the same way ( no coarse solve )

A face between neighbours i, j = i + e_k is `x_k = s`  <=>  w_j - w_i = |g_j|^2 - |g_i|^2 - 2 h s.

Each start is repaired ( EMPTY cells, under 1e-4 of their mass, get their weight raised, the step doubling ), then goes to sdot's Newton ( general path, `SumOfDiracs`, `continuation = "never"` ), with `keep_start`: the solver's own warm
repair runs, but it may not swap the start for the Voronoi. A second row lets sdot swap ( its default behaviour ).

    exp_plan_init.py run [N...]      the table ( results.json + results.md in the output dir )
    exp_plan_init.py figure [N]      the cells of each start on the bands image, next to the converged ones
"""
import json
import os
import sys
import time

import numpy as np

P_IMG  = 1024     # image resolution
SUB    = 2        # sub-samples per pixel and per axis for the rasterized plans
FLOOR  = 1e-3
OUT    = os.environ.get( "ERRAND_OUT_DIR", os.path.dirname( os.path.abspath( __file__ ) ) )


# -- images ---------------------------------------------------------------------------------------

def image_values( name, P = P_IMG ):
    """`values[ i, j ]` at x = ( i + 1/2 ) / P, y = ( j + 1/2 ) / P ( first axis = x, sdot's convention )"""
    t = ( np.arange( P ) + 0.5 ) / P
    x, y = np.meshgrid( t, t, indexing = "ij" )
    if name == "bands":
        th = np.deg2rad( 30 )
        v = ( np.sin( 2 * np.pi * ( x * np.cos( th ) + y * np.sin( th ) ) / 0.25 ) > 0 ).astype( float )
    elif name == "sine":
        v = ( 0.5 + 0.5 * np.sin( 6 * np.pi * x ) * np.sin( 4 * np.pi * y ) ) ** 4
    elif name == "star":
        from matplotlib.path import Path
        k = np.arange( 10 )
        r = np.where( k % 2 == 0, 0.42, 0.17 )
        a = np.pi / 2 + k * np.pi / 5
        poly = np.stack( [ 0.5 + r * np.cos( a ), 0.5 + r * np.sin( a ) ], axis = 1 )
        v = Path( poly ).contains_points( np.stack( [ x.ravel(), y.ravel() ], axis = 1 ) ).reshape( P, P ).astype( float )
    else:
        raise ValueError( name )
    v = v / v.max()
    return np.maximum( v, FLOOR )


def samples( values, sub = SUB ):
    """the image as weighted points ( sub x sub per pixel ), total mass 1"""
    P = values.shape[ 0 ]
    t = ( np.arange( P * sub ) + 0.5 ) / ( P * sub )
    x, y = np.meshgrid( t, t, indexing = "ij" )
    m = np.repeat( np.repeat( values, sub, axis = 0 ), sub, axis = 1 )
    m = m / m.sum()
    return np.stack( [ x.ravel(), y.ravel() ], axis = 1 ), m.ravel()


# -- grid and power diagrams ------------------------------------------------------------------------

def grid_positions( N ):
    t = ( np.arange( N ) + 0.5 ) / N
    x, y = np.meshgrid( t, t, indexing = "ij" )
    return np.stack( [ x.ravel(), y.ravel() ], axis = 1 )


def labels( pts, sites, w ):
    """argmin_i |x - g_i|^2 - w_i, by a nearest neighbour in 3D ( the lifted sites )"""
    from scipy.spatial import cKDTree
    C = w.max()
    lifted = np.concatenate( [ sites, np.sqrt( C - w )[ :, None ] ], axis = 1 )
    q = np.concatenate( [ pts, np.zeros( ( len( pts ), 1 ) ) ], axis = 1 )
    return cKDTree( lifted ).query( q, workers = -1 )[ 1 ]


def arr( x ):
    from loom.tensor.host import to_host
    return np.asarray( to_host( getattr( x, "raw", x ) ), dtype = float )


def masses( img, pos, w ):
    import sdot
    return arr( sdot.PowerDiagram( pos, w, distribution = img ).measures )


def start_stats( m, n ):
    r = m * n
    return dict( min_rel_mass = float( r.min() ), nb_empty = int( ( r < 1e-9 ).sum() ), nb_small = int( ( r < 0.1 ).sum() ),
                 max_rel_err = float( np.abs( r - 1 ).max() ), l1_rel_err = float( np.abs( r - 1 ).mean() ) )


def repair( img, pos, w, n, h, frac = 1e-4, max_rounds = 200 ):
    """the EMPTY cells ( under `frac` of their mass ) get their weight raised, the step doubling while they stay empty -- by FRONTS:
    only the empty cells with a non-empty grid neighbour move in a round, so that a cluster of empty cells is eaten from its border
    ( raising a whole cluster together changes nothing inside it: only differences of weights count )"""
    N = int( round( np.sqrt( n ) ) )
    w = w.copy()
    step = np.full( len( w ), 0.05 * h * h )
    for k in range( max_rounds ):
        m = masses( img, pos, w )
        bad = ( m * n < frac ).reshape( N, N )
        if not bad.any():
            return w, k
        ok = ~bad
        front = np.zeros_like( bad )
        front[ 1:, : ] |= ok[ :-1, : ]; front[ :-1, : ] |= ok[ 1:, : ]
        front[ :, 1: ] |= ok[ :, :-1 ]; front[ :, :-1 ] |= ok[ :, 1: ]
        front = ( front & bad ).ravel()
        if not front.any():
            front = bad.ravel()
        w[ front ] += step[ front ]
        step[ front ] *= 2
        step[ ~bad.ravel() ] = 0.05 * h * h            # a cell that came back starts again from a small step
    return w, max_rounds


# -- the starts ------------------------------------------------------------------------------------

def quantile( v, m, q ):
    """the position where the cumulated mass of the points ( sorted by `v` ) reaches the fraction q"""
    o = np.argsort( v, kind = "stable" )
    c = np.cumsum( m[ o ] )
    if c[ -1 ] <= 0:
        return float( np.median( v ) )
    return float( np.interp( q * c[ -1 ], c, v[ o ] ) )


def least_squares_from_faces( N, h, sx, sy ):
    """`sx[ i, j ]`: the face x = s between ( i, j ) and ( i + 1, j ), `[ N - 1, N ]`; `sy[ i, j ]`: y = s between ( i, j ) and
    ( i, j + 1 ), `[ N, N - 1 ]`. Weights by least squares on all the edges ( one pinned )."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    g = grid_positions( N )
    sq = ( g ** 2 ).sum( 1 ).reshape( N, N )
    idx = np.arange( N * N ).reshape( N, N )
    a = np.concatenate( [ idx[ :-1, : ].ravel(), idx[ :, :-1 ].ravel() ] )
    b = np.concatenate( [ idx[ 1:, : ].ravel(), idx[ :, 1: ].ravel() ] )
    rhs = np.concatenate( [ ( sq[ 1:, : ] - sq[ :-1, : ] - 2 * h * sx ).ravel(), ( sq[ :, 1: ] - sq[ :, :-1 ] - 2 * h * sy ).ravel() ] )
    E = len( a )
    B = sp.csr_matrix( ( np.concatenate( [ np.ones( E ), -np.ones( E ) ] ), ( np.concatenate( [ np.arange( E ) ] * 2 ), np.concatenate( [ b, a ] ) ) ),
                       shape = ( E, N * N ) )
    L = ( B.T @ B ).tolil()
    L[ 0, 0 ] += 1.0                                  # pin the constant
    w = spl.spsolve( L.tocsc(), B.T @ rhs )
    resid = B @ w - rhs
    return w - w.mean(), float( np.sqrt( ( resid ** 2 ).mean() ) ) / ( 2 * h )   # rms face displacement


def start_cubic( N, Nc, wc ):
    from scipy.interpolate import RectBivariateSpline
    tc = ( np.arange( Nc ) + 0.5 ) / Nc
    gc = grid_positions( Nc )
    f = RectBivariateSpline( tc, tc, ( wc - ( gc ** 2 ).sum( 1 ) ).reshape( Nc, Nc ), bbox = [ 0, 1, 0, 1 ], kx = 3, ky = 3 )
    t = ( np.arange( N ) + 0.5 ) / N
    g = grid_positions( N )
    return f( t, t ).ravel() + ( g ** 2 ).sum( 1 )


def start_local( N, Nc, wc, pts, pm ):
    """each coarse cell split among its 4 children by a Knothe split of the image inside it"""
    h, H = 1.0 / N, 1.0 / Nc
    gc = grid_positions( Nc )
    lab = labels( pts, gc, wc )
    order = np.argsort( lab, kind = "stable" )
    bounds = np.searchsorted( lab[ order ], np.arange( Nc * Nc + 1 ) )
    cut_x = np.zeros( Nc * Nc )
    cut_y = np.zeros( ( Nc * Nc, 2 ) )
    for I in range( Nc * Nc ):
        sel = order[ bounds[ I ]:bounds[ I + 1 ] ]
        if len( sel ) < 4:                              # ( an empty coarse cell: the cross through its dirac )
            cut_x[ I ], cut_y[ I ] = gc[ I, 0 ], gc[ I, 1 ]
            continue
        x, y, m = pts[ sel, 0 ], pts[ sel, 1 ], pm[ sel ]
        cut_x[ I ] = quantile( x, m, 0.5 )
        for a, side in enumerate( [ x <= cut_x[ I ], x > cut_x[ I ] ] ):
            cut_y[ I, a ] = quantile( y[ side ], m[ side ], 0.5 ) if side.sum() >= 2 else gc[ I, 1 ]
    cut_x, cut_y = cut_x.reshape( Nc, Nc ), cut_y.reshape( Nc, Nc, 2 )
    # the coarse faces ( for the cousins )
    sqc = ( gc ** 2 ).sum( 1 ).reshape( Nc, Nc )
    W = wc.reshape( Nc, Nc )
    face_x = ( sqc[ 1:, : ] - sqc[ :-1, : ] - ( W[ 1:, : ] - W[ :-1, : ] ) ) / ( 2 * H )   # [ Nc - 1, Nc ]
    face_y = ( sqc[ :, 1: ] - sqc[ :, :-1 ] - ( W[ :, 1: ] - W[ :, :-1 ] ) ) / ( 2 * H )   # [ Nc, Nc - 1 ]
    i = np.arange( N - 1 )[ :, None ]
    j = np.arange( N )[ None, : ]
    sib = ( i % 2 == 0 )
    sx = np.where( sib, cut_x[ i // 2, j // 2 ], face_x[ np.minimum( i // 2, Nc - 2 ), j // 2 ] )
    i = np.arange( N )[ :, None ]
    j = np.arange( N - 1 )[ None, : ]
    sib = ( j % 2 == 0 )
    sy = np.where( sib, cut_y[ i // 2, j // 2, i % 2 ], face_y[ i // 2, np.minimum( j // 2, Nc - 2 ) ] )
    return least_squares_from_faces( N, h, sx, sy )


def start_global( N, pts, pm ):
    """global Knothe map: column strips at the quantiles of the x marginal, then conditional quantiles along y"""
    h = 1.0 / N
    o = np.argsort( pts[ :, 0 ], kind = "stable" )
    c = np.cumsum( pm[ o ] ) / pm.sum()
    col = np.minimum( ( c * N ).astype( int ), N - 1 )          # the strip of each point ( by mass )
    sx_col = np.interp( np.arange( 1, N ) / N, c, pts[ o, 0 ] )
    sx = np.repeat( sx_col[ :, None ], N, axis = 1 )
    sy = np.zeros( ( N, N - 1 ) )
    strip_of = np.empty( len( pm ), dtype = int )
    strip_of[ o ] = col
    for a in range( N ):
        sel = strip_of == a
        y, m = pts[ sel, 1 ], pm[ sel ]
        oy = np.argsort( y, kind = "stable" )
        cy = np.cumsum( m[ oy ] ) / m.sum()
        sy[ a ] = np.interp( np.arange( 1, N ) / N, cy, y[ oy ] )
    return least_squares_from_faces( N, h, sx, sy )


# -- the solves ------------------------------------------------------------------------------------

def solve( img, pos, w0, continuation = "never", max_iter = 200, keep_start = True ):
    """`keep_start`: the solver keeps the given start ( after its own warm repair ) instead of swapping it for the Voronoi
    when a cell stays almost empty"""
    import sdot
    from sdot.OtProblem import Tuning
    t0 = time.perf_counter()
    kw = dict( continuation = continuation, max_iter = max_iter, on_failure = "ignore", tuning = Tuning( keep_start = keep_start and w0 is not None ) )
    if w0 is not None:
        kw[ "weights0" ] = w0
    sol = sdot.ot_solve( sdot.SumOfDiracs( pos ), img, **kw )
    dt = time.perf_counter() - t0
    st = sol.stats
    hist = sol.history
    ts = [ float( arr( r[ "t" ] ) ) for r in hist[ 1: ] ] if hist else [ ]
    return sol, dict( converged = bool( sol.converged ), time = dt, nb_iter = int( st[ "nb_iter" ] ), nb_damped = int( sum( t < 1 for t in ts ) ),
                      nb_backtracks = int( st[ "nb_backtracks" ] ), nb_continuation = int( st[ "nb_continuation_steps" ] ),
                      min_t = float( min( ts ) ) if ts else 1.0, status = str( st[ "status" ] ), start_used = str( st[ "start" ] ),
                      min_start_mass = float( st[ "min_start_mass" ] ) * len( pos ) )


def one_case( name, N ):
    import sdot
    vals = image_values( name )
    img = sdot.Image( vals )
    n, h = N * N, 1.0 / N
    Nc = N // 2
    pos, posc = grid_positions( N ), grid_positions( Nc )
    pts, pm = samples( vals )
    rows = [ ]

    # the references: sdot's own defaults ( its continuation / coarse to fine )
    _, s = solve( img, pos, None, continuation = "auto" )
    rows.append( dict( start = "sdot default (cloud path)", build = 0.0, **s ) )
    t0 = time.perf_counter()
    solg = sdot.ot_solve( sdot.SumOfDiracsOnAGrid( ( N, N ) ), img, on_failure = "ignore" )
    rows.append( dict( start = "sdot default (grid path)", build = 0.0, time = time.perf_counter() - t0, converged = bool( solg.converged ),
                       status = str( solg.stats[ "status" ] ),
                       nb_iter = int( solg.stats[ "nb_iter" ] ), nb_backtracks = int( solg.stats[ "nb_backtracks" ] ),
                       nb_continuation = int( solg.stats[ "nb_continuation_steps" ] ) ) )

    t0 = time.perf_counter()
    solc, sc = solve( img, posc, None, continuation = "auto" )
    wc = arr( solc.weights )
    t_coarse = time.perf_counter() - t0

    starts = { "zero": ( np.zeros( n ), 0.0, None ) }
    t0 = time.perf_counter(); w = start_cubic( N, Nc, wc ); starts[ "cubic" ] = ( w, t_coarse + time.perf_counter() - t0, None )
    t0 = time.perf_counter(); w, rf = start_local( N, Nc, wc, pts, pm ); starts[ "local plan" ] = ( w, t_coarse + time.perf_counter() - t0, rf )
    t0 = time.perf_counter(); w, rf = start_global( N, pts, pm ); starts[ "global Knothe" ] = ( w, time.perf_counter() - t0, rf )

    for key, ( w, build, rf ) in starts.items():
        st0 = start_stats( masses( img, pos, w ), n )
        t0 = time.perf_counter()
        wr, nr = repair( img, pos, w, n, h )
        t_rep = time.perf_counter() - t0
        st1 = start_stats( masses( img, pos, wr ), n )
        _, s = solve( img, pos, wr )
        rows.append( dict( start = key, build = build + t_rep, repair_rounds = nr, face_rms_over_h = None if rf is None else rf / h,
                           **{ "init_" + k: v for k, v in st0.items() }, **{ "rep_" + k: v for k, v in st1.items() }, **s ) )
        if key in ( "cubic", "local plan" ):                    # what sdot does with the repaired start by default ( Voronoi swap allowed )
            _, s = solve( img, pos, wr, keep_start = False )
            rows.append( dict( start = key + " (sdot may swap)", build = build + t_rep, **s ) )
    return dict( image = name, N = N, coarse = dict( sc, time = t_coarse ), rows = rows )


def table( results ):
    out = [ ]
    for r in results:
        out.append( f"\n### { r[ 'image' ] }, N = { r[ 'N' ] }  ( coarse { r[ 'N' ] // 2 }²: { r[ 'coarse' ][ 'time' ]:.2f} s, "
                    f"{ r[ 'coarse' ][ 'nb_iter' ] } it )\n" )
        out.append( "| start | init min m·n | init #empty | init #<0.1 | init max err | repair rounds | repaired min m·n | build s | conv | Newton it | damped | min t | solve s |" )
        out.append( "|---|---|---|---|---|---|---|---|---|---|---|---|---|" )
        for x in r[ "rows" ]:
            g = lambda k, f = "{}": f.format( x[ k ] ) if x.get( k ) is not None else "–"
            out.append( f"| { x[ 'start' ] } | { g( 'init_min_rel_mass', '{:.2g}' ) } | { g( 'init_nb_empty' ) } | { g( 'init_nb_small' ) } | "
                        f"{ g( 'init_max_rel_err', '{:.2g}' ) } | { g( 'repair_rounds' ) } | { g( 'rep_min_rel_mass', '{:.2g}' ) } | { g( 'build', '{:.2f}' ) } | { 'yes' if x[ 'converged' ] else 'NO' } | "
                        f"{ x[ 'nb_iter' ] } | { g( 'nb_damped' ) } | { g( 'min_t', '{:.2g}' ) } | { x[ 'time' ]:.2f} |" )
    return "\n".join( out )


def warm_up():
    """the kernels compiled once, before anything is timed"""
    import sdot
    vals = image_values( "sine", 64 )
    img = sdot.Image( vals )
    pos = grid_positions( 8 )
    solve( img, pos, None, continuation = "auto" )
    solve( img, pos, np.zeros( 64 ) )
    solve( img, pos, np.zeros( 64 ), keep_start = False )
    sdot.ot_solve( sdot.SumOfDiracsOnAGrid( ( 8, 8 ) ), img, on_failure = "ignore" )
    masses( img, pos, np.zeros( 64 ) )


def run( Ns ):
    warm_up()
    results = [ ]
    for name in [ "bands", "sine", "star" ]:
        for N in Ns:
            print( f"== { name } N = { N }", flush = True )
            r = one_case( name, N )
            results.append( r )
            print( table( [ r ] ), flush = True )
            with open( os.path.join( OUT, "results.json" ), "w" ) as f:
                json.dump( results, f, indent = 1 )
            with open( os.path.join( OUT, "results.md" ), "w" ) as f:
                f.write( table( results ) + "\n" )


# -- the figure ------------------------------------------------------------------------------------

def figure( N, name = "bands" ):
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    import sdot
    vals = image_values( name )
    img = sdot.Image( vals )
    n, Nc = N * N, N // 2
    pos, posc = grid_positions( N ), grid_positions( Nc )
    pts, pm = samples( vals )
    wc = arr( solve( img, posc, None, continuation = "auto" )[ 0 ].weights )
    ws = { "w = 0": np.zeros( n ), "cubic interp.": start_cubic( N, Nc, wc ), "local plan": start_local( N, Nc, wc, pts, pm )[ 0 ],
           "global Knothe": start_global( N, pts, pm )[ 0 ] }
    sol, _ = solve( img, pos, None, continuation = "auto" )
    ws[ "converged" ] = arr( sol.weights )
    R = 768
    t = ( np.arange( R ) + 0.5 ) / R
    X, Y = np.meshgrid( t, t, indexing = "ij" )
    q = np.stack( [ X.ravel(), Y.ravel() ], axis = 1 )
    fig, axs = plt.subplots( 1, len( ws ), figsize = ( 4 * len( ws ), 4.4 ) )
    for ax, ( title, w ) in zip( axs, ws.items() ):
        lab = labels( q, pos, w ).reshape( R, R )
        edge = np.zeros( ( R, R ), bool )
        edge[ 1:, : ] |= lab[ 1:, : ] != lab[ :-1, : ]
        edge[ :, 1: ] |= lab[ :, 1: ] != lab[ :, :-1 ]
        m = masses( img, pos, w ) * n
        bg = np.kron( vals, np.ones( ( 1, 1 ) ) )
        ax.imshow( bg.T, origin = "lower", extent = [ 0, 1, 0, 1 ], cmap = "gray", vmin = 0, vmax = 1.6 )
        ov = np.zeros( ( R, R, 4 ) )
        ov[ edge ] = [ 0.85, 0.1, 0.1, 1 ]
        small = ( m < 0.1 )[ lab ]
        ov[ small & ~edge ] = [ 0.1, 0.3, 1.0, 0.45 ]
        ax.imshow( ov.transpose( 1, 0, 2 ), origin = "lower", extent = [ 0, 1, 0, 1 ], interpolation = "nearest" )
        ax.set_title( f"{ title }\nmin m·n = { m.min():.2g}, { ( m < 1e-9 ).sum() } empty, { ( m < 0.1 ).sum() } < 0.1", fontsize = 10 )
        ax.set_xticks( [ ] ); ax.set_yticks( [ ] )
    fig.suptitle( f"{ name }, { N }×{ N } diracs: starting cells ( red ), cells under 10 % of their mass ( blue ) -- before repair" )
    fig.tight_layout( rect = [ 0, 0, 1, 0.92 ] )
    path = os.path.join( OUT, f"cells_{ name }_{ N }.png" )
    fig.savefig( path, dpi = 110 )
    print( "wrote", path )


if __name__ == "__main__":
    cmd = sys.argv[ 1 ] if len( sys.argv ) > 1 else "run"
    if cmd == "run":
        run( [ int( a ) for a in sys.argv[ 2: ] ] or [ 64, 128, 256 ] )
    elif cmd == "figure":
        for name in [ "bands", "star" ]:
            figure( int( sys.argv[ 2 ] ) if len( sys.argv ) > 2 else 32, name )
    elif cmd == "check":
        import sdot
        vals = np.full( ( 64, 64 ), 1e-3 ); vals[ :16, : ] = 1      # mass at small x: the dirac at small x gets small cells
        pos = grid_positions( 4 )
        m = masses( sdot.Image( vals ), pos, np.zeros( 16 ) )
        print( "masses ( x slowest ):", np.round( m.reshape( 4, 4 ), 3 ) )
