"""The outputs of `tomography-tents.md`: a cylinder of material pierced by packed bubbles, reconstructed from its projections with tents, coarse
to fine, every iteration of the L-BFGS recorded ( `docs/public/examples/tomography-tents.html` and the card of the gallery
`docs/public/examples/img/tomography-tents.png` ).

The `# region` blocks are what the page shows; the rest ( the reference object, the page ) is not the point of the example.

    python docs/examples/tomography_tents.py
"""
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import sdot
from loom.tensor import Axis, ShapeVar
from sdot.device_optim import lbfgs

jax.config.update( "jax_enable_x64", True )        # the transport sums masses of very different sizes

OUT = Path( __file__ ).resolve().parents[ 1 ] / "public" / "examples"
R = 0.5                                             # the cylinder: its section is the disk of radius R
GAP = 2 * R / 120                                   # between two balls: 1/120 of the diameter of the cylinder


# ── the reference object ( not the point of the example ) ─────────────────────

def packed_bubbles( nb = 34, smallest = 0.015, seed = 0, gravity = 2e-6 ):
    """a DENSE packing of the cylinder by bubbles of many sizes: `nb` random seeds grow together, pushing each other ( and sliding
    down under a light gravity ) so as not to overlap, until they are jammed; each one then grows alone into the room around it; a
    new bubble is born in the biggest hole left, as long as one of radius `smallest` fits; and again, the whole growing until
    nothing moves. The radii are finally reduced by `GAP / 2`: two bubbles in contact end `GAP` apart. Returns `( centers, radii )`."""
    from scipy.optimize import minimize
    rng = np.random.default_rng( seed )
    qmin = smallest + GAP / 2

    def relax( P, Q ):
        n = len( Q )
        i, j = np.triu_indices( n, 1 )
        def f( x ):
            P = x.reshape( -1, 2 )
            d = P[ j ] - P[ i ]
            dist = np.sqrt( ( d ** 2 ).sum( 1 ) ) + 1e-30
            overlap = np.maximum( Q[ i ] + Q[ j ] - dist, 0 )
            r = np.linalg.norm( P, axis = 1 ) + 1e-30
            wall = np.maximum( r + Q - R, 0 )
            grad = np.zeros_like( P )
            c = ( -2 * overlap / dist )[ :, None ] * d
            np.add.at( grad, j, c )
            np.add.at( grad, i, -c )
            grad += ( 2 * wall / r )[ :, None ] * P
            grad[ :, 1 ] += gravity * Q ** 2
            return ( overlap ** 2 ).sum() + ( wall ** 2 ).sum() + gravity * ( Q ** 2 * P[ :, 1 ] ).sum(), grad.ravel()
        res = minimize( f, P.ravel(), jac = True, method = "L-BFGS-B", options = dict( maxiter = 3000, ftol = 1e-16, gtol = 1e-12 ) )
        P = res.x.reshape( -1, 2 )
        overlap = np.maximum( Q[ i ] + Q[ j ] - np.linalg.norm( P[ j ] - P[ i ], axis = 1 ), 0 )
        wall = np.maximum( np.linalg.norm( P, axis = 1 ) + Q - R, 0 )
        return P, ( overlap ** 2 ).sum() + ( wall ** 2 ).sum()

    def grow_together( P, Q ):
        step = 1.1
        while step > 1.0003:
            P2, E = relax( P, Q * step )
            if E < 1e-10:
                P, Q = P2, Q * step
            else:
                step = 1 + ( step - 1 ) / 2
        return P, Q

    def grow_alone( P, Q ):
        for k in rng.permutation( len( Q ) ):
            d = np.linalg.norm( P - P[ k ], axis = 1 ) - Q
            d[ k ] = np.inf
            Q[ k ] = max( Q[ k ], min( d.min(), R - np.linalg.norm( P[ k ] ) ) )
        return P, Q

    def fill_holes( P, Q ):
        while True:
            c = R * np.sqrt( rng.random( 40000 ) )[ :, None ] * ( lambda a: np.stack( [ np.cos( a ), np.sin( a ) ], 1 ) )( 2 * np.pi * rng.random( 40000 ) )
            free = np.minimum( ( np.linalg.norm( c[ :, None ] - P[ None ], axis = 2 ) - Q[ None ] ).min( 1 ), R - np.linalg.norm( c, axis = 1 ) )
            k = free.argmax()
            if free[ k ] < qmin:
                return P, Q
            P, Q = np.vstack( [ P, c[ k ] ] ), np.append( Q, free[ k ] * 0.999 )

    a, rho = 2 * np.pi * rng.random( nb ), R * np.sqrt( rng.random( nb ) )
    P = np.stack( [ rho * np.cos( a ), rho * np.sin( a ) ], 1 )
    Q = 0.02 * 2.5 ** rng.random( nb )                                                  # sizes spread over a factor 2.5
    P, _ = relax( P, Q )
    for _ in range( 4 ):
        P, Q = grow_together( P, Q )
        P, Q = grow_alone( P, Q )
        P, Q = fill_holes( P, Q )
    P, Q = grow_together( P, Q )
    P, Q = grow_alone( P, Q )
    return P, Q * 0.999 - GAP / 2


def chords( centers, radii, nb_angles, nb_bins, extent ):
    """the projections of uniform disks: `( angles, profiles, s_min, dw )`, `profiles[ a, b ]` the mean density on the bin `b` of the
    detector at the angle `a` ( the chord `2 sqrt( r^2 - t^2 )` integrated bin by bin ), the angles in `[ 0, pi )`"""
    angles = np.pi * np.arange( nb_angles ) / nb_angles
    normals = np.stack( [ np.cos( angles ), np.sin( angles ) ], 1 )
    dw, s_min = extent / nb_bins, -extent / 2
    edges = s_min + dw * np.arange( nb_bins + 1 )
    profiles = np.zeros( ( nb_angles, nb_bins ) )
    for c, r in zip( centers, radii ):
        t = np.clip( edges[ None ] - ( normals @ c )[ :, None ], -r, r )
        G = t * np.sqrt( np.maximum( r * r - t * t, 0 ) ) + r * r * np.arcsin( t / r )
        profiles += ( G[ :, 1: ] - G[ :, :-1 ] ) / dw
    return angles, profiles, s_min, dw


def bubbly_cylinder( nb_angles = 600, nb_bins = 1024, extent = 1.2 ):
    """a cylinder of material pierced by packed bubbles: `( bubbles centers, radii, angles, profiles, s_min, dw )`"""
    centers, radii = packed_bubbles()
    angles, full, s_min, dw = chords( [ ( 0, 0 ) ], [ R ], nb_angles, nb_bins, extent )
    _, holes, _, _ = chords( centers, radii, nb_angles, nb_bins, extent )
    return centers, radii, angles, full - holes, s_min, dw


# ── the loss: tents projected at every angle against the measured profiles ───

# region loss
def tents_loss( angles, profiles, s_min, dw ):
    """`loss( x, half_width )`, `x` `[ n, 2 ]`, `half_width` a number or `[ n ]`: the sum over the angles of W2^2( the tents projected
    at the angle, the measured profile )"""
    normals = jnp.asarray( np.stack( [ np.cos( angles ), np.sin( angles ) ], 1 ) )

    def loss( x, half_width ):
        num_angle = Axis( ShapeVar( len( angles ) ), name = "num_angle" )           # the batch: one transport per angle
        tents = sdot.ProjectedSumOfTents( points = x, half_widths = jnp.broadcast_to( half_width, ( len( x ), ) ), normal = normals,
                                          batch_axes = [ num_angle ] )
        target = sdot.Image( values = jnp.asarray( profiles ), origin = [ s_min ], pixel_frame = [ [ dw ] ], batch_axes = [ num_angle ] )
        return sdot.ContinuousOtPlan1d( tents, target ).cost.sum()
    return loss


def stretched_tents_loss( angles, profiles, s_min, dw ):
    """`loss( x, M )`, `x` `[ n, 2 ]`, `M` `[ n, 2, 2 ]`: the same, for tents whose half-width at the angle `n` is `| M^T n |` ( the
    shadow of an ellipse: `M = w * identity` is the round tent of half-width `w` )"""
    normals = jnp.asarray( np.stack( [ np.cos( angles ), np.sin( angles ) ], 1 ) )

    def loss( x, M ):
        num_angle = Axis( ShapeVar( len( angles ) ), name = "num_angle" )
        tents = sdot.ProjectedSumOfAnisotropicTents( points = x, matrices = M, normal = normals, batch_axes = [ num_angle ] )
        target = sdot.Image( values = jnp.asarray( profiles ), origin = [ s_min ], pixel_frame = [ [ dw ] ], batch_axes = [ num_angle ] )
        return sdot.ContinuousOtPlan1d( tents, target ).cost.sum()
    return loss
# endregion loss


# ── the reconstruction: coarse to fine, L-BFGS ───────────────────────────────

# region multiscale
def coarse_to_fine( loss, dw, nb_tents = ( 1024, 4096 ), max_iter = 1500, on_step = None ):
    """the centers of the tents, coarse to fine: L-BFGS on `nb_tents[ 0 ]` tents thrown at random in the cylinder, then every tent
    split in `n_next / n` children, a little apart, and L-BFGS again. The tents narrow with their spacing, `R / sqrt( n )`. A descent
    stops when the points have settled: over 20 iterations they move by less than a twentieth of a bin, in root mean square.
    `on_step( stage, k, x, half_width )` sees every 20 to 100 iterations."""
    rng = np.random.default_rng( 1 )
    r, a = R * np.sqrt( rng.random( nb_tents[ 0 ] ) ), 2 * np.pi * rng.random( nb_tents[ 0 ] )
    x = jnp.asarray( np.stack( [ r * np.cos( a ), r * np.sin( a ) ], 1 ) )
    for level, n in enumerate( nb_tents ):
        if len( x ) != n:
            x = jnp.repeat( x, n // len( x ), axis = 0 ) + ( rng.random( ( n, 2 ) ) - 0.5 ) * R / np.sqrt( n )
        width = R / np.sqrt( n )
        value_and_grad = jax.jit( jax.value_and_grad( lambda x: loss( x, width ) ) )
        x, info = lbfgs( value_and_grad, x, max_iter = max_iter, ftol = 0, min_iter = 20, drift_tol = dw / 20, snapshot_every = 20 if n <= 1024 else 40,
                         on_snapshot = None if on_step is None else lambda k, x: on_step( level, k, np.asarray( x ), width ) )
    return x, width
# endregion multiscale

# region matrices
def stretch_tents( loss, x, width, dw, max_iter = 2000, on_step = None ):
    """every tent gets a matrix of its own, the points staying where they are: `M = L L^T / width + ( dw / 4 ) identity` ( symmetric,
    positive definite, with `L` lower triangular and free ), starting from the round tents of half-width `width`"""
    wmin = dw / 4
    def matrix( l ):
        L = jnp.zeros( ( len( l ), 2, 2 ) ).at[ :, 0, 0 ].set( l[ :, 0 ] ).at[ :, 1, 0 ].set( l[ :, 1 ] ).at[ :, 1, 1 ].set( l[ :, 2 ] )
        return L @ jnp.swapaxes( L, 1, 2 ) / width + wmin * jnp.eye( 2 )
    report = on_step or ( lambda *args: None )

    r = jnp.full( len( x ), np.sqrt( width * ( width - wmin ) ) )           # L = r identity, M = width identity
    l0 = jnp.stack( [ r, jnp.zeros_like( r ), r ], 1 )
    value_and_grad = jax.jit( jax.value_and_grad( lambda l: loss( x, matrix( l ) ) ) )
    l, _ = lbfgs( value_and_grad, l0, max_iter = max_iter, ftol = 0, min_iter = 20, drift_tol = dw / 40,
                  snapshot_every = 20, on_snapshot = lambda k, l: report( 2, k, x, matrix( l ) ) )
    return matrix( l )
# endregion matrices


def draw_tents( viz, p, shape ):
    """the tents at `p`: disks of radius `shape` `[ n ]` ( their half-width ), or the ellipses `M( disk )` for matrices `shape` `[ n, 2, 2 ]`"""
    if shape.ndim == 3:
        a = np.linspace( 0, 2 * np.pi, 8, endpoint = False )
        circle = np.stack( [ np.cos( a ), np.sin( a ) ], 1 )
        corners = p[ :, None ] + 0.8 * np.einsum( "irc,kc->ikr", shape, circle )
        ids = np.arange( len( circle ) * len( p ) ).reshape( len( p ), -1 )
        viz.add_edges( corners.reshape( -1, 2 ), np.stack( [ ids, np.roll( ids, -1, axis = 1 ) ], -1 ).reshape( -1, 2 ), color = viz.color_at( 0 ) )
    else:
        viz.add_points( p, radius = 0.8 * shape )


if __name__ == "__main__":
    t0 = time.perf_counter()
    centers, radii, angles, profiles, s_min, dw = bubbly_cylinder()
    print( f"{ len( radii ) } bubbles, { 100 * ( radii ** 2 ).sum() / R ** 2:.0f} % of the section ( { time.perf_counter() - t0:.1f} s )", flush = True )

    frames, t0, last = [], time.perf_counter(), [ None ]
    def on_step( level, k, x, width ):
        frames.append( ( level, k, np.asarray( x ), np.asarray( width ) ) )
        if last[ 0 ] != level:
            last[ 0 ] = level
            print( f"level { level }: { len( x ) } tents ( { time.perf_counter() - t0:.1f} s )", flush = True )
    loss = tents_loss( angles, profiles, s_min, dw )
    x, width = coarse_to_fine( loss, dw, on_step = on_step )
    print( f"fixed width { width / dw:.1f} bins: loss { float( loss( x, width ) ):.3e}" )
    loss_M = stretched_tents_loss( angles, profiles, s_min, dw )
    M = stretch_tents( loss_M, x, width, dw, on_step = on_step )
    print( f"matrices: median semi-axes { np.median( np.linalg.eigvalsh( np.asarray( M ) ) / dw, axis = 0 ) } bins: loss { float( loss_M( x, M ) ):.3e}, { max( f[ 1 ] for f in frames if f[ 0 ] == 2 ) } iterations" )
    print( f"{ len( frames ) } frames, { time.perf_counter() - t0:.1f} s" )

    viz = sdot.Visualizer( title = "Tents, coarse to fine", frame_axis = "iteration", fps = 8.0, flat_points = True )
    done, total = {}, 0                                 # the abscissa: the iterations since the start, all levels together
    for i, ( level, k, p, width ) in enumerate( frames ):
        total += k - done.get( level, 0 )
        done[ level ] = k
        if i:
            viz.new_frame( total )
        draw_tents( viz, p, width )
    viz.write_html( OUT / "tomography-tents.html" )

    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots( figsize = ( 4, 4 ), dpi = 100 )
    ax.set_aspect( "equal" ), ax.axis( "off" )
    ax.scatter( *np.asarray( x ).T, s = 0.3, c = "#1f77b4" )
    fig.savefig( OUT / "img" / "tomography-tents.png", bbox_inches = "tight", transparent = True )
