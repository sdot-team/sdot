"""The outputs of `gaussian-mixture-fit.md`: a `SumOfGaussians` fitted to a point cloud by minimizing W2 with L-BFGS, with
JAX and with PyTorch, one animated page per framework, the gaussians at each iteration
( `docs/public/examples/gaussian-mixture-fit-*.html` ), and the picture of the gallery, the last iteration on a transparent
background ( `docs/public/examples/img/gaussian-mixture-fit.png` ).

    python docs/examples/gaussian_mixture_fit.py [ jax torch ]
"""
import sys
import time
from pathlib import Path

import numpy as np
import sdot

OUT = Path( __file__ ).resolve().parents[ 1 ] / "public" / "examples"
NB_ITER = 40
NB_POINTS = 1000


def samples( n = NB_POINTS, seed = 0 ):
    """`n` points drawn from a mixture of 4 gaussians -- the data, `[ n, 2 ]`"""
    centers = np.array( [ [ -1.0, -0.6 ], [ 1.1, -0.4 ], [ 0.2, 1.0 ], [ -0.9, 0.9 ] ] )
    sigmas = np.array( [ 0.35, 0.25, 0.3, 0.2 ] )
    masses = np.array( [ 0.35, 0.25, 0.25, 0.15 ] )
    rng = np.random.default_rng( seed )
    k = rng.choice( 4, n, p = masses )
    return centers[ k ] + sigmas[ k, None ] * rng.standard_normal( ( n, 2 ) )


def start():
    """the starting parameters: 4 wide gaussians of equal masses, on a small square in the middle"""
    centers = np.array( [ [ -0.5, -0.5 ], [ 0.5, -0.5 ], [ 0.5, 0.5 ], [ -0.5, 0.5 ] ] )
    return centers, np.full( 4, 0.6 ), np.full( 4, 0.25 )


def bounds():
    """what L-BFGS-B keeps the flat parameters in: the centers are free, the sigmas and the masses positive ( a negative mass
    makes W2 unbounded below, and the first step of L-BFGS-B, which is not scaled, goes there at once )"""
    return [ ( None, None ) ] * 8 + [ ( 1e-3, None ) ] * 4 + [ ( 1e-6, None ) ] * 4


# ---- one fit per framework ------------------------------------------------------------------------------------------
# Each returns the history, a list of ( centers, sigmas, masses, w2 ) as numpy arrays, one entry per iteration. The
# masses need not sum to 1: `SumOfGaussians` normalizes them.

def fit_jax( points ):
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import minimize
    jax.config.update( "jax_enable_x64", True )

    x = jnp.asarray( points )

    def loss( params ):
        centers, sigmas, masses = params
        return sdot.w2( sdot.SumOfDiracs( x ), sdot.SumOfGaussians( centers, sigmas, weights = masses ) )

    # SciPy's L-BFGS works on ONE flat vector of numpy floats: `ravel_pytree` goes there and back
    theta0, unravel = ravel_pytree( tuple( jnp.asarray( p ) for p in start() ) )
    value_and_grad = jax.jit( jax.value_and_grad( lambda theta: loss( unravel( theta ) ) ) )

    history = []
    def fun( theta ):
        value, grad = value_and_grad( theta )
        if not history:
            history.append( record( unravel( theta ), value ) )
        return float( value ), np.asarray( grad )

    minimize( fun, np.asarray( theta0 ), jac = True, method = "L-BFGS-B", bounds = bounds(), options = { "maxiter": NB_ITER },
              callback = lambda intermediate_result: history.append( record( unravel( intermediate_result.x ), intermediate_result.fun ) ) )
    return history


def fit_torch( points ):
    import torch
    from scipy.optimize import minimize

    x = torch.as_tensor( points )
    shapes = [ p.shape for p in start() ]

    def unflatten( theta ):
        return torch.as_tensor( theta ).split( [ int( np.prod( s ) ) for s in shapes ] )

    def loss( centers, sigmas, masses ):
        return sdot.w2( sdot.SumOfDiracs( x ), sdot.SumOfGaussians( centers.reshape( shapes[ 0 ] ), sigmas, weights = masses ) )

    history = []
    def fun( theta ):
        theta = torch.tensor( theta, requires_grad = True )
        value = loss( *unflatten( theta ) )
        value.backward()
        if not history:
            history.append( record( unflatten( theta ), value ) )
        return float( value ), theta.grad.numpy()

    theta0 = np.concatenate( [ p.ravel() for p in start() ] )
    minimize( fun, theta0, jac = True, method = "L-BFGS-B", bounds = bounds(), options = { "maxiter": NB_ITER },
              callback = lambda intermediate_result: history.append( record( unflatten( intermediate_result.x ), intermediate_result.fun ) ) )
    return history


# ---- the pages ------------------------------------------------------------------------------------------------------

def to_host( x ):
    if hasattr( x, "detach" ):
        x = x.detach()
    return np.asarray( x, dtype = float )


def record( params, value ):
    centers, sigmas, masses = ( to_host( p ) for p in params )
    return centers.reshape( -1, 2 ), sigmas, masses / masses.sum(), float( to_host( value ) )


def write_page( name, points, history ):
    """the data ( gray: the background follows the light / dark preference of the system ), and for each gaussian its
    center and its circles at 1 and 2 sigmas, at each iteration"""
    t = np.linspace( 0, 2 * np.pi, 65 )[ :-1 ]
    circle = np.stack( [ np.cos( t ), np.sin( t ) ], axis = 1 )
    colors = [ "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728" ]

    viz = sdot.Visualizer( title = f"Gaussian mixture fit ( { name } )", frame_axis = "iteration" )
    for it, ( centers, sigmas, masses, w2 ) in enumerate( history ):
        if it:
            viz.new_frame( it )
        viz.add_points( points, color = "#888888" )
        for c, s, m, color in zip( centers, sigmas, masses, colors ):
            # the opacity says the mass: a gaussian that loses its mass fades out
            opacity = float( np.clip( 0.25 + 3 * m, 0, 1 ) )
            viz.add_points( c[ None ], radius = 0.04, color = color, opacity = opacity )
            viz.add_edges( c + s * circle, closed = True, color = color, opacity = opacity )
            viz.add_edges( c + 2 * s * circle, closed = True, color = color, opacity = opacity, dashed = True, nb_dashes = 1 )
    viz.write_html( OUT / f"gaussian-mixture-fit-{ name }.html" )


def write_png( points, history ):
    """the picture of the gallery: the data and the gaussians of the last iteration, on a transparent background"""
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt

    centers, sigmas, masses, w2 = history[ -1 ]
    colors = [ "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728" ]
    t = np.linspace( 0, 2 * np.pi, 129 )

    # 4:3 like the cards of the gallery, which crop to fill: every point stays in the frame
    fig = plt.figure( figsize = ( 8, 6 ) )
    ax = fig.add_axes( ( 0, 0, 1, 1 ) )
    ax.scatter( points[ :, 0 ], points[ :, 1 ], s = 6, color = "#888888", linewidths = 0 )
    for c, s, color in zip( centers, sigmas, colors ):
        ax.plot( c[ 0 ], c[ 1 ], "o", color = color, markersize = 5 )
        for k, style in ( ( 1, "-" ), ( 2, "--" ) ):
            ax.plot( c[ 0 ] + k * s * np.cos( t ), c[ 1 ] + k * s * np.sin( t ), style, color = color, linewidth = 2 )
    lo, hi = points.min( axis = 0 ), points.max( axis = 0 )
    mid, half = ( lo + hi ) / 2, 0.53 * max( hi[ 0 ] - lo[ 0 ], 4 / 3 * ( hi[ 1 ] - lo[ 1 ] ) )
    ax.set_xlim( mid[ 0 ] - half, mid[ 0 ] + half )
    ax.set_ylim( mid[ 1 ] - 3 / 4 * half, mid[ 1 ] + 3 / 4 * half )
    ax.axis( "off" )
    fig.savefig( OUT / "img" / "gaussian-mixture-fit.png", dpi = 100, transparent = True )
    plt.close( fig )


CASES = {
    "jax"  : fit_jax,
    "torch": fit_torch,
}

if __name__ == "__main__":
    points = samples()
    for name in sys.argv[ 1: ] or [ "jax" ]:
        t0 = time.perf_counter()
        history = CASES[ name ]( points )
        centers, sigmas, masses, w2 = history[ -1 ]
        print( f"{ name }: W2 { history[ 0 ][ 3 ] :.4g} -> { w2 :.4g} in { len( history ) - 1 } iterations, { time.perf_counter() - t0 :.1f}s" )
        for c, s, m in zip( centers, sigmas, masses ):
            print( f"  center ( { c[ 0 ] :+.3f}, { c[ 1 ] :+.3f} )  sigma { s :.3f}  mass { m :.3f}" )
        write_page( name, points, history )
    write_png( points, history )
