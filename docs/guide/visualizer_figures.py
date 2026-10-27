"""The figures of `visualizer.md` ( `docs/public/guide/visualizer-*.html` ): each one is a case of the page, written the way
the page tells it.

    python docs/guide/visualizer_figures.py [ plan scene descent sweep cloud-3d slice-4d convergence ]
"""
import sys
from pathlib import Path

import numpy as np
import sdot

OUT = Path( __file__ ).resolve().parents[ 1 ] / "public" / "guide"


def _target():
    """three gaussians in the unit square"""
    centers = np.array( [ [ 0.3, 0.3 ], [ 0.7, 0.35 ], [ 0.5, 0.75 ] ] )
    return sdot.SumOfGaussians( centers, sigmas = np.array( [ 0.12, 0.08, 0.1 ] ), weights = np.array( [ 1.0, 0.6, 0.8 ] ),
                                support_sigmas = 3 )


def plan( n = 100 ):
    """One object, one call: a plan with the segment from each seed to the barycenter of its cell."""
    rng = np.random.default_rng( 0 )
    p = sdot.plan( sdot.SumOfDiracs( rng.random( ( n, 2 ) ) ), _target() )
    p.write_html( OUT / "visualizer-plan.html", title = "A plan and its transport", transport = True, opacity = 0.5 )


def scene( n = 120 ):
    """Several things in one scene: the cells, faded; one cell and its neighbors highlighted; the 8 seeds that are the farthest
    from their barycenter in red."""
    rng = np.random.default_rng( 1 )
    x = rng.random( ( n, 2 ) )
    p = sdot.plan( sdot.SumOfDiracs( x ), _target() )
    bar = np.asarray( p.barycenters )
    far = np.zeros( n, dtype = bool )
    far[ np.argsort( np.linalg.norm( bar - x, axis = 1 ) )[ -8: ] ] = True

    viz = sdot.Visualizer( title = "One scene, several objects" )
    viz.add( p, opacity = 0.15, seeds = False )
    cx = p.power_diagram().cell_complex()
    i = int( np.argmin( np.linalg.norm( x - 0.5, axis = 1 ) ) )
    neighbors = sorted( { s for f in cx.cells[ i ].children for s in f.sides if s >= 0 and s != i } )
    cx.cells.add_to_viz( viz, ids = neighbors, color = ( "#c8c8c8", "#606060" ) )
    cx.cells.add_to_viz( viz, ids = [ i ], color = "#4e79a7" )
    viz.add_points( x[ ~far ], color = "#000000" )
    viz.add_points( x[ far ], color = "#d62728" )
    viz.add_edges( np.concatenate( [ x[ far ], bar[ far ] ] ),
                   np.stack( [ np.arange( far.sum() ), far.sum() + np.arange( far.sum() ) ], axis = 1 ),
                   color = "#d62728", dashed = True )
    viz.write_html( OUT / "visualizer-scene.html" )


def descent( n = 200 ):
    """A descent replayed: the cells and the seeds at each Lloyd iteration, one frame per iteration."""
    target = _target()
    frames = []
    sdot.quantization( target, n, seed = 0, callback = lambda it, pos, plan: frames.append( plan.power_diagram() ) )

    viz = sdot.Visualizer( title = "Lloyd, iteration by iteration", frame_axis = "iteration", fps = 4 )
    for it, pd in enumerate( frames ):
        if it:
            viz.new_frame( it )
        viz.add( pd, opacity = 0.35 )
    viz.write_html( OUT / "visualizer-descent.html" )
    print( f"descent: { len( frames ) } frames" )


def sweep( n = 200 ):
    """Two parameters: the same diracs sent to a gaussian whose width and center both vary -- one bar per axis."""
    rng = np.random.default_rng( 2 )
    x = rng.random( ( n, 2 ) )
    viz = sdot.Visualizer( title = "Two parameters", frame_axis = ( "sigma", "center" ), playable = ( False, True ) )
    first = True
    for s in ( 0.08, 0.12, 0.18, 0.25 ):
        for c in np.linspace( 0.3, 0.7, 5 ):
            if not first:
                viz.new_frame( ( s, round( float( c ), 2 ) ) )
            first = False
            g = sdot.SumOfGaussians( np.array( [ [ c, 0.5 ] ] ), sigmas = np.array( [ s ] ), support_sigmas = 3 )
            viz.add( sdot.plan( sdot.SumOfDiracs( x ), g ), opacity = 0.4 )
    viz.write_html( OUT / "visualizer-sweep.html" )


def cloud_3d( n = 2000 ):
    """3D: diracs quantizing three gaussians, colored by depth -- and the cells of the few nearest to the center, as wireframes."""
    centers = np.array( [ [ -0.5, -0.3, -0.2 ], [ 0.5, -0.2, 0.1 ], [ 0.0, 0.5, 0.3 ] ] )
    target = sdot.SumOfGaussians( centers, sigmas = np.array( [ 0.25, 0.15, 0.2 ] ), weights = np.array( [ 1.0, 0.6, 0.8 ] ),
                                  support_sigmas = 3 )
    history = []
    sdot.quantization( target, n, seed = 0, max_iter = 30, callback = lambda it, pos, plan: history.append( ( np.asarray( pos ), plan ) ) )
    x, p = history[ -1 ]

    viz = sdot.Visualizer( title = "A cloud in 3D", depth_colors = True )
    viz.add_points( x )
    near = np.argsort( np.linalg.norm( x - x.mean( axis = 0 ), axis = 1 ) )[ :6 ]
    cx = p.power_diagram().cell_complex()
    cx.cells.add_to_viz( viz, ids = near, faces = False, edge_color = ( "#000000", "#ffffff" ) )
    viz.write_html( OUT / "visualizer-cloud-3d.html" )


def slice_4d( n = 16 ):
    """4D: the cells of a power diagram in the unit hypercube -- the page shows a 3D slice, the 4th coordinate on a slider."""
    rng = np.random.default_rng( 4 )
    pd = sdot.PowerDiagram( rng.random( ( n, 4 ) ), boundaries = sdot.box_half_spaces( [ 0 ] * 4, [ 1 ] * 4 ) )
    pd.write_html( OUT / "visualizer-slice-4d.html", title = "A slice of a 4D diagram", opacity = 0.3 )


def convergence( n = 300 ):
    """The root mean square move of the diracs along the Lloyd iterations, for two numbers of diracs."""
    series = {}
    for m in ( n, 4 * n ):
        history = []
        sdot.quantization( _target(), m, seed = 0, tol = 3e-3, callback = lambda it, pos, plan: history.append( np.asarray( pos ) ) )
        series[ f"{ m } diracs" ] = [ float( np.sqrt( ( ( b - a ) ** 2 ).sum( axis = 1 ).mean() ) ) for a, b in zip( history, history[ 1: ] ) ]
    sdot.write_convergence_html( series, OUT / "visualizer-convergence.html", title = "Lloyd: how much the diracs move",
                                 ylabel = "root mean square move" )


CASES = {
    "plan"       : plan,
    "scene"      : scene,
    "descent"    : descent,
    "sweep"      : sweep,
    "cloud-3d"   : cloud_3d,
    "slice-4d"   : slice_4d,
    "convergence": convergence,
}

if __name__ == "__main__":
    for name in sys.argv[ 1: ] or CASES:
        CASES[ name ]()
