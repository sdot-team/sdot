"""The figures of `periodicity.md`: pictures ( `docs/public/guide/img/periodicity-*.png` ) and interactive pages of the HTML
viewer ( `sdot.Visualizer`, `docs/public/guide/periodicity-*.html` ).

    python docs/guide/ground-metrics/periodicity_figures.py [ gluings walls unfolded crossing torus klein moebius cone p4 hexagonal domain cylinder ]

Every page draws the cells AS THEY ARE COMPUTED: whole, around their diracs, not cut to the domain -- so a cell may lie outside
the domain, away from the others. The domain is dashed. The fainter copies are the cells moved by the elements of the group of
the first ring ( `PowerDiagram.periodicity_rings( 1 )` ): on the quotient, they glue to the cells without a seam.
"""
import sys
from pathlib import Path

import numpy as np
import sdot

OUT = Path( __file__ ).resolve().parents[ 2 ] / "public" / "guide"
BACKGROUND = "#ffffff"                                    # the outlines of the domain and the diracs are dark

UNIT = sdot.Box( nb_dims = 2 )
FLIP = np.diag( [ 1.0, -1.0 ] )
QUARTER = np.array( [ [ 0.0, -1.0 ], [ 1.0, 0.0 ] ] )

# the groups of the guide's code block, in its order
GROUPS = [
    ( "a torus"         , [ [ 1, 0 ], [ 0, 1 ] ] ),
    ( "a Klein bottle"  , [ ( FLIP, [ 1, 1 ] ), [ 0, 1 ] ] ),
    ( "a Moebius strip" , [ ( FLIP, [ 1, 1 ] ) ] ),
    ( "a cone"          , [ ( QUARTER, [ 0, 0 ] ) ] ),
    ( "a cylinder"      , [ True, False ] ),
]


# -- matplotlib -------------------------------------------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    return plt


def _save( fig, name ):
    ( OUT / "img" ).mkdir( parents = True, exist_ok = True )
    fig.tight_layout()
    fig.savefig( OUT / "img" / f"periodicity-{ name }.png", dpi = 110, facecolor = "#ffffff", bbox_inches = "tight" )


def _side( ax, a, b, nb_heads = 0, color = "#4e79a7" ):
    """a side of the square from `a` to `b`: glued ( `nb_heads` arrow heads in the middle, pointing from `a` to `b` ), or a
    wall ( `nb_heads = 0`: a thick grey line )"""
    a, b = np.asarray( a, float ), np.asarray( b, float )
    if nb_heads == 0:
        ax.plot( *zip( a, b ), color = "#8c8c8c", linewidth = 5, solid_capstyle = "butt" )
        return
    ax.plot( *zip( a, b ), color = color, linewidth = 2 )
    u = ( b - a ) / np.linalg.norm( b - a )
    for k in range( nb_heads ):
        m = ( a + b ) / 2 + ( k - ( nb_heads - 1 ) / 2 ) * 0.09 * u
        ax.annotate( "", xy = m + 0.06 * u, xytext = m - 0.02 * u,
                     arrowprops = dict( arrowstyle = "-|>,head_length=0.7,head_width=0.35", color = color, linewidth = 2 ) )


def gluings():
    """the unit square of each group, its sides glued by the group ( the same arrows are the same side ) or left as walls"""
    plt = _plt()
    o, x, y, xy = ( 0, 0 ), ( 1, 0 ), ( 0, 1 ), ( 1, 1 )
    blue, orange = "#4e79a7", "#f28e2b"
    sides = {
        "a torus"        : [ ( o, x, 1, blue ), ( y, xy, 1, blue ), ( o, y, 2, orange ), ( x, xy, 2, orange ) ],
        "a Klein bottle" : [ ( o, x, 1, blue ), ( y, xy, 1, blue ), ( o, y, 2, orange ), ( xy, x, 2, orange ) ],
        "a Moebius strip": [ ( o, x, 0, None ), ( y, xy, 0, None ), ( o, y, 2, orange ), ( xy, x, 2, orange ) ],
        "a cone"         : [ ( o, x, 1, blue ), ( o, y, 1, blue ), ( x, xy, 0, None ), ( y, xy, 0, None ) ],
        "a cylinder"     : [ ( o, x, 0, None ), ( y, xy, 0, None ), ( o, y, 2, orange ), ( x, xy, 2, orange ) ],
    }
    fig, axes = plt.subplots( 1, len( GROUPS ), figsize = ( 2.9 * len( GROUPS ), 3.3 ) )
    for ax, ( name, periodicity ) in zip( axes, GROUPS ):
        ax.fill( [ 0, 1, 1, 0 ], [ 0, 0, 1, 1 ], color = "#eef3f8" )
        for a, b, nb, color in sides[ name ]:
            _side( ax, a, b, nb, color )
        if name == "a cone":
            ax.plot( 0, 0, "ko", markersize = 5 )
            ax.annotate( "apex", ( 0, 0 ), ( 0.08, 0.1 ), fontsize = 9 )
        ax.set_title( name, fontsize = 11 )
        ax.set_xlim( -0.15, 1.15 )
        ax.set_ylim( -0.15, 1.15 )
        ax.set_aspect( "equal" )
        ax.axis( "off" )
    _save( fig, "gluings" )


# -- the viewer -------------------------------------------------------------------------------------------------------------

def walls():
    """the same diracs and masses with walls, then on a torus: the cells and their copies by the first ring of the torus"""
    rng = np.random.default_rng( 1 )
    n = 24
    pos = rng.uniform( 0, 1, size = ( n, 2 ) )
    mass = rng.uniform( 0.3, 2.0, size = n )
    viz = sdot.Visualizer( title = "Walls, then a torus", frame_axis = "periodic", playable = False, background = BACKGROUND )

    # with walls there is no group: its cells are moved by hand by the torus's ( `Cell.moved` ), to show that they do not glue
    rings = sdot.PowerDiagram( pos, distribution = UNIT, periodicity = True ).periodicity_rings( 1 )
    cells = sdot.plan( sdot.SumOfDiracs( pos, mass ), UNIT ).power_diagram().cells
    for k, a in enumerate( rings ):
        cells.moved( a ).add_to_viz( viz, opacity = 1.0 if k == 0 else 0.25, edge_opacity = 1.0 if k == 0 else 0.25, frames = k == 0 )
        viz.add_points( pos @ a[ :, :2 ].T + a[ :, 2 ], color = "#000000", opacity = 1.0 if k == 0 else 0.25, frames = k == 0 )

    viz.new_frame( 1 )
    p = sdot.plan( sdot.SumOfDiracs( pos, mass ), UNIT, periodicity = True )
    p.power_diagram().add_to_viz( viz )
    viz.write_html( OUT / "periodicity-walls.html" )


def unfolded():
    """the cells as they are computed: around their diracs, some of them given outside the domain; then with the first ring"""
    rng = np.random.default_rng( 2 )
    n = 12
    pos = rng.uniform( 0, 1, size = ( n, 2 ) )
    pos[ :3 ] += [ [ 1.6, 0.3 ], [ -1.4, 0.8 ], [ 0.2, -1.5 ] ]          # given outside the domain: their cells are there
    mass = rng.uniform( 0.3, 2.0, size = n )
    mass[ 5 ] = 6.0                                                    # a big cell, across a side
    pd = sdot.plan( sdot.SumOfDiracs( pos, mass ), UNIT, periodicity = True ).power_diagram()
    viz = sdot.Visualizer( title = "Periodic cells, as computed", frame_axis = "ring", playable = False, background = BACKGROUND )
    pd.add_to_viz( viz, nb_periodicity_rings = 0 )
    viz.new_frame( 1 )
    pd.add_to_viz( viz )
    viz.write_html( OUT / "periodicity-unfolded.html" )


def lloyd_page( name, title, target, periodicity, x, nb_iter, nb_frames = 20 ):
    """Lloyd's iterations on the quotient ( each dirac goes to the barycenter of its cell, next to it whatever the side it
    crossed ), recorded on `nb_frames` frames, each with the first ring of the group"""
    pb = sdot.OtProblem( sdot.SumOfDiracs( x ), target, periodicity = periodicity )
    kept = set( np.unique( np.geomspace( 1, nb_iter, nb_frames ).astype( int ) - 1 ).tolist() )
    viz = sdot.Visualizer( title = title, frame_axis = "iteration", background = BACKGROUND )
    first = True
    for it in range( nb_iter ):
        pb.source = sdot.SumOfDiracs( x )
        p = pb.solve()
        if it in kept:
            if not first:
                viz.new_frame( it )
            first = False
            p.power_diagram().add_to_viz( viz )
        x = np.asarray( p.barycenters ).reshape( x.shape )
    viz.write_html( OUT / f"periodicity-{ name }.html" )


def crossing( n = 12, nb_frames = 16 ):
    """a torus, one dirac moving to the right across the side of the support: its cell follows it, whole, out of the
    domain, and its copy comes in on the other side"""
    rng = np.random.default_rng( 4 )
    pos = rng.uniform( 0.05, 0.75, size = ( n, 2 ) )
    xs = np.linspace( 0.7, 1.45, nb_frames )
    viz = sdot.Visualizer( title = "A dirac crossing a side of a torus", frame_axis = "x", background = BACKGROUND,
                           frame_value = float( xs[ 0 ] ) )
    for k, x in enumerate( xs ):
        if k:
            viz.new_frame( float( x ) )
        pos[ 0 ] = [ x, 0.45 ]
        p = sdot.plan( sdot.SumOfDiracs( pos ), UNIT, periodicity = True )
        p.power_diagram().add_to_viz( viz )
        viz.add_points( pos[ :1 ], radius = 0.015, color = "#000000" )         # the moving dirac, bigger
    viz.write_html( OUT / "periodicity-crossing.html" )


def torus():
    rng = np.random.default_rng( 3 )
    lloyd_page( "torus", "Lloyd on a torus", UNIT, True, rng.uniform( 0, 1, size = ( 60, 2 ) ), 50 )


def klein():
    rng = np.random.default_rng( 6 )
    lloyd_page( "klein", "Lloyd on a Klein bottle", UNIT, [ ( FLIP, [ 1.0, 1.0 ] ), [ 0.0, 1.0 ] ],
                rng.uniform( 0, 1, size = ( 40, 2 ) ), 50 )


def moebius():
    rng = np.random.default_rng( 7 )
    target = sdot.Box( origin = [ 0.0, 0.0 ], frame = [ 2.0, 1.0 ] )
    lloyd_page( "moebius", "Lloyd on a Moebius strip", target, [ ( FLIP, [ 2.0, 1.0 ] ) ],
                rng.uniform( 0, 1, size = ( 50, 2 ) ) * [ 2.0, 1.0 ], 50 )


def cone():
    # the density concentrated near the apex: the cells there wrap around it
    g = ( np.arange( 32 ) + 0.5 ) / 32
    values = 0.2 + np.exp( - ( g[ :, None ] ** 2 + g[ None, : ] ** 2 ) / 0.15 )
    target = sdot.Image( values, frame = [ 1.0, 1.0 ] )
    rng = np.random.default_rng( 8 )
    lloyd_page( "cone", "Lloyd on a cone ( a quarter turn )", target, [ ( QUARTER, [ 0.0, 0.0 ] ) ],
                rng.uniform( 0, 1, size = ( 40, 2 ) ), 50 )


def p4():
    # the wallpaper group p4: quarter turns about the origin and the unit translations; fundamental domain: a square of side 1 / 2
    target = sdot.Box( origin = [ 0.0, 0.0 ], frame = [ 0.5, 0.5 ] )
    rng = np.random.default_rng( 9 )
    lloyd_page( "p4", "Lloyd on the wallpaper group p4", target, [ ( QUARTER, [ 0.0, 0.0 ] ), [ 1.0, 0.0 ] ],
                rng.uniform( 0, 0.5, size = ( 12, 2 ) ), 50 )


def hexagonal():
    # the lattice of the hexagonal packing: a box of height sqrt( 3 ) / 2, glued to itself with a shift of half its width
    h = np.sqrt( 3 ) / 2
    rng = np.random.default_rng( 4 )
    target = sdot.Box( origin = [ 0.0, 0.0 ], frame = [ 1.0, h ] )
    lloyd_page( "hexagonal", "Lloyd on a hexagonal torus", target, [ [ 1.0, 0.0 ], [ 0.5, h ] ],
                rng.uniform( 0, 1, size = ( 16, 2 ) ) * [ 1.0, h ], 80 )


def domain():
    """`periodicity = target.domain` on a regular hexagon: its opposite sides are glued by translations ( a hexagonal torus )"""
    t = np.arange( 6 ) * np.pi / 3
    target = sdot.ConvexHull( np.stack( [ np.cos( t ), np.sin( t ) ], -1 ) )
    rng = np.random.default_rng( 6 )
    lloyd_page( "domain", "A hexagon glued by target.domain", target, target.domain, rng.uniform( -0.5, 0.5, size = ( 40, 2 ) ), 30 )


def cylinder():
    # a cylinder whose axis is not an axis of the frame: a rectangle turned by 30 degrees, glued along its long side; its two
    # other sides stay walls. Booleans cannot say it ( they read an axis-aligned box ), a translation does.
    t = np.radians( 30 )
    u, v = np.array( [ np.cos( t ), np.sin( t ) ] ), np.array( [ - np.sin( t ), np.cos( t ) ] )
    target = sdot.ConvexHull( np.array( [ [ 0.0, 0.0 ], 2 * u, 2 * u + v, v ] ) )
    rng = np.random.default_rng( 5 )
    x = rng.uniform( 0, 1, size = ( 40, 2 ) )
    lloyd_page( "cylinder", "Lloyd on a turned cylinder", target, [ 2 * u ], x[ :, :1 ] * 2 * u + x[ :, 1: ] * v, 50 )


CASES = {
    "gluings"  : gluings,
    "walls"    : walls,
    "unfolded" : unfolded,
    "crossing" : crossing,
    "torus"    : torus,
    "klein"    : klein,
    "moebius"  : moebius,
    "cone"     : cone,
    "p4"       : p4,
    "hexagonal": hexagonal,
    "domain"   : domain,
    "cylinder" : cylinder,
}


if __name__ == "__main__":
    OUT.mkdir( parents = True, exist_ok = True )
    for name in sys.argv[ 1: ] or CASES:
        CASES[ name ]()
