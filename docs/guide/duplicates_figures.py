"""The pictures of `duplicates.md` ( `docs/public/guide/img/duplicates-*.png` ).

    python docs/guide/duplicates_figures.py [ limit virtual solver lloyd ]

Each figure is a row of panels drawn with matplotlib from the cells of the diagrams themselves ( `PowerDiagram.cells` ).
"""
import sys
from pathlib import Path

import numpy as np
import sdot
from sdot.groups import Groups

OUT = Path( __file__ ).resolve().parents[ 1 ] / "public" / "guide" / "img"
GREY = "#d9d9d9"
COLORS = [ "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2" ]


def _figure( nb ):
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots( 1, nb, figsize = ( 3.6 * nb, 3.8 ) )
    for ax in np.atleast_1d( axes ):
        ax.set_xlim( 0, 1 )
        ax.set_ylim( 0, 1 )
        ax.set_aspect( "equal" )
        ax.set_xticks( [] )
        ax.set_yticks( [] )
    return fig, np.atleast_1d( axes )


def _draw( ax, pd, colored = (), title = None, seeds = True, virtual = False ):
    """the cells of `pd`, those of `colored` in color ( in that order ), the others grey; the seeds as dots"""
    cells = pd.cells
    pos = np.asarray( pd.positions ).reshape( -1, 2 )
    col = { int( i ): COLORS[ k % len( COLORS ) ] for k, i in enumerate( colored ) }
    for i in range( len( pos ) ):
        v = cells.vertices( i )
        if len( v ) >= 3:
            ax.fill( v[ :, 0 ], v[ :, 1 ], color = col.get( i, GREY ), alpha = 0.85 if i in col else 0.6,
                     linewidth = 0.8, edgecolor = "#ffffff" )
    if seeds:
        ax.plot( pos[ :, 0 ], pos[ :, 1 ], "k.", markersize = 4 )
    if virtual and pd.groups is not None:
        g = pd.groups
        q = g.virtual_positions( pos )
        shared = np.flatnonzero( ( g.parent >= 0 ) & ( g.outer[ np.maximum( g.parent, 0 ) ] == 0 ) )
        for i in shared:
            ax.plot( q[ i, 0 ], q[ i, 1 ], "o", markersize = 6, markerfacecolor = "none", markeredgecolor = col.get( int( i ), "k" ), markeredgewidth = 1.5 )
            ax.plot( [ pos[ i, 0 ], q[ i, 0 ] ], [ pos[ i, 1 ], q[ i, 1 ] ], color = col.get( int( i ), "k" ), linewidth = 0.8, linestyle = ":" )
    if title:
        ax.set_title( title, fontsize = 10 )


def _save( fig, name ):
    OUT.mkdir( parents = True, exist_ok = True )
    fig.tight_layout()
    fig.savefig( OUT / f"duplicates-{ name }.png", dpi = 110, facecolor = "#ffffff" )


def _cloud( n, seed ):
    rng = np.random.default_rng( seed )
    pos = rng.uniform( 0.06, 0.94, size = ( n, 2 ) )
    pos[ 0 ] = pos[ 1 ] = [ 0.5, 0.5 ]
    return pos


def limit():
    """two diracs `p +- delta u` closer and closer ( ordinary diagrams ), and the exact duplicates whose virtual diracs are `p +- r u`"""
    box = sdot.box_half_spaces( [ 0, 0 ], [ 1, 1 ] )
    pos = _cloud( 16, 3 )
    u = np.array( [ np.cos( 0.5 ), np.sin( 0.5 ) ] )
    fig, axes = _figure( 3 )
    for ax, delta in zip( axes[ :2 ], ( 0.12, 0.03 ) ):
        p = pos.copy()
        p[ 0 ] -= delta * u
        p[ 1 ] += delta * u
        _draw( ax, sdot.PowerDiagram( p, boundaries = box ), colored = ( 0, 1 ), title = f"two diracs $\\delta = { delta }$ apart" )
    dirs = np.zeros_like( pos )
    dirs[ 0 ], dirs[ 1 ] = - u, u
    pd = sdot.PowerDiagram( pos, np.zeros( len( pos ) ), boundaries = box, groups = Groups.nested( [ [ 0, 1 ] ], pos, radius = 0.08, dirs = dirs ) )
    _draw( axes[ 2 ], pd, colored = ( 0, 1 ), title = "exact duplicates, an aggregate", virtual = True )
    _save( fig, "limit" )


def virtual():
    """a site of four duplicates: the fan of `groups = "auto"`, then other virtual weights, then a close dirac: an aggregate in an aggregate"""
    box = sdot.box_half_spaces( [ 0, 0 ], [ 1, 1 ] )
    pos = _cloud( 16, 5 )
    pos[ [ 2, 3 ] ] = pos[ 0 ]
    fig, axes = _figure( 3 )
    w = np.zeros( len( pos ) )
    pd = sdot.PowerDiagram( pos, w, boundaries = box, groups = "auto" )
    _draw( axes[ 0 ], pd, colored = ( 0, 1, 2, 3 ), title = "groups = 'auto': equal weights", virtual = True )
    w2 = w.copy()
    w2[ [ 1, 2, 3 ] ] = [ 0.004, -0.002, 0.0015 ]
    pd = sdot.PowerDiagram( pos, w2, boundaries = box, groups = pd.groups )
    _draw( axes[ 1 ], pd, colored = ( 0, 1, 2, 3 ), title = "the same, other virtual weights", virtual = True )
    p3 = pos.copy()
    p3[ 4 ] = p3[ 0 ] + [ 1e-9, 0.0 ]
    pd = sdot.PowerDiagram( p3, w, boundaries = box, groups = "auto" )
    _draw( axes[ 2 ], pd, colored = ( 0, 1, 2, 3, 4 ), title = "and a dirac $10^{-9}$ away", virtual = True )
    _save( fig, "virtual" )


def solver():
    """the same plan, `duplicates = "merge"` and `"split"`: the shared cell and its barycentre, or a piece each and its own"""
    pos = _cloud( 24, 7 )
    pos[ 2 ] = pos[ 0 ]
    fig, axes = _figure( 2 )
    for ax, mode in zip( axes, ( "merge", "split" ) ):
        p = sdot.OtProblem( sdot.SumOfDiracs( pos ), sdot.Box() ).solve( sdot.Iterative( tol = 1e-12, duplicates = mode ) )
        pd = p.power_diagram()
        _draw( ax, pd, colored = ( 0, 1, 2 ), title = f"duplicates = '{ mode }'" )
        b = np.asarray( p.barycenters ).reshape( -1, 2 )
        m = np.asarray( p.cell_masses ).reshape( -1 )
        for k, i in enumerate( ( 0, 1, 2 ) ):
            ax.annotate( "", xy = b[ i ], xytext = pos[ i ], arrowprops = dict( arrowstyle = "->", color = "k", linewidth = 1.2 ) )
            ax.plot( *b[ i ], "x", color = "k", markersize = 7 )
        ax.text( 0.02, 0.02, "masses " + ", ".join( f"{ m[ i ] * len( pos ) :.3f}" for i in ( 0, 1, 2 ) ) + "  ( x 1/n )", fontsize = 8,
                 transform = ax.transAxes )
    _save( fig, "solver" )


def lloyd():
    """Lloyd's iterations ( each dirac to the barycentre of its cell ) from a cloud with three pairs of duplicates"""
    rng = np.random.default_rng( 11 )
    x0 = rng.uniform( 0.1, 0.9, size = ( 20, 2 ) )
    x0[ 1 ], x0[ 3 ], x0[ 5 ] = x0[ 0 ], x0[ 2 ], x0[ 4 ]
    fig, axes = _figure( 2 )
    for ax, mode in zip( axes, ( "merge", "split" ) ):
        x = x0.copy()
        path = [ x.copy() ]
        pb = sdot.OtProblem( sdot.SumOfDiracs( x ), sdot.Box() )
        for it in range( 25 ):
            pb.source = sdot.SumOfDiracs( x )
            p = pb.solve( sdot.Iterative( tol = 1e-11, duplicates = mode ) )
            x = np.asarray( p.barycenters ).reshape( x.shape )
            path.append( x.copy() )
        _draw( ax, p.power_diagram(), colored = range( 6 ), title = f"25 steps of Lloyd, '{ mode }'", seeds = False )
        path = np.asarray( path )
        for i in range( len( x ) ):
            ax.plot( path[ :, i, 0 ], path[ :, i, 1 ], "-", color = "k", linewidth = 0.7 )
        ax.plot( path[ -1, :, 0 ], path[ -1, :, 1 ], "k.", markersize = 4 )
        ax.plot( x0[ :6, 0 ], x0[ :6, 1 ], "ks", markersize = 4, markerfacecolor = "none" )
    _save( fig, "lloyd" )


if __name__ == "__main__":
    for name in sys.argv[ 1: ] or [ "limit", "virtual", "solver", "lloyd" ]:
        globals()[ name ]()
        print( "written", OUT / f"duplicates-{ name }.png" )
