"""The outputs of `fokker-planck.md`: the Lagrangian scheme for linear diffusion of Leclerc, Mérigot, Santambrogio and Stra
( *Lagrangian discretization of crowd motion and linear diffusion*, SIAM J. Numer. Anal., 2020 ). One animated page per case
( `docs/public/examples/fokker-planck-*.html` ), the curves ( `docs/public/examples/img/fokker-planck-*.png` ) and the card
of the gallery ( `docs/public/examples/img/fokker-planck.png` ).

    python docs/examples/fokker_planck.py [ heat drift ]      ( `study`: the choice of eps and tau )
"""
import sys
import time
from pathlib import Path

import numpy as np
import sdot

OUT = Path( __file__ ).resolve().parents[ 1 ] / "public" / "examples"


class Diffusion:
    """`n` particles of mass `1 / n` in `domain` ( a `Box` ), moved by the gradient flow of `int V rho + H_eps( rho )`, `H_eps` the
    Moreau-Yosida regularization of the entropy:

        H_eps( mu ) = min_sigma W_2^2( sigma, mu ) / ( 2 eps ) + int sigma log sigma

    The optimal `sigma` is `exp( ( psi_i - |x - x_i|^2 ) / ( 2 eps ) )` on the Laguerre cell of `x_i` ( weights `psi` ), the
    weights being those that give each cell the mass `1 / n` of its particle: a power diagram whose cells integrate a RADIAL
    function of their seed ( `radial_func = sdot.RadialExp( eps )`, which `OtProblem` solves for ). The gradient of `H_eps` is `( x_i - b_i ) / ( n eps )`, `b_i`
    the barycenter of `sigma` on cell `i`, hence the explicit Euler scheme

        x_i <- x_i + tau ( ( b_i - x_i ) / eps - grad V( x_i ) )."""

    def __init__( self, domain, eps, grad_potential = None, tol = 1e-12 ):
        self.domain, self.eps = domain, float( eps )
        self.grad_potential = grad_potential
        self.settings = sdot.Iterative( tol = tol )
        self.problem = None
        self.nb_newton = 0

    def diagram( self, x, psi ):
        """the power diagram of a step, `sigma` on its cells ( for the pictures )"""
        return sdot.PowerDiagram( x, psi, distribution = self.domain, radial_func = sdot.RadialExp( self.eps ), kernel_dtype = "FP64" )

    def velocity( self, x ):
        """`( b - x ) / eps - grad V( x )`, `b` the barycenters of `sigma`, and the weights `psi`. One transport, whose cells
        integrate the radial function: every cell gets the mass `1 / n` of its particle ( started from the weights of the
        previous step, which the problem keeps )"""
        src = sdot.SumOfDiracs( x )
        if self.problem is None:
            self.problem = sdot.OtProblem( src, self.domain, radial_func = sdot.RadialExp( self.eps ) )
        self.problem.source = src
        plan = self.problem.solve( self.settings )
        self.nb_newton += int( plan.stats[ "nb_iter" ] )
        v = ( np.asarray( plan.barycenters ) - x ) / self.eps
        if self.grad_potential is not None:
            v = v - self.grad_potential( x )
        return v, np.asarray( plan.weights )

    def run( self, x, tau, duration, nb_frames = 60, every_print = 20 ):
        """the frames `( t, positions, weights )`, `nb_frames + 1` of them evenly spaced in time"""
        x = np.asarray( x, dtype = float )
        self.problem = None
        nb_steps = int( round( duration / tau ) )
        every = max( 1, nb_steps // nb_frames )
        frames, t0 = [], time.perf_counter()
        for k in range( nb_steps + 1 ):
            v, psi = self.velocity( x )
            if k % every == 0:
                frames.append( ( k * tau, x.copy(), psi.copy() ) )
            if k % every_print == 0:
                print( f"t = { k * tau:.4f}  E|x|^2 = { ( x ** 2 ).sum( 1 ).mean():.5f}  newton/step = { self.nb_newton / max( k, 1 ):.2f}  "
                       f"( { ( time.perf_counter() - t0 ) / max( k, 1 ) * 1e3:.0f} ms per step )", flush = True )
            if k < nb_steps:
                x = x + tau * v
        return frames


def grid_in_disk( h, radius ):
    """the points of `h Z^2` in the disk of center 0 and radius `radius`"""
    g = np.arange( - int( radius / h ) - 1, int( radius / h ) + 2 ) * h
    p = np.stack( np.meshgrid( g, g, indexing = "ij" ), -1 ).reshape( -1, 2 )
    return p[ ( p ** 2 ).sum( 1 ) < radius ** 2 ]


def study( hs = ( 1 / 30, ), c_epss = ( 0.5, 1.0, 2.0, 4.0 ), c_tau = 0.25, radius = 0.5, duration = 0.05, half_side = 1.5 ):
    """the growth of the second moment, `E|x|^2( t ) = r^2 / 2 + 4 t` for the heat equation, for several `h` and `eps = c h^2`
    ( `tau = c_tau eps` ): its slope over three windows of time"""
    domain = sdot.Box( origin = np.array( [ - half_side ] * 2 ), frame = np.array( [ 2 * half_side ] * 2 ) )
    for h in hs:
        x = grid_in_disk( h, radius )
        for c_eps in c_epss:
            eps = c_eps * h * h
            tau = c_tau * eps
            sim = Diffusion( domain, eps )
            t0 = time.perf_counter()
            frames = sim.run( x, tau, duration, nb_frames = 50, every_print = 10 ** 9 )
            t = np.array( [ f[ 0 ] for f in frames ] )
            m2 = np.array( [ ( f[ 1 ] ** 2 ).sum( 1 ).mean() for f in frames ] )
            slopes = [ np.polyfit( t[ sel ], m2[ sel ], 1 )[ 0 ] for sel in ( t <= duration / 5, ( t >= duration / 5 ) & ( t <= duration / 2 ), t >= duration / 2 ) ]
            print( f"h = 1/{ 1 / h:.0f} ( { len( x ) } particles ), eps = { c_eps } h^2, tau = { c_tau } eps: slopes of E|x|^2 = "
                   + ", ".join( f"{ v:.3f}" for v in slopes ) + f" ( 4 expected ), newton/step = { sim.nb_newton / int( round( duration / tau ) ):.2f}, "
                   f"{ time.perf_counter() - t0:.0f} s", flush = True )


# -- the outputs --------------------------------------------------------------------------------------------------------------

def sigma_pieces( sim, x, psi, max_rings = 10 ):
    """`sigma` drawn as it is: each cell cut into rings homothetic about its vertex mean -- one where `sigma` hardly varies over
    the cell, up to `max_rings` on the long cells of the border -- and each piece with the value of `sigma` at its centroid
    ( normalized by the largest one ). Returns the cells' polygons, and the pieces' polygons and values."""
    pd = sim.diagram( x, psi )
    cells = pd.cells
    polys, pieces, vals = [], [], []
    for i in range( len( x ) ):
        v = cells.vertices( i )
        polys.append( v )
        if len( v ) < 3:
            continue
        d2 = ( ( v - x[ i ] ) ** 2 ).sum( 1 )
        k = int( np.clip( np.ceil( ( d2.max() - d2.min() ) / ( 2 * sim.eps ) / 0.4 ), 1, max_rings ) )
        c = v.mean( 0 )
        t = np.linspace( 0, 1, k + 1 )
        ring = [ c + tk * ( v - c ) for tk in t ]
        pieces.append( ring[ 1 ] )
        for a, b in zip( ring[ 1:-1 ], ring[ 2: ] ):
            for j in range( len( v ) ):
                j1 = ( j + 1 ) % len( v )
                pieces.append( np.array( [ a[ j ], b[ j ], b[ j1 ], a[ j1 ] ] ) )
        for q in pieces[ len( vals ): ]:
            g = q.mean( 0 )
            vals.append( ( psi[ i ] - ( ( g - x[ i ] ) ** 2 ).sum() ) / ( 2 * sim.eps ) )
    vals = np.array( vals )
    return polys, pieces, np.exp( vals - vals.max() )


def palette( nb_levels ):
    import matplotlib
    cmap = matplotlib.colormaps[ "viridis" ]
    return [ matplotlib.colors.to_hex( cmap( k / ( nb_levels - 1 ) ) ) for k in range( nb_levels ) ]


def write_viewer( sim, frames, name, title, nb_levels = 48 ):
    """one frame per stored step: `sigma` ( `sigma_pieces` ), the edges of the cells, the particles"""
    colors = palette( nb_levels )
    viz = sdot.Visualizer( title = title, frame_axis = "time", frame_value = 0.0 )
    for k, ( t, x, psi ) in enumerate( frames ):
        if k:
            viz.new_frame( round( t, 5 ) )
        polys, pieces, val = sigma_pieces( sim, x, psi )
        level = np.minimum( ( val * nb_levels ).astype( int ), nb_levels - 1 )
        for lv in np.unique( level ):
            ids = np.flatnonzero( level == lv )
            pts = np.concatenate( [ pieces[ i ] for i in ids ] )
            ends = np.cumsum( [ len( pieces[ i ] ) for i in ids ] )
            viz.add_faces( pts, [ np.arange( e - len( pieces[ i ] ), e ) for i, e in zip( ids, ends ) ], color = colors[ lv ] )
        polys = [ q for q in polys if len( q ) >= 3 ]
        pts = np.concatenate( polys )
        ends = np.cumsum( [ len( q ) for q in polys ] )
        edges = np.concatenate( [ np.stack( [ np.arange( e - len( q ), e ), np.roll( np.arange( e - len( q ), e ), -1 ) ], 1 )
                                  for q, e in zip( polys, ends ) ] )
        viz.add_edges( pts, edges, color = "#ffffff", opacity = 0.3 )
        viz.add_points( x, color = "#000000" )
    viz.write_html( OUT / f"fokker-planck-{ name }.html" )


def exact_heat_cdf( r, t, radius ):
    """the mass within the radius `r` of the heat flow `d_t rho = laplacian rho` started from the uniform law on the disk of radius
    `radius` ( in the plane, no walls ): `rho( t ) = rho_0 * N( 0, 2 t )`, radial, `int_0^r rho 2 pi r dr`"""
    from scipy.special import i0e
    if t == 0:
        return np.minimum( r / radius, 1 ) ** 2
    s = ( np.arange( 2000 ) + 0.5 ) / 2000 * radius                    # the radii of the disk ( midpoints )
    rr = np.linspace( 0, r.max(), 4000 )
    # `rho( rr ) = 1 / ( pi R^2 ) 1 / ( 2 t ) int_0^R exp( - ( rr - s )^2 / ( 4 t ) ) i0e( rr s / ( 2 t ) ) s ds`
    k = np.exp( - ( rr[ :, None ] - s[ None, : ] ) ** 2 / ( 4 * t ) ) * i0e( rr[ :, None ] * s[ None, : ] / ( 2 * t ) ) * s[ None, : ]
    rho = k.sum( 1 ) * ( radius / 2000 ) / ( 2 * t ) / ( np.pi * radius ** 2 )
    cdf = np.concatenate( [ [ 0 ], np.cumsum( 0.5 * ( rho[ 1: ] * rr[ 1: ] + rho[ :-1 ] * rr[ :-1 ] ) * np.diff( rr ) ) ] ) * 2 * np.pi
    return np.interp( r, rr, cdf )


def write_heat_curves( runs, radius, name ):
    """left: the law of the distance to the center, particles ( steps ) against the exact heat flow ( lines ), at a few times;
    right: the second moment, `r^2 / 2 + 4 t` for the heat flow"""
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    fig, ( ax0, ax1 ) = plt.subplots( 1, 2, figsize = ( 10, 4 ) )
    frames = runs[ 0 ][ 1 ]
    picks = [ frames[ int( round( q * ( len( frames ) - 1 ) ) ) ] for q in ( 0, 0.1, 0.4, 1 ) ]
    r = np.linspace( 0, 1.0, 400 )
    cmap = matplotlib.colormaps[ "viridis" ]
    for c, ( t, x, _ ) in enumerate( picks ):
        col = cmap( c / ( len( picks ) - 1 ) * 0.85 )
        d = np.sort( np.sqrt( ( x ** 2 ).sum( 1 ) ) )
        ax0.step( d, ( np.arange( len( d ) ) + 1 ) / len( d ), where = "post", color = col, linewidth = 1.2, label = f"t = { t:.3f}" )
        ax0.plot( r, exact_heat_cdf( r, t, radius ), color = col, linestyle = "--", linewidth = 1 )
    ax0.set_xlabel( "distance to the center" )
    ax0.set_ylabel( "fraction of the mass" )
    ax0.set_title( "particles ( steps ) and heat flow ( dashed )" )
    ax0.legend( frameon = False )
    for label, fr in runs:
        t = np.array( [ f[ 0 ] for f in fr ] )
        ax1.plot( t, [ ( f[ 1 ] ** 2 ).sum( 1 ).mean() for f in fr ], label = label )
    t = np.array( [ f[ 0 ] for f in frames ] )
    ax1.plot( t, radius ** 2 / 2 + 4 * t, color = "#000000", linestyle = "--", linewidth = 1, label = "heat flow: $r^2/2 + 4t$" )
    ax1.set_xlabel( "t" )
    ax1.set_ylabel( "mean of $|x_i|^2$" )
    ax1.set_title( "second moment" )
    ax1.legend( frameon = False )
    fig.tight_layout()
    fig.savefig( OUT / "img" / f"fokker-planck-{ name }.png", dpi = 110 )
    plt.close( fig )


def write_card( sim, frame ):
    """the picture of the gallery: `sigma` and the cells, on a transparent background"""
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    _, x, psi = frame
    polys, pieces, val = sigma_pieces( sim, x, psi )
    fig = plt.figure( figsize = ( 8, 6 ) )
    ax = fig.add_axes( ( 0, 0, 1, 1 ) )
    ax.add_collection( PolyCollection( pieces, facecolors = matplotlib.colormaps[ "viridis" ]( val ), edgecolors = "face", linewidths = 0.2 ) )
    ax.add_collection( PolyCollection( polys, facecolors = "none", edgecolors = "#ffffff", linewidths = 0.3, alpha = 0.5 ) )
    ax.plot( x[ :, 0 ], x[ :, 1 ], ".", color = "#000000", markersize = 1.5 )
    ax.set_xlim( -0.8, 0.8 )   # 4:3 like the cards of the gallery
    ax.set_ylim( -0.6, 0.6 )
    ax.set_aspect( "equal" )
    ax.axis( "off" )
    fig.savefig( OUT / "img" / "fokker-planck.png", dpi = 100, transparent = True )
    plt.close( fig )


# -- the cases ----------------------------------------------------------------------------------------------------------------

BOX = dict( origin = np.array( [ -1.5, -1.5 ] ), frame = np.array( [ 3.0, 3.0 ] ) )


def heat( h = 1 / 30, c_eps = 8.0, c_tau = 0.25, radius = 0.5, duration = 0.05, name = "heat", card = True, viewer = True ):
    """the heat equation from the uniform law on a disk ( the figure 6 of the paper ): `h Z^2` in the disk, in the box `[ -1.5, 1.5 ]^2`"""
    x = grid_in_disk( h, radius )
    eps = c_eps * h * h
    print( f"{ name }: { len( x ) } particles, eps = { eps:.2e}", flush = True )
    sim = Diffusion( sdot.Box( **BOX ), eps )
    frames = sim.run( x, c_tau * eps, duration, nb_frames = 30 )
    if viewer:
        write_viewer( sim, frames, name, f"Heat equation, h = 1/{ 1 / h:.0f}" )
    if card:
        write_card( sim, frames[ len( frames ) // 4 ] )
    return sim, frames


def heat_both():
    """the two discretizations of the paper, h = 1/30 and 1/80, and their curves against the exact heat flow"""
    radius = 0.5
    _, coarse = heat( 1 / 30, radius = radius )
    _, fine = heat( 1 / 80, radius = radius, name = "heat-fine", card = False, viewer = False )
    write_heat_curves( [ ( "h = 1/80", fine ), ( "h = 1/30", coarse ) ], radius, "heat" )


def drift( h = 1 / 40, c_eps = 8.0, c_tau = 0.25, s = 0.25, duration = 0.25 ):
    """FOKKER-PLANCK: a potential `V( x ) = |x|^2 / ( 2 s^2 )`, whose equilibrium is the gaussian `exp( - V )`; the particles start
    uniform on a disk off the center"""
    x = grid_in_disk( h, 0.3 ) + np.array( [ 0.45, 0.0 ] )
    eps = c_eps * h * h
    print( f"drift: { len( x ) } particles, eps = { eps:.2e}", flush = True )
    sim = Diffusion( sdot.Box( **BOX ), eps, grad_potential = lambda x: x / s ** 2 )
    frames = sim.run( x, c_tau * eps, duration, nb_frames = 30 )
    write_viewer( sim, frames, "drift", "Fokker-Planck, quadratic potential" )
    write_drift_curves( frames, s )
    return sim, frames


def write_drift_curves( frames, s ):
    """the mean and the second moment against the Ornstein-Uhlenbeck process: `m( t ) = m_0 exp( - t / s^2 )`,
    `E|x|^2( t ) = 2 s^2 + ( E|x|^2( 0 ) - 2 s^2 - |m_0|^2 ) exp( - 2 t / s^2 ) + |m( t )|^2`"""
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    t = np.array( [ f[ 0 ] for f in frames ] )
    m = np.array( [ f[ 1 ].mean( 0 ) for f in frames ] )
    m2 = np.array( [ ( f[ 1 ] ** 2 ).sum( 1 ).mean() for f in frames ] )
    m_ex = m[ 0 ][ None, : ] * np.exp( - t / s ** 2 )[ :, None ]
    v0 = m2[ 0 ] - ( m[ 0 ] ** 2 ).sum()
    m2_ex = 2 * s ** 2 + ( v0 - 2 * s ** 2 ) * np.exp( - 2 * t / s ** 2 ) + ( m_ex ** 2 ).sum( 1 )
    fig, ( ax0, ax1 ) = plt.subplots( 1, 2, figsize = ( 10, 4 ) )
    ax0.plot( t, m[ :, 0 ], label = "particles" )
    ax0.plot( t, m_ex[ :, 0 ], color = "#000000", linestyle = "--", linewidth = 1, label = "exact" )
    ax0.set_xlabel( "t" )
    ax0.set_title( "mean of $x_{i,1}$" )
    ax0.legend( frameon = False )
    ax1.plot( t, m2, label = "particles" )
    ax1.plot( t, m2_ex, color = "#000000", linestyle = "--", linewidth = 1, label = "exact" )
    ax1.set_xlabel( "t" )
    ax1.set_title( "mean of $|x_i|^2$" )
    ax1.legend( frameon = False )
    fig.tight_layout()
    fig.savefig( OUT / "img" / "fokker-planck-drift.png", dpi = 110 )
    plt.close( fig )


CASES = { "heat": heat_both, "drift": drift }

if __name__ == "__main__":
    ( OUT / "img" ).mkdir( parents = True, exist_ok = True )
    if sys.argv[ 1: ] == [ "study" ]:
        study()
    for name in [ a for a in sys.argv[ 1: ] if a != "study" ] or ( CASES if len( sys.argv ) == 1 else [] ):
        CASES[ name ]()
