"""The outputs of `free-surface-fluid.md`: a fluid with a free surface, Gallouët-Mérigot scheme with partial transport. One
animated page per case ( `docs/public/examples/free-surface-*.html` ) and the card of the gallery
( `docs/public/examples/img/free-surface-fluid.png` ).

    python docs/examples/free_surface_fluid.py [ dam_break dam_break_3d ]
"""
import sys
import time
from pathlib import Path

import numpy as np
import sdot

OUT = Path( __file__ ).resolve().parents[ 1 ] / "public" / "examples"


class Fluid:
    """`n` particles of areas `masses` in the domain given by `target` ( density 1 where the fluid may go, 0 elsewhere ).
    The matter of particle `i` is its cell, Laguerre cell cut by the ball of radius `sqrt( w_i )`, whose area is `masses[ i ]`.
    The pressure pulls each dirac towards the barycenter `b_i` of its cell, with a stiffness `1 / eps^2`:

        p'' = g + ( b( p ) - p ) / eps^2

    integrated by velocity Verlet ( symplectic: the energy `1/2 m |v|^2 - m g.p + m |p - b|^2 / ( 2 eps^2 )` oscillates
    but does not drift ). One transport per step, started from the weights of the previous one.

    `sigma`: a SURFACE ENERGY `sigma sum_i |S_i inter V_i|`, the spheres inside the cells ( `sphere_areas`: the free surface, and
    the voids between particles, which have some ), and its force `- sigma grad E_s` -- through the transport, the masses staying
    what they are ( `jax.grad` of the plan ). Without it, nothing costs a void: the cells bead up into balls."""

    def __init__( self, target, masses, eps, gravity = ( 0.0, -1.0 ), sigma = 0.0, surface = "sphere" ):
        self.target, self.masses, self.eps = target, np.asarray( masses, dtype = float ), float( eps )
        self.gravity = np.asarray( gravity, dtype = float )
        self.sigma = float( sigma )
        self.surface_kind = surface
        self.problem = None
        self.surface_area = 0.0

    def plan( self, p, weights0 = None ):
        src = sdot.SumOfDiracs( p, self.masses, target_mass = None )
        if weights0 is not None:
            return sdot.OtProblem( src, self.target ).solve( sdot.Iterative( weights0 = weights0 ) )
        first = None
        if self.problem is None:
            self.problem = sdot.OtProblem( src, self.target )
            self.last = None
            if getattr( self, "start_weights", None ) is not None:   # ( those of `settle` )
                first = sdot.Iterative( weights0 = self.start_weights )
        self.problem.source = src
        try:
            plan = self.problem.solve( first )   # from the last weights
        except sdot.OtNotConverged:
            if self.last is None:
                raise
            # the last weights leave a cell empty at `p` ( a fast particle overtakes another ): get there in smaller moves
            plan = self.walk( *self.last, p )
        self.last = ( np.array( p ), np.asarray( plan.weights ) )
        return plan

    def walk( self, q, w, p, depth = 0 ):
        """the plan at `p` from the weights `w` of the plan at `q`, through the midpoint when the direct way fails"""
        try:
            return self.plan( p, weights0 = w )
        except sdot.OtNotConverged:
            if depth == 10:
                raise
        m = 0.5 * ( q + p )
        return self.walk( m, np.asarray( self.walk( q, w, m, depth + 1 ).weights ), p, depth + 1 )

    def surface( self, p, target, weights0 ):
        """`( E_s / sigma, its gradient with respect to the positions, barycenters, weights )`, the plan solved from `weights0`"""
        import jax
        import jax.numpy as jnp
        jax.config.update( "jax_enable_x64", True )

        def area( q ):
            plan = sdot.OtProblem( sdot.SumOfDiracs( q, jnp.asarray( self.masses ), target_mass = None ), target ).solve( sdot.Iterative( weights0 = weights0 ) )
            w = jnp.asarray( plan.weights )
            if self.surface_kind == "sphere":            # the spheres inside the cells
                A = jnp.sum( jnp.asarray( plan.sphere_areas ) )
            elif self.surface_kind == "ball":            # what the balls hold of nothing, `|B_i| - m_i` ( `r^2 = w`, the sink at 0 )
                ball = jnp.pi * w if q.shape[ 1 ] == 2 else 4 / 3 * jnp.pi * jnp.maximum( w, 0 ) ** 1.5
                A = jnp.sum( ball ) - float( np.sum( self.masses ) )
            else:                                        # "band": the empty part of each cell within a slightly bigger ball
                size = float( np.mean( self.masses ) ) ** ( 1 / q.shape[ 1 ] )
                c = ( 0.25 * size ) ** 2                 # `r'^2 = r^2 + c`: the same diagram, a sink at `- c`
                wider = sdot.PowerDiagram( q, w, distribution = target, sink_weight = - c )
                A = jnp.sum( jnp.asarray( wider.measures ) ) - float( np.sum( self.masses ) )
            return A, ( jnp.asarray( plan.barycenters ).reshape( q.shape ), w )

        ( A, ( b, w ) ), g = jax.value_and_grad( area, has_aux = True )( jnp.asarray( p ) )
        return float( A ), np.asarray( g ), np.asarray( b ), np.asarray( w )

    def acceleration( self, p ):
        if not self.sigma:
            plan = self.plan( p )
            b = np.asarray( plan.barycenters ).reshape( p.shape )
            return self.gravity + ( b - p ) / self.eps ** 2, b, np.asarray( plan.weights )
        w0 = np.asarray( self.plan( p ).weights )        # ( the weights at `p`, by the robust way: the start of the traced solve )
        self.surface_area, gA, b, w = self.surface( p, self.target, w0 )
        return self.gravity + ( b - p ) / self.eps ** 2 - self.sigma * gA / self.masses[ :, None ], b, w

    def settle( self, p, gate, tol = 1e-3, max_iter = 300 ):
        """a state AT REST behind `gate` ( a domain that holds the fluid, the dam ): `p = b( p ) + eps^2 f`, where the pressure
        holds the forces `f` ( gravity, and the surface's ): the acceleration is zero. Iterated as Lloyd shifted by the forces,
        `p <- b( p ) + eps^2 f` ( damped with a surface ), until it moves less than `tol` times the size of a particle. A lattice is not at rest: the particles of the free surface are not balls,
        and the springs inside do not hold the weight above them -- started from it, the fluid would first relax"""
        p = np.asarray( p, dtype = float )
        size = float( np.mean( self.masses ) ) ** ( 1 / p.shape[ 1 ] )
        problem = sdot.OtProblem( sdot.SumOfDiracs( p, self.masses, target_mass = None ), gate )
        # the first solve from equal weights whose balls cover the cells of the lattice: every cell holds its seed ( the start
        # by balls leaves cells empty on a lattice, where many seeds are equally far )
        settings = sdot.Iterative( weights0 = np.full( len( p ), p.shape[ 1 ] * size ** 2 ) )
        prev = None
        for it in range( max_iter ):
            for _ in range( 10 ):                        # a move that leaves a cell empty at the start: half of it
                problem.source = sdot.SumOfDiracs( p, self.masses, target_mass = None )
                try:
                    plan = problem.solve( settings if prev is None else sdot.Iterative( weights0 = prev[ 1 ] ) )
                    break
                except sdot.OtNotConverged:
                    if prev is None:
                        raise
                    p = 0.5 * ( prev[ 0 ] + p )
            settings = None
            b = np.asarray( plan.barycenters ).reshape( p.shape )
            self.start_weights = np.asarray( plan.weights )
            prev = ( p, self.start_weights )
            f = self.gravity
            if self.sigma:                               # ( and the surface holds itself )
                _, gA, b, _ = self.surface( p, gate, self.start_weights )
                f = f - self.sigma * gA / self.masses[ :, None ]
            # ( damped with a surface: the full step overshoots, the surface force being stiff )
            q = p + ( 0.3 if self.sigma else 1.0 ) * ( b + self.eps ** 2 * f - p )
            move = np.abs( q - p ).max()
            p = q
            if move < tol * size:
                break
        print( f"settled in { it + 1 } iterations ( last move { move / size:.1e} particle )" )
        return p

    def run( self, p, dt, duration, nb_frames = 100 ):
        """the frames `( t, positions, weights, energy )`, `nb_frames + 1` of them evenly spaced in time"""
        p = np.asarray( p, dtype = float )
        v = np.zeros_like( p )
        a, b, w = self.acceleration( p )
        nb_steps = int( round( duration / dt ) )
        every = max( 1, nb_steps // nb_frames )
        frames, t0 = [ ( 0.0, p, w, self.energy( p, v, b ) ) ], time.perf_counter()
        for k in range( 1, nb_steps + 1 ):
            v = v + 0.5 * dt * a
            p = p + dt * v
            a, b, w = self.acceleration( p )
            v = v + 0.5 * dt * a
            if k % every == 0:
                frames.append( ( k * dt, p, w, self.energy( p, v, b ) ) )
                print( f"t = { k * dt:.3f}  energy = { frames[ -1 ][ 3 ]:.6f}  max |p - b| = { np.abs( p - b ).max():.2e}  "
                       f"( { ( time.perf_counter() - t0 ) / k * 1e3:.0f} ms per step )", flush = True )
        return frames

    def energy( self, p, v, b ):
        m = self.masses[ :, None ]
        return float( np.sum( m * ( 0.5 * v ** 2 - p * self.gravity + 0.5 * ( p - b ) ** 2 / self.eps ** 2 ) ) ) + self.sigma * self.surface_area

    def write_viewer( self, frames, name, title, ball_resolution = 12 ):
        """the matter only, the arcs and the spheres coarse ( `ball_resolution` ): the page holds every frame"""
        viz = sdot.Visualizer( title = title, frame_axis = "time", frame_value = 0.0 )
        for k, ( t, p, w, _ ) in enumerate( frames ):
            if k:
                viz.new_frame( round( t, 3 ) )
            self.plan( p, weights0 = w ).add_to_viz( viz, seeds = False, opacity_beyond_balls = 0, ball_resolution = ball_resolution,
                                                     support_only = False, domain = True, domain_color = "#888888" )
        viz.write_html( OUT / f"free-surface-{ name }.html" )


def hexagonal( a, inside ):
    """the points of a hexagonal lattice of spacing `a` ( rows `a sqrt( 3 ) / 2` apart ) for which `inside( x, y )`; the
    cells of the interior are hexagons of area `a^2 sqrt( 3 ) / 2`"""
    h = a * np.sqrt( 3 ) / 2
    j, i = np.meshgrid( np.arange( int( 4 / h ) ), np.arange( int( 4 / a ) ), indexing = "ij" )
    x, y = ( i + np.where( j % 2, 0.75, 0.25 ) ) * a, ( j + 0.5 ) * h
    keep = inside( x, y )
    return np.stack( [ x[ keep ], y[ keep ] ], axis = 1 )


def cubic( a, inside, jitter = 1e-3 ):
    """the centers of the cubes of side `a` for which `inside( x, y, z )`, moved by `jitter a` at most: the cells are the cubes,
    whose faces lie on the walls of a box aligned with the lattice -- the fluid is full there. ( The jitter breaks the vertices
    where eight cells meet. )"""
    g = ( np.arange( int( 4 / a ) ) + 0.5 ) * a
    p = np.stack( np.meshgrid( g, g, g, indexing = "ij" ), axis = -1 ).reshape( -1, 3 )
    p = p[ inside( p[ :, 0 ], p[ :, 1 ], p[ :, 2 ] ) ]
    return p + jitter * a * np.random.default_rng( 0 ).uniform( -1, 1, p.shape )


def write_card( fluid, frame ):
    """the picture of the gallery: what each particle holds, on a transparent background"""
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    from sdot.PowerDiagram import _disc_in_polygon

    _, p, w, _ = frame
    pd = fluid.plan( p, weights0 = w ).power_diagram()
    pos, r2 = np.asarray( pd.positions ).reshape( -1, 2 ), np.asarray( pd.weights ).reshape( -1 )
    colors = plt.get_cmap( "Blues" )( np.linspace( 0.45, 0.9, len( pos ) ) )[ np.random.default_rng( 0 ).permutation( len( pos ) ) ]
    fig = plt.figure( figsize = ( 8, 6 ) )
    ax = fig.add_axes( ( 0, 0, 1, 1 ) )
    for i in range( len( pos ) ):
        c = _disc_in_polygon( pd.cells.vertices( i ), pos[ i ], np.sqrt( max( r2[ i ], 0 ) ) )
        if len( c ) >= 3:
            ax.fill( c[ :, 0 ], c[ :, 1 ], color = colors[ i ], linewidth = 0.3, edgecolor = "#ffffff" )
    ax.plot( [ 0, 0, 2 ], [ 1, 0, 0 ], color = "#555555", linewidth = 2 )
    ax.set_xlim( -0.03, 1.37 )   # 4:3 like the cards of the gallery
    ax.set_ylim( -0.03, 1.02 )
    ax.set_aspect( "equal" )
    ax.axis( "off" )
    fig.savefig( OUT / "img" / "free-surface-fluid.png", dpi = 100, transparent = True )
    plt.close( fig )


# -- the cases ----------------------------------------------------------------------------------------------------------------

def dam_break( a = 0.025, eps = 0.02, sigma = 0.6, surface = "band" ):
    """a column of water 0.5 wide and 0.8 high against the left wall of a box 2 x 1, released at t = 0"""
    target = sdot.Box( frame = np.array( [ 2.0, 1.0 ] ), target_mass = 2.0 )
    p = hexagonal( a, lambda x, y: ( x < 0.5 ) & ( y < 0.8 ) )
    print( f"dam break: { len( p ) } particles" )
    fluid = Fluid( target, np.full( len( p ), a * a * np.sqrt( 3 ) / 2 ), eps, sigma = sigma, surface = surface )
    gate = sdot.Box( frame = np.array( [ 0.5, 1.0 ] ), target_mass = 0.5 )
    frames = fluid.run( fluid.settle( p, gate ), dt = eps / 10, duration = 3.0, nb_frames = 60 )
    fluid.write_viewer( frames, "dam-break", "Dam break" )
    write_card( fluid, frames[ 15 ] )
    return frames


def dam_break_3d( a = 0.1, eps = 0.02, sigma = 0.02 ):
    """the same in 3D, `y` upwards: a column 0.5 wide, 0.8 high and 0.5 deep in a box 2 x 1 x 0.5, as deep as the column"""
    target = sdot.Box( frame = np.array( [ 2.0, 1.0, 0.5 ] ), target_mass = 1.0 )
    p = cubic( a, lambda x, y, z: ( x < 0.5 ) & ( y < 0.8 ) & ( z < 0.5 ) )
    print( f"dam break 3d: { len( p ) } particles" )
    fluid = Fluid( target, np.full( len( p ), a ** 3 ), eps, gravity = ( 0.0, -1.0, 0.0 ), sigma = sigma )
    gate = sdot.Box( frame = np.array( [ 0.5, 1.0, 0.5 ] ), target_mass = 0.25 )
    frames = fluid.run( fluid.settle( p, gate ), dt = eps / 10, duration = 1.5, nb_frames = 15 )
    fluid.write_viewer( frames, "dam-break-3d", "Dam break in 3D", ball_resolution = 10 )
    return frames


CASES = { "dam_break": dam_break, "dam_break_3d": dam_break_3d }

if __name__ == "__main__":
    ( OUT / "img" ).mkdir( parents = True, exist_ok = True )
    for name in sys.argv[ 1: ] or CASES:
        CASES[ name ]()
