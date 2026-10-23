"""The 2D transport of a sum of PYRAMIDS onto an image whose pixels are DIRACS: the smooth counterpart, on a detector, of
`ProjectedSumOfTents` ( a radiograph against the projections of 3D objects ).

THE MODEL. The source is continuous: n pyramids, each the product of two tents ( 1 - |u - u_i| / w_i )_+ ( 1 - |v - v_i| / w_i )_+,
of mass m_i, RASTERIZED EXACTLY on a fine grid ( `scale` x `scale` cells per pixel: the mass of each pyramid in each fine
cell, the product of its two 1D tent masses; the density uniform inside a fine cell ). The target is discrete: a dirac at
the center of each pixel, of mass the pixel's. The semi-discrete transport between them is sdot's ( `OtProblem`, Newton on
the dual weights of the pixels, CPU or card ), warm started from the weights of the last call. It is smooth in the
pyramids' positions and widths ( no kink when two of them cross ). Its derivative, by the envelope theorem: the Kantorovich
potential integrated over each fine cell ( the VJP of the diagram's moments at the solved weights, in doubles: their
differences between neighbour cells are what the gradient is made of ), through the normalization, then the raster ( jax ).

A uniform `background` ( 1e-3 of the mass by default ) is added to both sides: every pixel-dirac needs source mass around
it, and nearly empty pixels make Newton crawl. The mass of a pyramid that leaves the grid stays in its border cells ( the grid's two extreme edges sent to -inf / +inf,
as the grid method of `ContinuousOtPlan1d` ).

    pt = PixelTransport2d( values, origin, pixel_size, scale = 4 )
    cost = pt.cost( uv, half_widths )
    cost, ( d_uv, d_half_widths ) = pt.value_and_grad( uv, half_widths )

`values` `[ N0, N1 ]`, pixel ( i, j ) covering origin + [ i, i + 1 ] x du, [ j, j + 1 ] x dv; `uv` `[ n, 2 ]`, `half_widths`
`[ n ]` ( or a scalar ), `masses` `[ n ]` ( equal by default ). Both sides are normalized to mass 1.
"""
import numpy as np

import jax
import jax.numpy as jnp

from .distributions.Image import Image
from .distributions.SumOfDiracs import SumOfDiracs
from .OtProblem import Iterative, OtProblem


def _tent_G( t ):
    """the tent's primitive on [ -1, 1 ] ( 0 / 1 outside )"""
    u = jnp.clip( t, -1, 1 )
    return 0.5 + u - u * jnp.abs( u ) / 2


def _axis_masses( x, w, x0, h, M, L ):
    """the masses, along one axis, of tents at x of half-widths w in the cells [ x0 + j h, x0 + ( j + 1 ) h ]: the first
    cell j0 [ n ] and the masses of the L cells from it [ n, L ] ( the end edges sent to -inf / +inf )"""
    j0 = jnp.clip( jnp.floor( ( x - w - x0 ) / h ).astype( jnp.int32 ), 0, max( M - L, 0 ) )
    e = j0[ :, None ] + jnp.arange( L + 1, dtype = jnp.int32 )[ None, : ]
    t = ( x0 + e.astype( x.dtype ) * h - x[ :, None ] ) / w[ :, None ]
    t = jnp.where( e <= 0, -2.0, jnp.where( e >= M, 2.0, t ) )
    return j0, jnp.diff( _tent_G( t ), axis = 1 )


def raster_pyramids( uv, half_widths, masses, origin, cell, shape, max_half_width ):
    """the DENSITY [ M0, M1 ] of the pyramids on the grid of cells `cell` = ( h0, h1 ) from `origin` ( exact cell
    masses, divided by the cells' area ); jax, differentiable. `max_half_width`: a concrete bound of the half-widths
    ( it sets the window of cells each pyramid writes )"""
    uv = jnp.asarray( uv )
    dt = uv.dtype
    n = uv.shape[ 0 ]
    w = jnp.broadcast_to( jnp.asarray( half_widths, dt ), ( n, ) )
    m = jnp.broadcast_to( jnp.asarray( masses, dt ), ( n, ) )
    h0, h1 = float( cell[ 0 ] ), float( cell[ 1 ] )
    M0, M1 = int( shape[ 0 ] ), int( shape[ 1 ] )
    wmax = float( max_half_width )
    L0 = min( M0, int( np.ceil( 2 * wmax / h0 - 1e-9 ) ) + 1 )
    L1 = min( M1, int( np.ceil( 2 * wmax / h1 - 1e-9 ) ) + 1 )
    i0, a0 = _axis_masses( uv[ :, 0 ], w, float( origin[ 0 ] ), h0, M0, L0 )
    j0, a1 = _axis_masses( uv[ :, 1 ], w, float( origin[ 1 ] ), h1, M1, L1 )
    cm = m[ :, None, None ] * a0[ :, :, None ] * a1[ :, None, : ]                        # [ n, L0, L1 ]
    ii = jnp.broadcast_to( ( i0[ :, None ] + jnp.arange( L0, dtype = jnp.int32 ) )[ :, :, None ], cm.shape )
    jj = jnp.broadcast_to( ( j0[ :, None ] + jnp.arange( L1, dtype = jnp.int32 ) )[ :, None, : ], cm.shape )
    grid = jnp.zeros( ( M0, M1 ), dt ).at[ ii.ravel(), jj.ravel() ].add( cm.ravel() )
    return grid / ( h0 * h1 )


# ── the derivative with respect to the source's density, by the envelope theorem ───────────────────────────────────
#
# W = sum_c mu_c psibar_c + sum_j w_j nu_j at the optimal weights, mu_c the normalized mass of fine cell c, psibar_c the
# mean over the cell of the Kantorovich potential psi( x ) = |x - p_j|^2 - w_j on the Laguerre cell of pixel j. With the
# density v_c of the cells ( mass v_c A_c, normalized by S = sum_c v_c A_c ): dW / dv_c = ( A_c / S ) ( psibar_c - sum mu psibar ).
# The pixels are a regular grid and their weights small: a fine cell meets only the Laguerre cells of the 3 x 3 pixels
# around its own, each the square clipped by the bisectors with that pixel's 8 neighbours ( Sutherland-Hodgman, at most
# 4 + 8 vertices ), and the quadratic is integrated exactly on each piece. The pieces must tile the square ( `coverage` ):
# where they do not, the weights reach further and the caller falls back on sdot's derivative.

_MAXV = 13


def _clip( verts, count, nrm, c ):
    """the convex polygon ( `verts` [ K, 2 ], `count` of them ) clipped by n . x <= c"""
    K = verts.shape[ 0 ]
    idx = jnp.arange( K )
    valid = idx < count
    nxt = jnp.where( idx + 1 < count, idx + 1, 0 )
    a, b = verts, verts[ nxt ]
    da, db = a @ nrm - c, b @ nrm - c
    ina, inb = da <= 0, db <= 0
    t = da / jnp.where( da - db == 0, 1.0, da - db )
    cross = a + t[ :, None ] * ( b - a )
    emit_a = valid & ina
    emit_x = valid & ( ina != inb )
    # the output in order: per edge, a ( if inside ), then the crossing ( if any )
    pts = jnp.stack( [ a, cross ], 1 ).reshape( 2 * K, 2 )
    keep = jnp.stack( [ emit_a, emit_x ], 1 ).reshape( 2 * K )
    pos = jnp.cumsum( keep ) - 1
    out = jnp.zeros( ( K + 1, 2 ), verts.dtype ).at[ jnp.where( keep, pos, K ) ].set( pts )
    return out[ :K ], jnp.minimum( jnp.sum( keep ), K )


def _poly_integrals( verts, count, q ):
    """area and integral of |x - q|^2 over a convex polygon ( fan from its first vertex )"""
    K = verts.shape[ 0 ]
    v = verts - q
    i = jnp.arange( 1, K - 1 )
    ok = i + 1 < count
    a, b, c = v[ 0 ], v[ i ], v[ i + 1 ]
    T = 0.5 * ( ( b[ :, 0 ] - a[ 0 ] ) * ( c[ :, 1 ] - a[ 1 ] ) - ( b[ :, 1 ] - a[ 1 ] ) * ( c[ :, 0 ] - a[ 0 ] ) )
    T = jnp.where( ok, T, 0.0 )
    s2 = ( a @ a ) + jnp.sum( b * b, 1 ) + jnp.sum( c * c, 1 ) + b @ a + jnp.sum( b * c, 1 ) + c @ a
    return jnp.sum( T ), jnp.sum( T * s2 / 6 )


def potential_means( weights, origin, pixel_size, scale ):
    """( psibar [ M0, M1 ], coverage [ M0, M1 ] ): the mean of the Kantorovich potential over each fine cell, and the
    fraction of the cell the 3 x 3 neighbouring Laguerre cells cover ( 1 when the result is exact )"""
    w = jnp.asarray( weights )
    N0, N1 = w.shape
    du, dv = float( pixel_size[ 0 ] ), float( pixel_size[ 1 ] )
    o0, o1 = float( origin[ 0 ] ), float( origin[ 1 ] )
    s = int( scale )
    h0, h1 = du / s, dv / s
    M0, M1 = N0 * s, N1 * s
    dt = w.dtype
    offs = jnp.array( [ ( a, b ) for a in ( -1, 0, 1 ) for b in ( -1, 0, 1 ) ], jnp.int32 )         # [ 9, 2 ]
    nbr = jnp.array( [ ( a, b ) for a in ( -1, 0, 1 ) for b in ( -1, 0, 1 ) if ( a, b ) != ( 0, 0 ) ], jnp.int32 )   # [ 8, 2 ]
    center = lambda i, j: jnp.stack( [ o0 + ( i + 0.5 ) * du, o1 + ( j + 0.5 ) * dv ] ).astype( dt )

    def one_cell( a, b ):
        x0, y0 = o0 + a * h0, o1 + b * h1
        sq = jnp.zeros( ( _MAXV, 2 ), dt ).at[ :4 ].set( jnp.array( [ [ x0, y0 ], [ x0 + h0, y0 ], [ x0 + h0, y0 + h1 ], [ x0, y0 + h1 ] ], dt ) )
        pi, pj = a // s, b // s

        def piece( off ):
            i, j = pi + off[ 0 ], pj + off[ 1 ]
            inside = ( i >= 0 ) & ( i < N0 ) & ( j >= 0 ) & ( j < N1 )
            ic, jc = jnp.clip( i, 0, N0 - 1 ), jnp.clip( j, 0, N1 - 1 )
            p, wp = center( ic, jc ), w[ ic, jc ]
            verts, count = sq, jnp.int32( 4 )
            for k in range( 8 ):
                ik, jk = ic + nbr[ k, 0 ], jc + nbr[ k, 1 ]
                ok = ( ik >= 0 ) & ( ik < N0 ) & ( jk >= 0 ) & ( jk < N1 )
                ikc, jkc = jnp.clip( ik, 0, N0 - 1 ), jnp.clip( jk, 0, N1 - 1 )
                q, wq = center( ikc, jkc ), w[ ikc, jkc ]
                # |x - p|^2 - wp <= |x - q|^2 - wq  <=>  2 x . ( q - p ) <= |q|^2 - |p|^2 - wq + wp
                nrm = 2 * ( q - p )
                c = q @ q - p @ p - wq + wp
                nv, nc = _clip( verts, count, nrm, c )
                verts = jnp.where( ok, nv, verts )
                count = jnp.where( ok, nc, count )
            area, quad = _poly_integrals( verts, count, p )
            area = jnp.where( inside & ( count >= 3 ), area, 0.0 )
            quad = jnp.where( inside & ( count >= 3 ), quad, 0.0 )
            return area, quad - wp * area

        areas, ints = jax.vmap( piece )( offs )
        A = h0 * h1
        return jnp.sum( ints ) / A, jnp.sum( areas ) / A

    aa, bb = jnp.meshgrid( jnp.arange( M0 ), jnp.arange( M1 ), indexing = "ij" )
    psi, cov = jax.vmap( jax.vmap( one_cell ) )( aa, bb )
    return psi, cov


class PixelTransport2d:
    """see the module"""

    def __init__( self, values, origin, pixel_size, scale = 4, tol = 1e-10, max_iter = 100, precision = "fp64", background = 1e-3,
                  dirac_split = 1 ):
        values = np.asarray( values, dtype = np.float64 )
        # the same background on the image as on the pyramids ( see `background` ): its pixels never empty
        values = values + float( background ) * values.mean()
        #: `dirac_split` x `dirac_split` diracs per pixel, its mass shared evenly: a finer target. One dirac per pixel PULLS
        #: the pyramids to the pixels' centers ( the cost is lower on a dirac ), which every view does alike along a common
        #: axis ( the rotation axis of a tomography: the points' z snap to the detector rows ); the fine grid keeps its
        #: cells ( `scale` per pixel, a multiple of the split )
        self.dirac_split = s = int( dirac_split )
        if scale % s:
            raise ValueError( f"PixelTransport2d: `scale` ( { scale } ) must be a multiple of `dirac_split` ( { s } )" )
        #: split: the same problem with one dirac per pixel, whose solved weights start the first solve ( a cold Newton on
        #: split x split times more diracs crawls; a pixel's weight is a close guess for its sub-diracs )
        self._coarse = PixelTransport2d( values, origin, pixel_size, scale = scale, tol = tol * s * s, max_iter = max_iter,
                                         precision = precision, background = 0.0 ) if s > 1 else None
        if s > 1:
            values = np.repeat( np.repeat( values, s, axis = 0 ), s, axis = 1 )
            pixel_size = ( float( pixel_size[ 0 ] ) / s, float( pixel_size[ 1 ] ) / s )
            scale = scale // s
        self.N0, self.N1 = values.shape
        self.origin = np.asarray( origin, dtype = np.float64 )
        self.du, self.dv = float( pixel_size[ 0 ] ), float( pixel_size[ 1 ] )
        self.scale = int( scale )
        self.cell = ( self.du / self.scale, self.dv / self.scale )
        self.shape = ( self.N0 * self.scale, self.N1 * self.scale )
        i, j = np.meshgrid( np.arange( self.N0 ), np.arange( self.N1 ), indexing = "ij" )
        self.centers = np.stack( [ self.origin[ 0 ] + ( i.ravel() + 0.5 ) * self.du, self.origin[ 1 ] + ( j.ravel() + 0.5 ) * self.dv ], 1 )
        mass = values.ravel() * self.du * self.dv
        if ( mass <= 0 ).any():
            raise ValueError( "PixelTransport2d: every pixel needs a positive mass ( a positive `background` )" )
        self.masses = mass / mass.sum()
        self.settings = dict( tol = tol, max_iter = max_iter, precision = precision )
        #: a uniform density added to the pyramids' raster, this fraction of their mass, and to the image ( a fraction of
        #: its mean ): every pixel-dirac needs a source around it. Measured ( 64 x 16 pixels, 1000 pyramids moved by 0.3
        #: pixel ): 161 Newton steps at 1e-6, 9 at 1e-3, 6 at 1e-1
        self.background = float( background )
        #: the dual weights of the pixels of the last solve ( the next one starts from them ), `None` at first
        self.weights = None
        #: the solver's statistics of the last solve ( `nb_iter`, `status`... )
        self.stats = None

    def _solve_into( self, density ):
        """a solve whose weights are kept ( the warm start of the next one )"""
        plan = self._solve( density )
        self.weights = np.asarray( plan.weights.raw )
        return plan

    def _put( self, x ):
        return jnp.asarray( x )

    def _solve( self, density ):
        if self.weights is None and self._coarse is not None:
            # ( the coarse problem has the same fine grid: the same density )
            self._coarse._solve_into( density )
            s = self.dirac_split
            wc = self._coarse.weights.reshape( self._coarse.N0, self._coarse.N1 )
            self.weights = np.repeat( np.repeat( wc, s, axis = 0 ), s, axis = 1 ).ravel()
        w0 = None if self.weights is None else self._put( self.weights )
        it = Iterative( weights0 = w0, on_failure = "ignore", **self.settings )
        img = Image( values = self._put( density ), origin = list( self.origin ), pixel_frame = [ [ self.cell[ 0 ], 0.0 ], [ 0.0, self.cell[ 1 ] ] ] )
        return OtProblem( SumOfDiracs( self._put( self.centers ), weights = self._put( self.masses ) ), img ).solve( it )

    def raster( self, uv, half_widths, masses = None, max_half_width = None ):
        n = jnp.asarray( uv ).shape[ 0 ]
        m = jnp.full( n, 1.0 / n ) if masses is None else jnp.asarray( masses ) / jnp.sum( jnp.asarray( masses ) )
        if max_half_width is None:
            max_half_width = float( np.max( np.asarray( half_widths ) ) )
        d = raster_pyramids( uv, half_widths, m, self.origin, self.cell, self.shape, max_half_width )
        area = self.shape[ 0 ] * self.cell[ 0 ] * self.shape[ 1 ] * self.cell[ 1 ]
        return d + self.background / area

    def cost( self, uv, half_widths, masses = None ):
        plan = self._solve( self.raster( uv, half_widths, masses ) )
        self.weights = np.asarray( plan.weights.raw )
        self.stats = { k: np.asarray( v ) for k, v in plan.stats.items() }
        return float( plan.cost.raw )

    def value_and_grad_adjoint( self, uv, half_widths, masses = None ):
        """`value_and_grad` through sdot's own derivative of the plan ( a frozen Newton step, two laplacian solves: ~30
        times the solve ); the reference of `value_and_grad`"""
        uv = jnp.asarray( uv, jnp.float64 )
        n = uv.shape[ 0 ]
        hw = jnp.broadcast_to( jnp.asarray( half_widths, jnp.float64 ), ( n, ) )
        wmax = float( np.max( np.asarray( hw ) ) )
        density, vjp = jax.vjp( lambda p, w: self.raster( p, w, masses, wmax ), uv, hw )
        dens = jax.lax.stop_gradient( density )
        ( cost, weights ), g_dens = jax.value_and_grad( lambda d: self._cost_of_density( d ), has_aux = True )( dens )
        self.weights = np.asarray( weights )
        g_uv, g_hw = vjp( g_dens )
        return float( cost ), ( g_uv, g_hw )

    def _cost_of_density( self, density ):
        plan = self._solve( density )
        return plan.cost.raw, jax.lax.stop_gradient( plan.weights.raw )

    def potential_integrals( self, weights ):
        """`[ M0, M1 ]`: per fine cell, the integral over it of the Kantorovich potential psi( x ) = |x - p_j|^2 - w_j on the
        Laguerre cell of pixel j, at the given weights ( `PowerDiagram.potential_integrals`: one sweep, piece by piece,
        in doubles -- their differences between neighbour cells are what the gradient is made of )"""
        from .PowerDiagram import PowerDiagram
        img = Image( values = self._put( jnp.ones( self.shape ) ), origin = list( self.origin ), pixel_frame = [ [ self.cell[ 0 ], 0.0 ], [ 0.0, self.cell[ 1 ] ] ] )
        pd = PowerDiagram( self._put( self.centers ), self._put( weights ), distribution = img, kernel_dtype = "FP64" )
        r = pd.potential_integrals()
        return jnp.asarray( getattr( r, "raw", getattr( r, "value", r ) ) )

    def value_and_grad( self, uv, half_widths, masses = None ):
        """the cost and its derivatives with respect to the positions [ n, 2 ] and the half-widths [ n ], by the envelope
        theorem: the solve is not differentiated; the derivative with respect to the raster is the potential integrated
        over each fine cell ( `potential_integrals` ) through the normalization of the source, then the raster's VJP"""
        uv = self._put( jnp.asarray( uv, jnp.float64 ) )
        n = uv.shape[ 0 ]
        hw = self._put( jnp.broadcast_to( jnp.asarray( half_widths, jnp.float64 ), ( n, ) ) )
        wmax = float( np.max( np.asarray( hw ) ) )
        density, vjp = jax.vjp( lambda p, w: self.raster( p, w, masses, wmax ), uv, hw )
        dens = jax.lax.stop_gradient( density )
        plan = self._solve( dens )
        cost = float( plan.cost.raw )
        self.weights = np.asarray( plan.weights.raw )
        self.stats = { k: np.asarray( v ) for k, v in plan.stats.items() }
        I = self.potential_integrals( self.weights )
        A = self.cell[ 0 ] * self.cell[ 1 ]
        S = jnp.sum( dens ) * A
        g_dens = ( I - A * jnp.sum( dens * I ) / S ) / S
        g_uv, g_hw = vjp( g_dens )
        return cost, ( g_uv, g_hw )
