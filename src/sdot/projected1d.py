"""1D transport of PROJECTED objects onto a fine detector grid, without any sort: the loss of a 2D
reconstruction from a sinogram, at the scale of 1e7 objects x 600 angles.

THE LOSS: for each angle a ( normal n_a ), the objects ( centers x_i, half-widths w_i, equal masses
1/N ) project onto the detector as the density mu_a( s ) = sum_i f_i( s - x_i.n_a ) / N, with f_i a
profile of mass 1 whose PRIMITIVE is known in closed form; the loss is sum_a W2^2( mu_a, nu_a ),
nu_a being the measured profile ( piecewise constant on the detector bins ), both normalized to
mass 1.

THE PROFILES ( `shape` ):
  "dome"  the projection, at EVERY angle, of the radial density ( 1 - |x|^2/r^2 )_+^( k - 1/2 ):
          f( s ) = c_k ( 1 - t^2 )_+^k / r, t = s / r ( Abel transform ). k = 1 is the closest to a
          disk ( k = 1/2 ), k = 2 has a continuous derivative at the edge.
  "tent"  ( 1 - |t| )_+ / w with w = ONE fine cell: a dirac seen by the grid ( a linear deposit ).

HOW ( three steps, none of which materializes an [ angles, N ] array ):
  1. DEPOSIT: every object puts into each fine cell [ e_j, e_j+1 ] it covers its EXACT mass
     ( G( t_j+1 ) - G( t_j ) ) / N, G the primitive of f: mu[ angles, M ], M = scale x bins.
  2. GRID TRANSPORT, per angle: W2^2 between two densities that are piecewise constant on the same
     grid, exactly ( merge of the two cumulative distributions, closed-form integral of the squared
     difference of the two quantile functions on each piece ); its derivative with respect to the
     cell masses, g = dL/dmu, by autodiff ( O( angles x M ), small ).
  3. GATHER: dL/dx_i = sum_a n_a sum_j g[ a, j ] dmu[ a, j ]/ds, by the same primitives, and the
     same for the half-widths.
The only approximation is the grid: mu is taken as uniform inside each fine cell. It vanishes as
`scale` grows ( see `exact_w2` for the reference without grid ).

This is the PURE JAX version: the reference, and the path wherever no kernel exists. Steps 1 and 3 are
the hot spots ( O( N x angles x cells per object ) ); they are written so that fused kernels ( one
launch for every angle, the projection recomputed on the fly ) can take their place.
"""
from math import comb, gamma, pi, sqrt

import jax
import jax.numpy as jnp
import numpy as np


# ── the profiles ─────────────────────────────────────────────────────────────

def _dome_coeffs( k ):
    """c_k and the coefficients of G( t ) - 1/2 = c_k sum_j C( k, j ) ( -1 )^j t^( 2j+1 ) / ( 2j+1 )"""
    c = gamma( k + 1.5 ) / ( sqrt( pi ) * gamma( k + 1 ) )
    return c, [ c * comb( k, j ) * ( -1 ) ** j / ( 2 * j + 1 ) for j in range( k + 1 ) ]


def profile( shape, k = 1 ):
    """( f, G ): the normalized profile f( t ) on [ -1, 1 ] ( integral 1 ) and its primitive G, G( -1 ) = 0,
    G( 1 ) = 1 -- functions of jax arrays, defined everywhere ( 0 / 1 outside )."""
    if shape == "tent":
        def f( t ):
            return jnp.maximum( 1 - jnp.abs( t ), 0 )
        def G( t ):
            u = jnp.clip( t, -1, 1 )
            return 0.5 + u - u * jnp.abs( u ) / 2
        return f, G
    if shape == "dome":
        c, a = _dome_coeffs( k )
        def f( t ):
            return c * jnp.maximum( 1 - t * t, 0 ) ** k
        def G( t ):
            u = jnp.clip( t, -1, 1 )
            u2, p, s = u * u, u, 0.5
            for aj in a:
                s, p = s + aj * p, p * u2
            return s
        return f, G
    raise ValueError( f"unknown shape { shape!r} ( dome, tent )" )


# ── step 2: the transport between two piecewise-constant densities on the same grid ──

def grid_w2( mu, nu, h, Cn = None ):
    """W2^2 between the densities whose cell masses are `mu` and `nu` ( [ M ] each, same total ),
    uniform inside the cells [ j h, ( j + 1 ) h ]. Exact for that model; differentiable in `mu`.
    `Cn`: the cumulated masses of `nu` ( [ M + 1 ], from 0 ), when they are already known."""
    M = mu.shape[ 0 ]
    dt = mu.dtype
    zero = jnp.zeros( 1, dt )
    Cm = jnp.concatenate( [ zero, jnp.cumsum( mu ) ] )
    if Cn is None:
        Cn = jnp.concatenate( [ zero, jnp.cumsum( nu ) ] )
    tot = Cn[ -1 ]
    Cm = Cm * ( tot / Cm[ -1 ] )       # same total, exactly
    # the breakpoints of the two quantile functions, merged: the pieces where both are affine
    t = jnp.sort( jnp.concatenate( [ Cm[ 1:-1 ], Cn ] ) )
    t0, t1 = t[ :-1 ], t[ 1: ]
    tm = 0.5 * ( t0 + t1 )

    def quantile( C, m ):
        # the fraction of cell j reached, clipped to [ 0, 1 ]: under jit, XLA may evaluate the same cumulative
        # sum twice with different roundings, and a cell of nearly zero mass would then amplify the mismatch
        j = jnp.clip( jnp.searchsorted( C, tm, side = "right" ) - 1, 0, M - 1 )
        mj = m[ j ]
        inv = 1 / jnp.where( mj > 0, mj, 1 )
        def frac( t ):
            # the value clipped, the derivative of the unclipped expression ( a plain clip would zero the
            # derivative whenever a rounding puts the fraction at 1 + eps )
            u = ( t - C[ j ] ) * inv
            return u + jax.lax.stop_gradient( jnp.clip( u, 0, 1 ) - u )
        return ( ( j + frac( t0 ) ) * h, ( j + frac( t1 ) ) * h )

    m_scaled = jnp.diff( Cm )
    qm0, qm1 = quantile( Cm, m_scaled )
    qn0, qn1 = quantile( Cn, nu )
    d0, d1 = qm0 - qn0, qm1 - qn1
    return jnp.sum( ( t1 - t0 ) * ( d0 * d0 + d0 * d1 + d1 * d1 ) ) / ( 3 * tot )


# ── the loss ─────────────────────────────────────────────────────────────────

class ProjectedLoss1d:
    """`loss( points, radii )`: sum over the angles of W2^2( projected objects, measured profile ).

    `angles` [ A ] ( normals ( cos, sin ) ), `values` [ A, B ] the measured profiles on the bins
    [ s_min + b dw, s_min + ( b + 1 ) dw ] -- or [ K, A, B ]: K independent problems of the same geometry,
    solved together ( `points` [ K, N, 2 ], `radii` [ K, N ], the loss summed over the problems ).
    `scale`: fine cells per bin. `max_radius`: the largest half-width an object will have ( it sets how many
    cells one object may cover, a static size ); `None`: 4 fine cells. `chunk`: at most this many
    ( angle, object ) pairs alive at once in the pure jax passes. `backend`: "kernel" ( the fused passes of
    `sdot/Projected1d.h`, CPU or CUDA after loom's device ) or "jax" ( the reference ).

    Differentiable in `points` and `radii` ( a custom vjp: two passes over the objects in all ). Jit it from
    outside; the instance holds no traced state. `sharded( mesh )`: the same loss with the angles split
    between the devices of a mesh.
    """

    def __init__( self, angles, values, s_min, dw, scale = 8, shape = "dome", k = 1, max_radius = None,
                  dtype = jnp.float32, chunk = 1 << 24, backend = "kernel", grid_dtype = None ):
        angles = np.asarray( angles, dtype = np.float64 )
        values = np.asarray( values, dtype = np.float64 )
        self.batched = values.ndim == 3
        if not self.batched:
            values = values[ None ]
        K, A, B = values.shape
        self.K, self.A, self.B, self.scale, self.M = K, A, B, int( scale ), int( scale ) * B
        self.h = dw / self.scale
        self.s_min = float( s_min )
        self.shape, self.k, self.dtype, self.chunk = shape, k, dtype, int( chunk )
        # the grid transport works on cumulated masses in [ 0, 1 ], where a cell may hold 1e-8: in float32 the
        # quantiles inside such cells lose every digit, and the gradient with them ( measured: 47 % off at
        # 1e5 objects ). It is O( angles x M ), small: float64 whenever jax allows it
        if grid_dtype is None:
            grid_dtype = jnp.float64 if jax.config.jax_enable_x64 else dtype
        self.grid_dtype = grid_dtype
        mass = values * dw
        tot = mass.sum( axis = 2, keepdims = True )
        tot = np.where( tot > 0, tot, 1 )
        # the target, per fine cell: each bin's mass split evenly between its `scale` cells; and its cumulated
        # masses, computed here ( a constant under jit that XLA would otherwise fold, slowly )
        nu = np.repeat( mass / tot / self.scale, self.scale, axis = 2 )
        Cn = np.concatenate( [ np.zeros( ( K, A, 1 ) ), np.cumsum( nu, axis = 2 ) ], axis = 2 )
        #: what the angles carry: their normals [ A, 2 ], the target nu and its cumulated masses Cn [ K, A, M ( + 1 ) ]
        self.geometry = dict( normals = jnp.asarray( np.stack( [ np.cos( angles ), np.sin( angles ) ], 1 ), dtype ),
                              nu = jnp.asarray( nu, grid_dtype ), Cn = jnp.asarray( Cn, grid_dtype ) )
        if shape == "tent":
            self.max_halfwidth = self.h
        else:
            self.max_halfwidth = 4 * self.h if max_radius is None else float( max_radius )
        # cells one object may cover: its support [ s - w, s + w ] starts in cell j0 and ends at most ceil( 2w/h ) further
        self.L = int( np.ceil( 2 * self.max_halfwidth / self.h - 1e-9 ) ) + 1
        self.f, self.G = profile( shape, k )
        if backend not in ( "kernel", "jax" ):
            raise ValueError( f"unknown backend { backend!r} ( kernel, jax )" )
        self.backend = backend
        # the fused kernels' parameters, in the points' float ( [ 3 ] = 1 / N, set at each call )
        c, a = _dome_coeffs( k ) if shape == "dome" else ( 0.0, [ ] )
        self.params = jnp.asarray( [ self.s_min, self.h, self.max_halfwidth, 0.0, c, *a ], dtype )
        self._loss = self._make_loss()

    @property
    def normals( self ):
        return self.geometry[ "normals" ]

    def _batched( self, points, radii ):
        points = jnp.asarray( points, self.dtype )
        if not self.batched:
            points = points[ None ]
        if radii is None:
            radii = jnp.full( points.shape[ :2 ], self.max_halfwidth, self.dtype )
        radii = jnp.asarray( radii, self.dtype ).reshape( points.shape[ :2 ] )
        return points, radii

    def __call__( self, points, radii = None ):
        points, radii = self._batched( points, radii )
        g = self.geometry
        return self._loss( points, radii, g[ "normals" ], g[ "nu" ], g[ "Cn" ] )

    def sharded( self, mesh, axis = None ):
        """the loss with the ANGLES split between the devices of `mesh` ( along its axis `axis`, by default its
        only one ): each device deposits, transports and gathers its own angles, the points and radii being
        replicated; the losses and the gradients are summed between devices ( one all-reduce of N x 2 floats per
        gradient ). A must be a multiple of the axis size."""
        from jax.sharding import NamedSharding, PartitionSpec as P
        try:
            from jax import shard_map
        except ImportError:
            from jax.experimental.shard_map import shard_map
        axis = axis or mesh.axis_names[ 0 ]
        if self.A % mesh.shape[ axis ]:
            raise ValueError( f"{ self.A } angles cannot be split evenly between { mesh.shape[ axis ] } devices" )
        specs = dict( normals = P( axis, None ), nu = P( None, axis, None ), Cn = P( None, axis, None ) )
        # Cn restarts from 0 on every angle: it splits with its angles as it is
        geo = { k: jax.device_put( v, NamedSharding( mesh, specs[ k ] ) ) for k, v in self.geometry.items() }

        def varying( x ):
            # the replicated points seen as varying along the angles: the custom vjp then returns per-device
            # gradients, and the transpose of this cast sums them ( the all-reduce )
            if hasattr( jax.lax, "pcast" ):
                return jax.lax.pcast( x, axis, to = "varying" )
            return jax.lax.pvary( x, axis )

        def local( points, radii, normals, nu, Cn ):
            return jax.lax.psum( self._loss( varying( points ), varying( radii ), normals, nu, Cn ), axis )

        f = shard_map( local, mesh = mesh, in_specs = ( P(), P(), specs[ "normals" ], specs[ "nu" ], specs[ "Cn" ] ),
                       out_specs = P() )

        def loss( points, radii = None ):
            points, radii = self._batched( points, radii )
            return f( points, radii, geo[ "normals" ], geo[ "nu" ], geo[ "Cn" ] )
        return loss

    # -- the angle chunks ---------------------------------------------------

    def _angle_chunk( self, A, n ):
        """the largest divisor of A such that chunk x n <= self.chunk ( >= 1 )"""
        best = 1
        for c in range( 1, A + 1 ):
            if A % c == 0 and c * n <= self.chunk:
                best = c
        return best

    def _halfwidths( self, radii ):
        if self.shape == "tent":
            return jnp.full_like( radii, self.h )
        return jnp.minimum( radii, self.max_halfwidth )

    def _windows( self, normals, points, w ):
        """for a chunk of angles: s [ c, n ], the first cell j0 [ c, n ], and the edge parameters
        t_l = ( e_( j0 + l ) - s ) / w for l = 0 .. L, with the grid's two extreme edges sent to -inf / +inf
        ( the mass of an object that leaves the grid is kept in its border cells )"""
        s = normals @ points.T                                                   # [ c, n ]
        j0 = jnp.floor( ( s - w - self.s_min ) / self.h ).astype( jnp.int32 )
        j0 = jnp.clip( j0, 0, self.M - self.L )
        l = jnp.arange( self.L + 1, dtype = jnp.int32 )
        e = j0[ ..., None ] + l                                                  # [ c, n, L + 1 ]
        t = ( self.s_min + e.astype( self.dtype ) * self.h - s[ ..., None ] ) / w[ :, None ]
        t = jnp.where( e == 0, -2, jnp.where( e == self.M, 2, t ) )
        return s, j0, t

    # -- step 1 and step 3 ( batched: points [ K, N, 2 ] ) ---------------------

    def deposit( self, points, radii, normals = None ):
        """mu [ K, A, M ]: the exact mass of the projected objects in each fine cell ( [ A, M ] for points [ N, 2 ] )"""
        normals = self.normals if normals is None else normals
        points = jnp.asarray( points, self.dtype )
        if points.ndim == 2:
            return self.deposit( points[ None ], jnp.asarray( radii )[ None ], normals )[ 0 ]
        if self.backend == "kernel":
            return self._kernel_call( "deposit", points, radii, normals )
        return jax.vmap( self._deposit_one, in_axes = ( 0, 0, None ) )( points, radii, normals )

    def gather( self, points, radii, g, normals = None ):
        """( dL/dpoints [ K, N, 2 ], dL/dradii [ K, N ] ) from dg [ K, A, M + 1 ], the differences of dL/dmu between
        neighbour cells ( `edge_differences` ) -- unbatched in, unbatched out"""
        normals = self.normals if normals is None else normals
        points = jnp.asarray( points, self.dtype )
        if points.ndim == 2:
            gp, gr = self.gather( points[ None ], jnp.asarray( radii )[ None ], jnp.asarray( g )[ None ], normals )
            return gp[ 0 ], gr[ 0 ]
        if self.backend == "kernel":
            return self._kernel_call( "gather", points, radii, normals, g )
        return jax.vmap( self._gather_one, in_axes = ( 0, 0, 0, None ) )( points, radii, g, normals )

    def _deposit_one( self, points, radii, normals ):
        A, n = normals.shape[ 0 ], points.shape[ 0 ]
        c = self._angle_chunk( A, n * ( self.L + 1 ) )
        w = self._halfwidths( radii )
        inv_n = 1 / n

        def one( nc ):
            _, j0, t = self._windows( nc, points, w )
            m = jnp.diff( self.G( t ), axis = -1 ) * inv_n                       # [ c, n, L ]
            cells = j0[ ..., None ] + jnp.arange( self.L, dtype = jnp.int32 )
            rows = jnp.broadcast_to( jnp.arange( nc.shape[ 0 ] )[ :, None, None ], cells.shape )
            return jnp.zeros( ( nc.shape[ 0 ], self.M ), self.dtype ).at[ rows, cells ].add( m )

        return jax.lax.map( one, normals.reshape( A // c, c, 2 ) ).reshape( A, self.M )

    def _gather_one( self, points, radii, g, normals ):
        A, n = normals.shape[ 0 ], points.shape[ 0 ]
        c = self._angle_chunk( A, n * ( self.L + 1 ) )
        w = self._halfwidths( radii )
        inv_n = 1 / n

        def one( carry, xs ):
            nc, gc = xs
            _, j0, t = self._windows( nc, points, w )
            ft = self.f( t )                                                     # [ c, n, L + 1 ], 0 at the sent-away edges
            edges = j0[ ..., None ] + jnp.arange( self.L + 1, dtype = jnp.int32 )
            dgl = jnp.take_along_axis( gc[ :, None, : ], edges, axis = 2 )       # [ c, n, L + 1 ]
            # d m_l / ds = ( f( t_l ) - f( t_l+1 ) ) / ( w n ): summed by parts against g, sum_e f( t_e ) dg_e ( and the
            # same with t f( t ) for d/dw )
            ds = jnp.sum( dgl * ft, axis = -1 ) * ( inv_n / w )                  # [ c, n ]
            tf = jnp.where( jnp.abs( t ) <= 1, t * ft, 0 )
            dw = jnp.sum( dgl * tf, axis = -1 ) * ( inv_n / w )
            gp, gw = carry
            return ( gp + ds.T @ nc, gw + jnp.sum( dw, axis = 0 ) ), None

        # zeros that vary like the normals ( under `shard_map` the carry must keep the type of the body's output )
        zero = 0 * normals[ 0, 0 ]
        init = ( jnp.zeros_like( points ) + zero, jnp.zeros_like( radii ) + zero )
        ( gp, gw ), _ = jax.lax.scan( one, init, ( normals.reshape( A // c, c, 2 ), g.reshape( A // c, c, self.M + 1 ) ) )
        if self.shape == "tent":
            gw = jnp.zeros_like( radii )
        else:
            gw = jnp.where( radii < self.max_halfwidth, gw, 0 )
        return gp.astype( points.dtype ), gw.astype( radii.dtype )

    def _kernel_call( self, what, points, radii, normals, g = None ):
        """one fused pass over the objects, by `sdot/Projected1d.h` ( a CPU and a CUDA version, after loom's device )"""
        import loom
        from loom import Axis, RealTensor, ShapeVar
        from loom.compilation.FfiCode import FfiCode
        K, n = points.shape[ 0 ], points.shape[ 1 ]
        A = normals.shape[ 0 ]
        bits = dict( size = 8 * jnp.dtype( self.dtype ).itemsize )
        params = self.params.at[ 3 ].set( 1 / n )
        tpl = f"{ 0 if self.shape == 'tent' else 1 },{ self.k if self.shape == 'dome' else 0 },{ self.L }"
        common = "queue, args.inputs.points, args.inputs.radii, args.inputs.normals, args.inputs.params"
        num_k = Axis( ShapeVar( K ), name = "num_problem" )
        num_a, num_p = Axis( ShapeVar( A ), name = "num_angle" ), Axis( ShapeVar( n ), name = "num_point" )
        if what == "deposit":
            hist = RealTensor[ num_k, num_a, Axis( ShapeVar( self.M ), name = "num_cell" ), bits ]()
            loom.ffi_call( "projected1d_deposit",
                           FfiCode.inline( f"sdot::projected1d::deposit<{ tpl }>( { common }, args.outputs.hist );",
                                           includes = [ "sdot/Projected1d.h" ] ),
                           points = points, radii = radii, normals = normals, params = params,
                           hist = loom.out( hist ), has_dynamic_capacity = False )
            # a kernel's output carries no `shard_map` mark: typed as varying like the normals ( the angles of this
            # device ), or the transpose of the grid step would sum dL/dmu between devices
            return hist.value + 0 * normals[ 0, 0 ]
        gp = RealTensor[ num_k, num_p, Axis( ShapeVar( 2 ), name = "dim" ), bits ]()
        gr = RealTensor[ num_k, num_p, bits ]()
        loom.ffi_call( "projected1d_gather",
                       FfiCode.inline( f"sdot::projected1d::gather<{ tpl }>( { common }, args.inputs.g, "
                                       "args.outputs.grad_points, args.outputs.grad_radii );",
                                       includes = [ "sdot/Projected1d.h" ] ),
                       points = points, radii = radii, normals = normals, params = params, g = g,
                       grad_points = loom.out( gp ), grad_radii = loom.out( gr ), has_dynamic_capacity = False )
        return gp.value, gr.value

    def _kernel_grid( self, mu, nu, Cn, normals ):
        """the grid transport by `sdot/Projected1d.h::grid` ( a sweep per row, in double ): ( loss, dg [ K, A, M + 1 ] ),
        dg the differences of dL/dmu between neighbour cells ( what `gather` takes )"""
        import loom
        from loom import Axis, RealTensor, ShapeVar
        from loom.compilation.FfiCode import FfiCode
        K, A, M = mu.shape
        num_k, num_a = Axis( ShapeVar( K ), name = "num_problem" ), Axis( ShapeVar( A ), name = "num_angle" )
        loss = RealTensor[ num_k, num_a, dict( size = 64 ) ]()
        dg = RealTensor[ num_k, num_a, Axis( ShapeVar( M + 1 ), name = "num_edge" ), dict( size = 8 * jnp.dtype( self.dtype ).itemsize ) ]()
        loom.ffi_call( "projected1d_grid",
                       FfiCode.inline( "sdot::projected1d::grid( queue, args.inputs.mu, args.inputs.nu, args.inputs.Cn, "
                                       "args.inputs.params, args.outputs.loss, args.outputs.dg );",
                                       includes = [ "sdot/Projected1d.h" ] ),
                       mu = mu, nu = nu.astype( jnp.float64 ), Cn = Cn.astype( jnp.float64 ), params = self.params,
                       loss = loom.out( loss ), dg = loom.out( dg ), has_dynamic_capacity = False )
        # the kernel works in grid units ( h = 1 ); typed as varying like the normals under `shard_map` ( see `_kernel_call` )
        h2 = self.h * self.h
        zero = 0 * normals[ 0, 0 ]
        return jnp.sum( loss.value ) * h2 + zero.astype( jnp.float64 ), dg.value * jnp.asarray( h2, self.dtype ) + zero

    def edge_differences( self, g ):
        """dg [ ..., M + 1 ] from g = dL/dmu [ ..., M ]: g_e - g_e-1 on the inner edges, 0 on the two end ones, in g's
        float, then cast to the points' one"""
        d = jnp.diff( g, axis = -1 )
        pad = [ ( 0, 0 ) ] * ( g.ndim - 1 ) + [ ( 1, 1 ) ]
        return jnp.pad( d, pad ).astype( self.dtype )

    def grid_loss( self, mu, nu = None, Cn = None ):
        """step 2 for every problem and angle: sum W2^2( mu, nu ), in world units ( mu [ K, A, M ] )"""
        nu = self.geometry[ "nu" ] if nu is None else nu
        Cn = self.geometry[ "Cn" ] if Cn is None else Cn
        h = jnp.asarray( self.h, self.grid_dtype )
        mu = mu.astype( self.grid_dtype )
        if mu.ndim == 2:
            mu = mu[ None ]
        w2 = jax.vmap( jax.vmap( grid_w2, in_axes = ( 0, 0, None, 0 ) ), in_axes = ( 0, 0, None, 0 ) )
        return jnp.sum( w2( mu, nu, h, Cn ) )

    def _make_loss( self ):
        kernel_grid = self.backend == "kernel" and jax.config.jax_enable_x64

        def value_and_g( points, radii, normals, nu, Cn ):
            mu = self.deposit( points, radii, normals )
            if kernel_grid:
                return self._kernel_grid( mu, nu, Cn, normals )
            val, vjp = jax.vjp( lambda m: self.grid_loss( m, nu, Cn ), mu )
            ( g, ) = vjp( jnp.ones_like( val ) )          # ( ones_like: under shard_map, varying like val )
            return val, self.edge_differences( g )

        @jax.custom_vjp
        def loss( points, radii, normals, nu, Cn ):
            return value_and_g( points, radii, normals, nu, Cn )[ 0 ]

        def fwd( points, radii, normals, nu, Cn ):
            val, g = value_and_g( points, radii, normals, nu, Cn )
            return val, ( points, radii, normals, g )

        def bwd( res, ct ):
            points, radii, normals, dg = res
            gp, gr = self.gather( points, radii, dg * ct.astype( dg.dtype ), normals )
            # the cotangents typed like the primal inputs ( under `shard_map`, varying along the angles or not: a kernel's
            # outputs carry no such mark )
            return gp + jnp.zeros_like( points ), gr + jnp.zeros_like( radii ), None, None, None

        loss.defvjp( fwd, bwd )
        return loss


# ── the reference without grid ───────────────────────────────────────────────

def _pieces( shape, k, centers, halfwidths ):
    """the profiles as polynomial pieces in s: ( lo, hi, coefficients in s, low degree first )"""
    from numpy.polynomial import Polynomial as P
    out = [ ]
    for p, w in zip( centers, halfwidths ):
        u = P( [ -p / w, 1 / w ] )                       # u = ( s - p ) / w
        if shape == "tent":
            out.append( ( p - w, p, ( ( 1 + u ) / w ).coef ) )
            out.append( ( p, p + w, ( ( 1 - u ) / w ).coef ) )
        else:
            c, _ = _dome_coeffs( k )
            out.append( ( p - w, p + w, ( c * ( 1 - u * u ) ** k / w ).coef ) )
    return out


def exact_w2( shape, k, s, halfwidths, values, s_min, dw ):
    """W2^2( sum of the projected profiles / N, measured profile ) for ONE angle, without any grid: the
    events ( piece starts and ends ) are sorted and swept, the density being a polynomial between two
    events ( the coefficients of the active pieces add up ); then each piece is cut where the source CDF
    crosses a cumulated bin mass, and int ( s - T( s ) )^2 dmu is integrated by Gauss-Legendre ( exact:
    polynomials ). numpy, float64: a reference for tests, for small N."""
    from numpy.polynomial import Polynomial as P
    s = np.asarray( s, dtype = np.float64 )
    n = len( s )
    pieces = _pieces( shape, k, s, np.asarray( halfwidths, dtype = np.float64 ) )
    deg = max( len( c ) for _, _, c in pieces )
    ev = [ ]
    for lo, hi, c in pieces:
        c = np.pad( c, ( 0, deg - len( c ) ) ) / n
        ev.append( ( lo, c ) )
        ev.append( ( hi, -c ) )
    ev.sort( key = lambda e: e[ 0 ] )
    xs = np.array( [ e[ 0 ] for e in ev ] )
    coef = np.cumsum( np.array( [ e[ 1 ] for e in ev ] ), axis = 0 )            # density on [ xs[ q ], xs[ q + 1 ] ]

    values = np.asarray( values, dtype = np.float64 )
    nm = values * dw / ( values.sum() * dw )
    Cn = np.concatenate( [ [ 0 ], np.cumsum( nm ) ] )
    edges = s_min + dw * np.arange( len( values ) + 1 )

    gx, gw = np.polynomial.legendre.leggauss( ( 3 * deg + 4 ) // 2 + 1 )
    total, F = 0.0, 0.0
    for q in range( len( xs ) - 1 ):
        a, b = xs[ q ], xs[ q + 1 ]
        if b <= a:
            continue
        rho = P( coef[ q ] )
        R = rho.integ( lbnd = a )                    # the mass from a
        mq = R( b )
        if mq <= 0:
            continue
        # cut [ a, b ] where F + R( s ) crosses a cumulated bin mass
        levels = Cn[ ( Cn > F ) & ( Cn < F + mq ) ]
        cuts = [ a ]
        for lv in levels:
            x0, x1 = a, b
            for _ in range( 200 ):
                xm = 0.5 * ( x0 + x1 )
                if F + R( xm ) < lv:
                    x0 = xm
                else:
                    x1 = xm
            cuts.append( 0.5 * ( x0 + x1 ) )
        cuts.append( b )
        for c0, c1 in zip( cuts[ :-1 ], cuts[ 1: ] ):
            if c1 <= c0:
                continue
            xm = 0.5 * ( c0 + c1 )
            Fm = F + R( xm )
            j = min( max( np.searchsorted( Cn, Fm, side = "right" ) - 1, 0 ), len( nm ) - 1 )
            if nm[ j ] <= 0:
                continue
            x = xm + 0.5 * ( c1 - c0 ) * gx
            T = edges[ j ] + ( F + R( x ) - Cn[ j ] ) / nm[ j ] * dw
            total += 0.5 * ( c1 - c0 ) * np.sum( gw * ( x - T ) ** 2 * rho( x ) )
        F += mq
    return total


def morton_order( points, bits = 16 ):
    """a permutation that sorts the points along a Z curve: neighbours in memory are then neighbours in space, so
    the threads of a block project onto nearby cells ( the atomics of the deposit and the reads of the gather hit
    the same cache lines: 20x faster deposits at 1e7 objects on a 2080 Ti ). The loss does not depend on the order
    of the objects. jax arrays in, jax array out ( on the device of the points )."""
    p = jnp.asarray( points )
    lo, hi = jnp.min( p, axis = 0 ), jnp.max( p, axis = 0 )
    q = ( ( p - lo ) / jnp.maximum( hi - lo, 1e-30 ) * ( ( 1 << bits ) - 1 ) ).astype( jnp.uint32 )

    def spread( v ):
        v = v & jnp.uint32( 0xFFFF )
        v = ( v | ( v << 8 ) ) & jnp.uint32( 0x00FF00FF )
        v = ( v | ( v << 4 ) ) & jnp.uint32( 0x0F0F0F0F )
        v = ( v | ( v << 2 ) ) & jnp.uint32( 0x33333333 )
        v = ( v | ( v << 1 ) ) & jnp.uint32( 0x55555555 )
        return v

    return jnp.argsort( spread( q[ :, 0 ] ) | ( spread( q[ :, 1 ] ) << 1 ) )
