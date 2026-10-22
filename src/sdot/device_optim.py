"""Optimizers that run ON THE DEVICE: the whole loop is one compiled `lax.while_loop`, the host only sees
the result ( or a snapshot every `snapshot_every` steps ). One value + gradient evaluation per trial.

`lbfgs` and `gradient_descent` take `value_and_grad( x ) -> ( f, g )`, a jax-traceable function of an
array of any shape, and return `( x, info )` with `info` = { "losses": [ f_0, f_1, ... ] ( host numpy, the
loss after each step, f_0 the starting one ), "nb_steps", "nb_evals", "reason" }.

The stopping rules ( shared ): at most `max_iter` steps; after step k > `min_iter`, stop when the relative
decrease `( f_prev - f ) / max( |f_prev|, |f|, 1 )` <= `ftol`, or when no coordinate moved by more than
`disp_tol` in that step ( `None`: no such test ), or when the line search cannot decrease f.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


REASONS = { 0: "running", 1: "max_iter", 2: "ftol", 3: "disp_tol", 4: "line search failed", 5: "drift_tol" }


def _vdot( a, b ):
    return jnp.vdot( a.reshape( -1 ), b.reshape( -1 ) )


def _two_loop( g, S, Y, rho, head, count, m ):
    """H g by the two-loop recursion on the ring buffers S, Y [ m, D ] ( newest at head - 1 )"""
    q = g.reshape( -1 )

    def back( i, carry ):
        q, alpha = carry
        j = ( head - 1 - i ) % m
        valid = i < count
        a = jnp.where( valid, rho[ j ] * jnp.vdot( S[ j ], q ), 0 )
        return q - a * Y[ j ], alpha.at[ j ].set( a )

    q, alpha = jax.lax.fori_loop( 0, m, back, ( q, jnp.zeros( m, q.dtype ) ) )
    jn = ( head - 1 ) % m
    yy = jnp.vdot( Y[ jn ], Y[ jn ] )
    gamma = jnp.where( count > 0, jnp.vdot( S[ jn ], Y[ jn ] ) / jnp.where( yy > 0, yy, 1 ), 1 )
    r = gamma * q

    def fwd( i, r ):
        j = ( head - count + i ) % m
        valid = i < count
        b = rho[ j ] * jnp.vdot( Y[ j ], r )
        return jnp.where( valid, r + S[ j ] * ( alpha[ j ] - b ), r )

    r = jax.lax.fori_loop( 0, m, fwd, r )
    return r.reshape( g.shape )


def _stop_reason( k, f_prev, f, disp, ok, max_iter, min_iter, ftol, disp_tol, f_scale = 1.0 ):
    eligible = k > min_iter
    rel = ( f_prev - f ) / jnp.maximum( jnp.maximum( jnp.abs( f_prev ), jnp.abs( f ) ), f_scale )
    reason = jnp.int32( 0 )
    reason = jnp.where( k >= max_iter, 1, reason )
    reason = jnp.where( eligible & ( rel <= ftol ), 2, reason )
    if disp_tol is not None:
        reason = jnp.where( eligible & ( disp <= disp_tol ), 3, reason )
    reason = jnp.where( eligible & ~ok, 4, reason )
    return reason


@partial( jax.jit, static_argnames = ( "vg", "m", "nb", "max_iter", "min_iter", "ftol", "disp_tol", "max_ls", "c1", "f_scale" ) )
def _lbfgs_segment( state, vg, m, nb, max_iter, min_iter, ftol, disp_tol, max_ls, c1, f_scale ):
    """at most `nb` more steps from `state`"""

    def cond( st ):
        return ( st[ "reason" ] == 0 ) & ( st[ "seg" ] < nb )

    def body( st ):
        x, f, g, k = st[ "x" ], st[ "f" ], st[ "g" ], st[ "k" ]
        d = -_two_loop( g, st[ "S" ], st[ "Y" ], st[ "rho" ], st[ "head" ], st[ "count" ], m )
        gd = _vdot( g, d )
        # not a descent direction: forget the memory, take the gradient
        bad = ~( gd < 0 )
        d = jnp.where( bad, -g, d )
        gd = jnp.where( bad, -_vdot( g, g ), gd )
        count = jnp.where( bad, 0, st[ "count" ] )
        # first step ( no memory ): a unit step along -g / |g|
        t0 = jnp.where( count == 0, 1 / jnp.maximum( jnp.sqrt( _vdot( g, g ) ), 1e-30 ), 1.0 ).astype( x.dtype )

        def ls_cond( c ):
            t, fn, gn, i, ok = c
            return ( ~ok ) & ( i < max_ls )

        def ls_body( c ):
            t, _, _, i, _ = c
            t = jnp.where( i == 0, t, t * 0.5 )
            fn, gn = vg( x + t * d )
            ok = jnp.isfinite( fn ) & ( fn <= f + c1 * ( t * gd ).astype( f.dtype ) )
            return t, fn, gn, i + 1, ok

        t, fn, gn, nls, ok = jax.lax.while_loop( ls_cond, ls_body, ( t0, f, g, jnp.int32( 0 ), jnp.bool_( False ) ) )
        s = jnp.where( ok, t * d, 0 )
        xn = x + s
        fn = jnp.where( ok, fn, f )
        gn = jnp.where( ok, gn, g )
        y = gn - g
        sy = _vdot( s, y )
        upd = ok & ( sy > 1e-10 * _vdot( y, y ) )
        head = st[ "head" ]
        S = jnp.where( upd, st[ "S" ].at[ head ].set( s.reshape( -1 ) ), st[ "S" ] )
        Y = jnp.where( upd, st[ "Y" ].at[ head ].set( y.reshape( -1 ) ), st[ "Y" ] )
        rho = jnp.where( upd, st[ "rho" ].at[ head ].set( 1 / jnp.where( sy > 0, sy, 1 ) ), st[ "rho" ] )
        head = jnp.where( upd, ( head + 1 ) % m, head )
        # a failed line search before `min_iter`: drop the memory and go on along the gradient
        count = jnp.where( upd, jnp.minimum( count + 1, m ), jnp.where( ok, count, 0 ) )
        k = k + 1
        disp = jnp.max( jnp.abs( s ) )
        reason = _stop_reason( k, f, fn, disp, ok, max_iter, min_iter, ftol, disp_tol, f_scale )
        return dict( x = xn, f = fn, g = gn, k = k, S = S, Y = Y, rho = rho, head = head, count = count,
                     reason = reason, seg = st[ "seg" ] + 1, nev = st[ "nev" ] + nls,
                     losses = st[ "losses" ].at[ k ].set( fn ) )

    state = dict( state, seg = jnp.int32( 0 ) )
    return jax.lax.while_loop( cond, body, state )


def lbfgs( value_and_grad, x0, *, max_iter = 100, ftol = 1e-10, min_iter = 0, disp_tol = None, memory = 10,
           max_ls = 25, c1 = 1e-4, f_scale = 1.0, drift_tol = None, snapshot_every = None, on_snapshot = None ):
    """L-BFGS with a backtracking Armijo line search, entirely on the device ( see the module docstring ).

    The stop on the decrease is relative: `( f_prev - f ) / max( |f_prev|, |f|, f_scale ) <= ftol`. `f_scale = 1` makes it
    absolute for a loss below 1 ( where it would almost never stop ): `f_scale = 0` compares the decrease with the loss itself.

    `drift_tol`: stop when the points have settled, that is when `x` moved by at most `drift_tol` ( in the 2-norm, divided by
    the square root of its size: the root mean square of the displacements ) over the last `snapshot_every` steps ( 10 by
    default with a `drift_tol` ). The loss of a reconstruction is flat along the walls it draws, and settles long before the
    points do: this is the test for the points.

    `on_snapshot( k, x )` is called every `snapshot_every` steps ( and at the end ) with the step count and
    the device array of the positions: the loop pauses only there ( one host synchronization per
    segment ). `value_and_grad` must be hashable ( a function defined once, not a fresh lambda per call,
    or the loop recompiles )."""
    x0 = jnp.asarray( x0 )
    D = x0.size
    f0, g0 = value_and_grad( x0 )
    dt = x0.dtype
    # the loss keeps its own float ( float64 when the loss is computed so, even for float32 points ): the decreases
    # that the line search and `ftol` compare are far below float32's resolution of a loss near convergence
    state = dict( x = x0, f = f0, g = g0, k = jnp.int32( 0 ),
                  S = jnp.zeros( ( memory, D ), dt ), Y = jnp.zeros( ( memory, D ), dt ), rho = jnp.zeros( memory, dt ),
                  head = jnp.int32( 0 ), count = jnp.int32( 0 ), reason = jnp.int32( 0 ), nev = jnp.int32( 1 ),
                  losses = jnp.full( max_iter + 1, jnp.nan, f0.dtype ).at[ 0 ].set( f0 ) )
    if drift_tol is not None and not snapshot_every:
        snapshot_every = 10
    nb = max_iter if not snapshot_every else int( snapshot_every )
    x_last = x0
    while True:
        state = _lbfgs_segment( state, value_and_grad, memory, nb, max_iter, min_iter, float( ftol ),
                                None if disp_tol is None else float( disp_tol ), max_ls, c1, float( f_scale ) )
        if drift_tol is not None and int( state[ "reason" ] ) == 0:
            drift = float( jnp.sqrt( jnp.mean( ( state[ "x" ] - x_last ) ** 2 ) ) )
            x_last = state[ "x" ]
            if int( state[ "k" ] ) > min_iter and drift <= drift_tol:
                state = dict( state, reason = jnp.int32( 5 ) )
        done = int( state[ "reason" ] ) != 0
        if on_snapshot is not None and ( snapshot_every or done ):
            on_snapshot( int( state[ "k" ] ), state[ "x" ] )
        if done:
            break
    return state[ "x" ], _info( state )


def _info( state ):
    k = int( state[ "k" ] )
    return dict( losses = np.asarray( state[ "losses" ][ : k + 1 ] ), nb_steps = k, nb_evals = int( state[ "nev" ] ),
                 reason = REASONS[ int( state[ "reason" ] ) ] )


@partial( jax.jit, static_argnames = ( "vg", "nb", "max_iter", "lr", "min_iter", "ftol", "disp_tol" ) )
def _gd_segment( state, vg, nb, max_iter, lr, min_iter, ftol, disp_tol ):
    def cond( st ):
        return ( st[ "reason" ] == 0 ) & ( st[ "seg" ] < nb )

    def body( st ):
        s = -lr * st[ "g" ]
        x = st[ "x" ] + s
        f, g = vg( x )
        k = st[ "k" ] + 1
        reason = _stop_reason( k, st[ "f" ], f, jnp.max( jnp.abs( s ) ), jnp.isfinite( f ), max_iter, min_iter, ftol, disp_tol )
        return dict( x = x, f = f, g = g, k = k, reason = reason, seg = st[ "seg" ] + 1,
                     nev = st[ "nev" ] + 1, losses = st[ "losses" ].at[ k ].set( f ) )

    return jax.lax.while_loop( cond, body, dict( state, seg = jnp.int32( 0 ) ) )


def gradient_descent( value_and_grad, x0, *, lr = 0.1, max_iter = 100, ftol = -1.0, min_iter = 0, disp_tol = None,
                      snapshot_every = None, on_snapshot = None ):
    """Plain gradient steps x <- x - lr g, on the device. `ftol < 0` ( the default ): no early stop on the decrease."""
    x0 = jnp.asarray( x0 )
    f0, g0 = value_and_grad( x0 )
    state = dict( x = x0, f = f0, g = g0, k = jnp.int32( 0 ), reason = jnp.int32( 0 ), nev = jnp.int32( 1 ),
                  losses = jnp.full( max_iter + 1, jnp.nan, f0.dtype ).at[ 0 ].set( f0 ) )
    nb = max_iter if not snapshot_every else int( snapshot_every )
    while True:
        state = _gd_segment( state, value_and_grad, nb, max_iter, float( lr ), min_iter, float( ftol ),
                             None if disp_tol is None else float( disp_tol ) )
        done = int( state[ "reason" ] ) != 0
        if on_snapshot is not None and ( snapshot_every or done ):
            on_snapshot( int( state[ "k" ] ), state[ "x" ] )
        if done:
            break
    return state[ "x" ], _info( state )

