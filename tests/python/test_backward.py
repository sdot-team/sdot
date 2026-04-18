"""Gradient checks for measure_backward and make_hypercube_backward (2D)."""
import numpy as np
import pytest


EPS = 1e-6
ATOL = 1e-8
RTOL = 1e-5


# ---------------------------------------------------------------------------
# Pure-Python reference implementations
# ---------------------------------------------------------------------------

def _measure_2d( vp ):
    """Shoelace formula. vp: (n, 2) array, CCW order."""
    n = len( vp )
    s = sum( vp[i, 0] * vp[(i+1)%n, 1] - vp[(i+1)%n, 0] * vp[i, 1] for i in range( n ) )
    return s / 2


def _measure_grad_vp_2d( vp ):
    """Analytical gradient of shoelace w.r.t. vertex positions."""
    n = len( vp )
    g = np.zeros_like( vp )
    for k in range( n ):
        kp = (k + 1) % n
        km = (k + n - 1) % n
        g[k, 0] = ( vp[kp, 1] - vp[km, 1] ) / 2
        g[k, 1] = ( vp[km, 0] - vp[kp, 0] ) / 2
    return g


def _fd_grad( f, x, eps=EPS ):
    """Central finite-difference gradient of scalar f(x) w.r.t. flat x."""
    x = x.ravel().copy()
    g = np.zeros_like( x )
    for i in range( len( x ) ):
        x[i] += eps;  fp = f( x )
        x[i] -= 2*eps; fm = f( x )
        x[i] += eps
        g[i] = (fp - fm) / (2 * eps)
    return g


# ---------------------------------------------------------------------------
# measure_backward 2D
# ---------------------------------------------------------------------------

def test_measure_backward_2d_formula():
    """Analytical gradient of shoelace matches finite differences."""
    vp = np.array( [ [0.1, 0.2], [1.3, 0.05], [1.1, 0.85], [-0.1, 0.9] ], dtype=np.float64 )

    grad_anal = _measure_grad_vp_2d( vp )
    grad_num  = _fd_grad( lambda x: _measure_2d( x.reshape( -1, 2 ) ), vp ).reshape( -1, 2 )

    np.testing.assert_allclose( grad_anal, grad_num, rtol=RTOL, atol=ATOL )


def test_measure_backward_2d_unit_square():
    """Known values for the unit square (CCW order)."""
    vp = np.array( [ [0., 0.], [1., 0.], [1., 1.], [0., 1.] ], dtype=np.float64 )

    assert abs( _measure_2d( vp ) - 1.0 ) < 1e-14

    g = _measure_grad_vp_2d( vp )
    # vertex 0: d/dx = (y1-y3)/2 = (0-1)/2 = -0.5,  d/dy = (x3-x1)/2 = (0-1)/2 = -0.5
    np.testing.assert_allclose( g[0], [-0.5, -0.5], atol=1e-14 )
    # vertex 1: d/dx = (y2-y0)/2 = (1-0)/2 = +0.5,  d/dy = (x0-x2)/2 = (0-1)/2 = -0.5
    np.testing.assert_allclose( g[1], [0.5, -0.5], atol=1e-14 )
    # vertex 2: d/dx = (y3-y1)/2 = (1-0)/2 = +0.5,  d/dy = (x1-x3)/2 = (1-0)/2 = +0.5
    np.testing.assert_allclose( g[2], [0.5, 0.5], atol=1e-14 )
    # vertex 3: d/dx = (y0-y2)/2 = (0-1)/2 = -0.5,  d/dy = (x2-x0)/2 = (1-0)/2 = +0.5
    np.testing.assert_allclose( g[3], [-0.5, 0.5], atol=1e-14 )


# ---------------------------------------------------------------------------
# make_hypercube_backward 2D — vertex positions part
# ---------------------------------------------------------------------------

def _hypercube_vp_2d( frame ):
    """Vertex positions of 2D hypercube from frame (3×2). CCW order {0,1,3,2}."""
    bitmasks = [0, 1, 3, 2]
    vp = np.zeros( (4, 2) )
    for k, bm in enumerate( bitmasks ):
        vp[k] = frame[0] + ( (bm>>0)&1 ) * frame[1] + ( (bm>>1)&1 ) * frame[2]
    return vp


def _hypercube_vp_grad_frame_2d( frame, g_out_vp ):
    """Gradient of sum(g_out_vp * vp) w.r.t. frame."""
    bitmasks = [0, 1, 3, 2]
    g_frame = np.zeros_like( frame )
    for k, bm in enumerate( bitmasks ):
        g_frame[0] += g_out_vp[k]
        if (bm>>0) & 1: g_frame[1] += g_out_vp[k]
        if (bm>>1) & 1: g_frame[2] += g_out_vp[k]
    return g_frame


def test_make_hypercube_backward_2d_vp():
    """Gradient of (g_out_vp · vp) w.r.t. frame matches FD (vertex-position part)."""
    frame = np.array( [ [0.1, 0.2], [1.3, 0.05], [-0.1, 0.95] ], dtype=np.float64 )
    g_out_vp = np.random.default_rng( 42 ).standard_normal( (4, 2) )

    def loss( f ):
        return float( np.sum( g_out_vp * _hypercube_vp_2d( f.reshape(3,2) ) ) )

    grad_anal = _hypercube_vp_grad_frame_2d( frame, g_out_vp )
    grad_num  = _fd_grad( loss, frame ).reshape( 3, 2 )

    np.testing.assert_allclose( grad_anal, grad_num, rtol=RTOL, atol=ATOL )


# ---------------------------------------------------------------------------
# make_hypercube_backward 2D — cut planes part
# ---------------------------------------------------------------------------

def _hypercube_cp_2d( frame ):
    """Cut planes (4×3) of 2D hypercube from frame."""
    ax = frame[1];  bx = frame[2];  origin = frame[0]
    det = ax[0]*bx[1] - ax[1]*bx[0]
    # r0 = row 0 of F^{-1}:  FT * r0 = e0
    r0 = np.array( [ bx[1], -ax[1] ] ) / det   # solve [[ax0,bx0],[ax1,bx1]] r0 = [1,0]
    r1 = np.array( [-bx[0],  ax[0] ] ) / det   # solve FT * r1 = [0,1]
    d0 = r0 @ origin
    d1 = r1 @ origin
    cp = np.zeros( (4, 3) )
    cp[0] = [ -r1[0], -r1[1], -d1      ]  # lower y-face
    cp[1] = [ +r0[0], +r0[1], +d0 + 1  ]  # upper x-face
    cp[2] = [ +r1[0], +r1[1], +d1 + 1  ]  # upper y-face
    cp[3] = [ -r0[0], -r0[1], -d0      ]  # lower x-face
    return cp


def _hypercube_cp_grad_frame_2d( frame, g_out_cp ):
    """Analytical gradient of (g_out_cp · cp) w.r.t. frame."""
    r0_raw = _hypercube_cp_2d( frame )[1, :2]   # = +r0
    r1_raw = -_hypercube_cp_2d( frame )[0, :2]  # = +r1
    r0 = r0_raw;  r1 = r1_raw
    origin = frame[0]

    G_r0 = g_out_cp[1, :2] - g_out_cp[3, :2]
    G_r1 = g_out_cp[2, :2] - g_out_cp[0, :2]
    G_d0 = g_out_cp[1, 2]  - g_out_cp[3, 2]
    G_d1 = g_out_cp[2, 2]  - g_out_cp[0, 2]

    g_frame = np.zeros_like( frame )

    # origin
    g_frame[0] += G_d0 * r0 + G_d1 * r1

    # total gradient w.r.t. r0, r1 (including through d0, d1)
    tG_r0 = G_r0 + G_d0 * origin
    tG_r1 = G_r1 + G_d1 * origin

    # FT^{-T} has r0, r1 as rows  →  grad_FT via outer products
    p0 = r0 @ tG_r0;  q0 = r1 @ tG_r0
    p1 = r0 @ tG_r1;  q1 = r1 @ tG_r1

    # grad_frame(1+c, r) = grad_FT[r, c]
    g_frame[1, 0] += -p0*r0[0] - p1*r1[0]
    g_frame[1, 1] += -q0*r0[0] - q1*r1[0]
    g_frame[2, 0] += -p0*r0[1] - p1*r1[1]
    g_frame[2, 1] += -q0*r0[1] - q1*r1[1]

    return g_frame


def test_make_hypercube_backward_2d_cp():
    """Gradient of (g_out_cp · cp) w.r.t. frame matches FD (cut-plane part)."""
    frame = np.array( [ [0.1, 0.2], [1.3, 0.05], [-0.1, 0.95] ], dtype=np.float64 )
    g_out_cp = np.random.default_rng( 7 ).standard_normal( (4, 3) )

    def loss( f ):
        return float( np.sum( g_out_cp * _hypercube_cp_2d( f.reshape(3,2) ) ) )

    grad_anal = _hypercube_cp_grad_frame_2d( frame, g_out_cp )
    grad_num  = _fd_grad( loss, frame ).reshape( 3, 2 )

    np.testing.assert_allclose( grad_anal, grad_num, rtol=RTOL, atol=ATOL )


def test_make_hypercube_backward_2d_combined():
    """Gradient of (g_vp·vp + g_cp·cp) w.r.t. frame matches FD (both parts)."""
    frame = np.array( [ [0.1, 0.2], [1.3, 0.05], [-0.1, 0.95] ], dtype=np.float64 )
    rng = np.random.default_rng( 13 )
    g_vp = rng.standard_normal( (4, 2) )
    g_cp = rng.standard_normal( (4, 3) )

    def loss( f ):
        fr = f.reshape( 3, 2 )
        return float( np.sum( g_vp * _hypercube_vp_2d( fr ) ) + np.sum( g_cp * _hypercube_cp_2d( fr ) ) )

    grad_anal = _hypercube_vp_grad_frame_2d( frame, g_vp ) + _hypercube_cp_grad_frame_2d( frame, g_cp )
    grad_num  = _fd_grad( loss, frame ).reshape( 3, 2 )

    np.testing.assert_allclose( grad_anal, grad_num, rtol=RTOL, atol=ATOL )


if __name__ == "__main__":
    test_measure_backward_2d_formula()
    test_measure_backward_2d_unit_square()
    test_make_hypercube_backward_2d_vp()
    test_make_hypercube_backward_2d_cp()
    test_make_hypercube_backward_2d_combined()
    print( "all gradient checks passed" )
