# import numpy as np
# import pytest
import sdot

# def test_spline_grid_1d_forward():
#     n = 5
#     v = np.linspace( 0, 1, n ) ** 3
#     knots = [ np.linspace( 0, 1, n ) ]

#     sg = sd.SplineGrid( v, knots=knots, continuity=0 )
#     pg = sg.normalized_version()

#     # check continuity at intermediate knots
#     # In first interval: c0 + c1*x + c2*x^2 + c3*x^3
#     # At x=0, should be 0.
#     # At x=0.25, should be 0.25^3 = 0.015625

#     # Wait, the first cell is [0, 0.25]
#     coeffs = pg.values[ 0 ] # c0, c1, c2, c3
#     val_at_0 = coeffs[ 0 ]
#     val_at_025 = coeffs[ 0 ] + coeffs[ 1 ] * 0.25 + coeffs[ 2 ] * 0.25**2 + coeffs[ 3 ] * 0.25**3

#     assert np.allclose( val_at_0, 0.0 )
#     assert np.allclose( val_at_025, 0.25**3 )


# def test_spline_grid_1d_forward():
#     n = 5
#     v = np.linspace( 0, 1, n ) ** 3
#     knots = [ np.linspace( 0, 1, n ) ]

#     sg = sd.SplineGrid( v, knots=knots, continuity=1 )
#     pg = sg.normalized_version()

#     assert pg.values.shape == ( n - 1, 4 )

#     # check continuity at intermediate knots
#     # In first interval: c0 + c1*x + c2*x^2 + c3*x^3
#     # At x=0, should be 0.
#     # At x=0.25, should be 0.25^3 = 0.015625

#     # Wait, the first cell is [0, 0.25]
#     coeffs = pg.values[ 0 ] # c0, c1, c2, c3
#     val_at_0 = coeffs[ 0 ]
#     val_at_025 = coeffs[ 0 ] + coeffs[ 1 ] * 0.25 + coeffs[ 2 ] * 0.25**2 + coeffs[ 3 ] * 0.25**3

#     assert np.allclose( val_at_0, 0.0 )
#     assert np.allclose( val_at_025, 0.25**3 )

# def test_spline_grid_2d_forward():
#     nx, ny = 4, 5
#     x = np.linspace( 0, 1, nx )
#     y = np.linspace( 0, 1, ny )
#     X, Y = np.meshgrid( x, y, indexing='ij' )
#     v = X**3 + Y**3

#     knots = [ x, y ]
#     sg = sd.SplineGrid( v, knots=knots, continuity=1 )
#     pg = sg.normalized_version()

#     assert pg.values.shape == ( nx - 1, ny - 1, 16 )

#     # check a point
#     # cell (1, 1) is [0.333, 0.666] x [0.25, 0.5]
#     ix, iy = 1, 1
#     coeffs = pg.values[ ix, iy ] # lex order: MSB is x
#     # order 3 means p0*4 + p1
#     # p(x,y) = sum_{p,q} c_{pq} x^p y^q

#     val = 0
#     test_x, test_y = 0.5, 0.3
#     for p in range( 4 ):
#         for q in range( 4 ):
#             val += coeffs[ p * 4 + q ] * ( test_x**p ) * ( test_y**q )

#     assert np.allclose( val, test_x**3 + test_y**3 )

# def test_spline_grid_1d_backward():
#     n = 5
#     v = np.linspace( 0, 1, n )**3
#     knots = [ np.linspace( 0, 1, n ) ]

#     # use sdot.driver
#     v = sd.driver.t1( v )
#     knots = [ sd.driver.t1( k ) for k in knots ]

#     sg = sd.SplineGrid( v, knots=knots, continuity=1 )
#     pg = sg.normalized_version()

#     # If we are using JAX or PyTorch, we can use their grad functions.
#     # But let's just trigger the backward call.
#     if sd.driver.normalized_framework == "jax":
#         import jax
#         def loss_fn( v ):
#             sg = sd.SplineGrid( v, knots=knots, continuity=1 )
#             pg = sg.normalized_version()
#             return ( pg.values**2 ).sum()
#         grad = jax.grad( loss_fn )( v )
#         assert grad is not None
#     elif sd.driver.normalized_framework == "torch":
#         import torch
#         v = v.clone().detach().requires_grad_(True)
#         sg = sd.SplineGrid( v, knots=knots, continuity=1 )
#         pg = sg.normalized_version()
#         loss = ( pg.values**2 ).sum()
#         loss.backward()
#         assert v.grad is not None
def test_spline_grid_c1_1d():
    """Test 1D SplineGrid with continuity=1 (natural cubic spline).

    Values [1, 2, 1] at knots [0, 1, 2].  The natural cubic spline has:
      Interval [0,1]:  1 + (3/2)x - (1/2)x^3
      Interval [1,2]:  (9/2)x - 3x^2 + (1/2)x^3
    Verified: values at control points, C1 continuity at x=1, midpoint values.
    """
    s = sdot.SplineGrid( [ 1, 2, 1 ], continuity = 1 )
    g = s.normalized_version()

    tol = 1e-10

    # Values at control points must match original SplineGrid values
    assert abs( g[ 0 ] - 1 ) < tol, f"g[0]={g[0]}, expected 1"
    assert abs( g[ 1 ] - 2 ) < tol, f"g[1]={g[1]}, expected 2"
    assert abs( g[ 2 ] - 1 ) < tol, f"g[2]={g[2]}, expected 1"

    # C1 continuity: one-sided finite-difference derivatives at x=1 must agree
    eps = 1e-6
    deriv_left  = ( g[ 1 ] - g[ 1 - eps ] ) / eps
    deriv_right = ( g[ 1 + eps ] - g[ 1 ] ) / eps
    assert abs( deriv_left - deriv_right ) < 1e-4, \
        f"C1 broken at x=1: left deriv={deriv_left:.6f}, right deriv={deriv_right:.6f}"

    # Midpoint values from known closed-form polynomials (natural spline on [1,2,1])
    x_mid0 = 0.5
    expected_mid0 = 1 + 1.5 * x_mid0 - 0.5 * x_mid0 ** 3
    assert abs( g[ x_mid0 ] - expected_mid0 ) < tol, \
        f"g[0.5]={g[x_mid0]}, expected {expected_mid0}"

    x_mid1 = 1.5
    expected_mid1 = 4.5 * x_mid1 - 3 * x_mid1 ** 2 + 0.5 * x_mid1 ** 3
    assert abs( g[ x_mid1 ] - expected_mid1 ) < tol, \
        f"g[1.5]={g[x_mid1]}, expected {expected_mid1}"

    print( "test_spline_grid_c1_1d passed" )
