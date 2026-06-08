import faulthandler
import pytest
import sdot
faulthandler.enable()

# def test_cell_2D_diff():
#     # c = Cell.aligned_hypercube( [ 0, 0 ], [ 1, 1 ] )
#     c = Cell( 2 )
#     info( c.nb_vertices )
#     # import torch

#     # def loss( scale ):
#     #     min_pos = sdot.driver.zeros( [ 2 ] )
#     #     max_pos = scale * sdot.driver.ones( [ 2 ] )
#     #     c = Cell.aligned_hypercube( min_pos, max_pos )
#     #     return ( c.measure - 2 )**2


#     # scale = sdot.driver.t0( 1 )
#     # # scale.requires_grad = True
#     # # torch.autograd.gradcheck( loss, scale )

#     # sdot.driver.optimize_using_lbfgs( loss, scale )

#     # ic( scale )

#     # c.cut( [ 1, 0 ], 0.3 )
#     # ic( c.vertex_positions )
#     # ic( c.vertex_indices )
#     # ic( c.edge_indices )
#     # ic( c.cut_planes )
#     # ic( c.measure )
#     # ic( c.faces )

#     # import torch

#     # def loss( scale ):
#     #     return sdot.cpp_binding( "test_alac", "sdot/cell/Cell.h" )( sdot.Return( sdot.driver.t0( 0 ) ), scale - 22 )

#     # scale = sdot.driver.t0( 10 )
#     # sdot.driver.optimize_using_lbfgs( loss, scale )

#     # ic( scale )


# # def test_cell_3D():
# #     c = Cell.aligned_simplex( 3 )
# #     # c.cut( [ 1, 0, 0 ], 0.5 )
# #     info( c.vertex_indices )
# #     info( c.edge_indices )
# #     info( c.measure )

# def test_cell_3D_basic():
#     c = sdot.Cell.make_unbounded( 2 )

#     # c.cut( [ 1, 0 ], 0.3 )
#     info( c.vertex_positions )
#     # info( c.vertex_indices )
#     info( c.cut_planes )
#     info( c.cut_ids )

#     # info( c.measure )
#     # info( c.measure )
#     # ic( c.faces )

#     # c.plot()

# def test_cell_2D_basic():
#     c = sdot.Cell.make_aligned_hypercube( [ 0, 0, 0 ], [ 2, 1, 1 ] )

#     # c.cut( [ 1, 0 ], 0.3 )
#     info( c.vertex_positions )
#     # info( c.vertex_indices )
#     info( c.cut_planes )
#     info( c.cut_ids )

#     info( c.measure )
#     # info( c.measure )
#     # ic( c.faces )

#     # c.plot()

# def test_cell_2D_grad():
#     def f( s ):
#         c = sdot.Cell.make_aligned_hypercube( [ 0, 0 ], [ s, s ] )
#         return c.measure

#     assert f( 2.0 ) == 4

#     import jax
#     assert jax.grad( f )( 2.0 ) == 4

# def test_cell_3D_grad():
#     import jax

#     # V = s*s*1 = s^2, dV/ds = 2s
#     def f( s ):
#         c = sdot.Cell.make_aligned_hypercube( [ 0, 0, 0 ], [ s, s, 1 ] )
#         infox( c )
#         return c.measure

#     #info( b )

#     assert f( 2.0 ) == 4.0
#     assert jax.grad( f )( 2.0 ) == 4.0

#     # V = a*b*c, dV/da = b*c, dV/db = a*c, dV/dc = a*b
#     def g( s ):
#         c = sdot.Cell.make_aligned_hypercube( [ 0, 0, 0 ], [ s, 2 * s, 3 * s ] )
#         return c.measure

#     assert abs( g( 1.0 ) - 6.0 ) < 1e-6          # 1*2*3
#     assert abs( jax.grad( g )( 1.0 ) - 18.0 ) < 1e-4  # d(6s^3)/ds = 18s^2 at s=1

# def test_cell_2D_batch():
#     if sdot.driver.available_gpus:
#         sdot.driver.device = "gpu"
#     # sdot.driver.device = "cpu"

#     def f( s ):
#         # c = sdot.BatchOfCell.aligned_hypercube( [ [ 0, 0, 0 ], [ 0, 0, 0 ] ], [ [ s, 1, 1 ], [ 2 * s, 1, 1 ] ] )
#         c = sdot.BatchOfCells.make_aligned_hypercubes( [ [ 0, 0 ] ], [ [ s, 1 ] ] )
#         return c.measures[ 0 ]

#     assert f( 2.0 ) == 2

#     import jax
#     assert jax.grad( f )( 2.0 ) == 1

# def test_cell_2D_cut():
#     c = sdot.Cell.make_aligned_hypercube( [ 0, 0 ], [ 1, 1 ] )
#     c.cut( [ 1, 0 ], 0.5 )
#     info( c )

def test_cell_aligned_hypercube():
    # basic ctor
    c = sdot.Cell.make_aligned_hypercube( [ 0, 0 ], [ 1, 1 ] )
    assert c.nb_vertices == 4
    assert c.nb_cuts == 4

    # grad: scalar output
    def f_scalar( s ):
        c = sdot.Cell.make_aligned_hypercube( [ 0, 0 ], [ s, 2 * s ] )
        return c.measure

    g_scalar = sdot.driver.grad( f_scalar, 1.0 )
    # print( "grad(measure) wrt s at s=1:", g_scalar )
    # measure(s) = s * 2s = 2s^2, so d/ds = 4s -> 4 at s=1
    assert abs( float( g_scalar ) - 4.0 ) < 1e-5, f"expected 4.0, got {g_scalar}"

    # grad: tensor output — jacobian of vertex_positions wrt s, shape == vertex_positions.shape
    def f_tensor( s ):
        c = sdot.Cell.make_aligned_hypercube( [ 0, 0 ], [ s, 2 * s ] )
        return c.measure

    g_tensor = sdot.driver.grad( f_tensor, 1.0 )
    print( "grad(vertex_positions) wrt s at s=1:", g_tensor )

    # grad: multiple differentiable args — returns tuple
    def f_two( a, b ):
        c = sdot.Cell.make_aligned_hypercube( [ 0, 0 ], [ a, b ] )
        return c.measure

    g_a, g_b = sdot.driver.grad( f_two, 1.0, 2.0 )
    print( "grad(measure) wrt (a,b) at (1,2):", g_a, g_b )
    # measure = a*b, d/da = b = 2, d/db = a = 1
    assert abs( float( g_a ) - 2.0 ) < 1e-5, f"expected g_a=2.0, got {g_a}"
    assert abs( float( g_b ) - 1.0 ) < 1e-5, f"expected g_b=1.0, got {g_b}"

def test_cell_vmap():
    for dim in [ 3 ]:
        def make_and_measure( scale ):
            c = sdot.Cell.make_aligned_hypercube( [ 0 ] * dim, [ scale ] * dim )
            return c.measure

        scales = sdot.driver.array( [ 1.0, 2.0, 3.0 ] )
        measures = sdot.driver.vmap( make_and_measure )( scales )
        assert sum( abs( measures - scales ** dim ) ) < 1e-6


if __name__ == "__main__":
    # test_cell_aligned_hypercube()
    test_cell_vmap()
