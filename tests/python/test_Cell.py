import faulthandler; faulthandler.enable()
import sdot

def test_cell_2D_diff():
    # c = Cell.aligned_hypercube( [ 0, 0 ], [ 1, 1 ] )
    c = Cell( 2 )
    info( c.nb_vertices )
    # import torch

    # def loss( scale ):
    #     min_pos = sdot.driver.zeros( [ 2 ] )
    #     max_pos = scale * sdot.driver.ones( [ 2 ] )
    #     c = Cell.aligned_hypercube( min_pos, max_pos )
    #     return ( c.measure - 2 )**2


    # scale = sdot.driver.t0( 1 )
    # # scale.requires_grad = True
    # # torch.autograd.gradcheck( loss, scale )

    # sdot.driver.optimize_using_lbfgs( loss, scale )

    # ic( scale )

    # c.cut( [ 1, 0 ], 0.3 )
    # ic( c.vertex_positions )
    # ic( c.vertex_indices )
    # ic( c.edge_indices )
    # ic( c.cut_planes )
    # ic( c.measure )
    # ic( c.faces )

    # import torch

    # def loss( scale ):
    #     return sdot.cpp_binding( "test_alac", "sdot/cell/Cell.h" )( sdot.Return( sdot.driver.t0( 0 ) ), scale - 22 )

    # scale = sdot.driver.t0( 10 )
    # sdot.driver.optimize_using_lbfgs( loss, scale )

    # ic( scale )


# def test_cell_3D():
#     c = Cell.aligned_simplex( 3 )
#     # c.cut( [ 1, 0, 0 ], 0.5 )
#     info( c.vertex_indices )
#     info( c.edge_indices )
#     info( c.measure )

def test_cell_3D_basic():
    c = sdot.Cell.make_unbounded( 2 )

    # c.cut( [ 1, 0 ], 0.3 )
    info( c.vertex_positions )
    # info( c.vertex_indices )
    info( c.cut_planes )
    info( c.cut_ids )

    # info( c.measure )
    # info( c.measure )
    # ic( c.faces )

    # c.plot()

def test_cell_2D_basic():
    c = sdot.Cell.make_aligned_hypercube( [ 0, 0, 0 ], [ 2, 1, 1 ] )

    # c.cut( [ 1, 0 ], 0.3 )
    info( c.vertex_positions )
    # info( c.vertex_indices )
    info( c.cut_planes )
    info( c.cut_ids )

    info( c.measure )
    # info( c.measure )
    # ic( c.faces )

    # c.plot()

def test_cell_2D_grad():
    def f( s ):
        c = sdot.Cell.make_aligned_hypercube( [ 0, 0 ], [ s, s ] )
        return c.measure

    assert f( 2.0 ) == 4

    import jax
    assert jax.grad( f )( 2.0 ) == 4

def test_cell_3D_grad():
    import jax

    # V = s*s*1 = s^2, dV/ds = 2s
    def f( s ):
        c = sdot.Cell.make_aligned_hypercube( [ 0, 0, 0 ], [ s, s, 1 ] )
        infox( c )
        return c.measure

    #info( b )

    assert f( 2.0 ) == 4.0
    assert jax.grad( f )( 2.0 ) == 4.0

    # V = a*b*c, dV/da = b*c, dV/db = a*c, dV/dc = a*b
    def g( s ):
        c = sdot.Cell.make_aligned_hypercube( [ 0, 0, 0 ], [ s, 2 * s, 3 * s ] )
        return c.measure

    assert abs( g( 1.0 ) - 6.0 ) < 1e-6          # 1*2*3
    assert abs( jax.grad( g )( 1.0 ) - 18.0 ) < 1e-4  # d(6s^3)/ds = 18s^2 at s=1

def test_cell_2D_batch():
    if sdot.driver.available_gpus:
        sdot.driver.device = "gpu"
    # sdot.driver.device = "cpu"

    def f( s ):
        # c = sdot.BatchOfCell.aligned_hypercube( [ [ 0, 0, 0 ], [ 0, 0, 0 ] ], [ [ s, 1, 1 ], [ 2 * s, 1, 1 ] ] )
        c = sdot.BatchOfCells.make_aligned_hypercubes( [ [ 0, 0 ] ], [ [ s, 1 ] ] )
        return c.measures[ 0 ]

    assert f( 2.0 ) == 2

    import jax
    assert jax.grad( f )( 2.0 ) == 1

def test_cell_2D_cut():
    c = sdot.Cell.make_aligned_hypercube( [ 0, 0 ], [ 1, 1 ] )
    c.cut( [ 1, 0 ], 0.5 )
    info( c )

if __name__ == "__main__":
    # test_cell_2D_basic()
    # test_cell_2D_grad()
    # test_cell_3D_grad()
    # test_cell_2D_cut()
    # test_cell_2D_batch()
    # test_cell_2D_batch()
    # test_cell_2D_diff()
    # test_cell_3D()

    c = sdot.Cell.make_full_space( 2 )
    # b = sdot.BatchOfCells.make_full_spaces( 2, 4 )

    info( c )
