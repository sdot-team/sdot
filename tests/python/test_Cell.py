from sdot.Cell import Cell
from icecream import ic

# import faulthandler
# faulthandler.enable()

# import numpy

# def test_cell_2D():
#     c = Cell.aligned_simplex( 2 )
#     c.cut( [ 1, 1 ], 0.5 )
#     ic( c.vertex_positions )
#     ic( c.vertex_indices )
#     ic( c.edge_indices )
#     ic( c.cut_planes )
#     ic( c.measure )

def test_cell_2D():
    c = Cell.aligned_hypercube( 2 )
    c.cut( [ 1, 0 ], 0.3 )
    ic( c.vertex_positions )
    ic( c.vertex_indices )
    ic( c.edge_indices )
    ic( c.cut_planes )
    ic( c.measure )
    ic( c.faces )

    c.plot()

def test_cell_3D():
    c = Cell.aligned_simplex( 3 )
    # c.cut( [ 1, 0, 0 ], 0.5 )
    ic( c.vertex_indices )
    ic( c.edge_indices )
    ic( c.measure )

if __name__ == "__main__":
    test_cell_2D()
    # test_cell_3D()
