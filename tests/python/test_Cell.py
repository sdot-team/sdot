from sdot.Cell import Cell

# import faulthandler
# faulthandler.enable()

# import numpy

def test_cell_2D():
    c = Cell.aligned_simplex( 2 )
    ic( c.measure )

def test_cell_3D():
    c = Cell.aligned_simplex( 3 )
    ic( c.measure )

if __name__ == "__main__":
    # test_cell_2D()
    test_cell_3D()
