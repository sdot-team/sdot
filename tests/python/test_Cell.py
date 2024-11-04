from sdot import Cell
import pytest
import numpy

def cut_and_check( c, dir, off ):
    print( "---------------------", dir, off, "---------------------" )
    c.cut_boundary( dir, off )

    print( "vertex_coords", c.vertex_coords( allow_lower_dim = True ) )
    print( "vertex_refs", c.vertex_refs( allow_lower_dim = True ) )
    print( "true_dimensionality", c.true_dimensionality )
    print( "nb_vertices", c.nb_vertices )
    print( "nb_cuts", c.nb_cuts )
    print( "bounded", c.bounded )
    print( "empty", c.empty )


def test_Cell_emptyness_bounded():
    c = Cell( ndim = 2 )
    assert c.nb_vertices == 0
    assert c.bounded == False
    assert c.empty == False

    c.cut_boundary( [ +1, 0 ], 1 )
    assert c.nb_vertices == 0
    assert c.bounded == False
    assert c.empty == False

    c.cut_boundary( [ 0, +1 ], 1 )
    assert c.nb_vertices == 1
    assert c.bounded == False
    assert c.empty == False

    c.cut_boundary( [ -1, 0 ], 1 )
    assert c.nb_vertices == 2
    assert c.bounded == False
    assert c.empty == False

    c.cut_boundary( [ 0, -1 ], 1 )
    assert c.nb_vertices == 4
    assert c.bounded == True
    assert c.empty == False

    c.cut_boundary( [ 1, 0 ], 0.5 )
    assert c.nb_vertices == 4
    assert c.bounded == True
    assert c.empty == False

    c.cut_boundary( [ 1, 0 ], -2 )
    assert c.nb_vertices == 0
    assert c.bounded == True
    assert c.empty == True

    c.cut_boundary( [ 1, 0 ], -3 )
    assert c.nb_vertices == 0
    assert c.bounded == True
    assert c.empty == True

def test_Cell_emptyness_unbounded():
    c = Cell( ndim = 3 )
    assert c.nb_vertices == 0
    assert c.bounded == False
    assert c.empty == False

    cut_and_check( c, [ 1.0, 1.0, 0 ], 1.0 )
    cut_and_check( c, [ 1.0, 1.0, 0 ], 1.5 )

# test_Cell_emptyness_bounded()
# test_Cell_emptyness_unbounded()

def test_base_3D():
    c = Cell( ndim = 3 )

    # make 2D cuts
    dirs = [
        [ -0.5, -0.7, 0 ],
        [ -0.5, +0.8, 0 ],
        [ +1.0, +0.0, 0 ],
    ]
    for dir in dirs:
        c.cut( dir, 1 )

    for dir, coords in zip( dirs, c.vertex_coords( True ) ):
        sp = c.base().T @ coords
        assert pytest.approx( numpy.dot( sp, dir ) ) == 1

    # add 3D
    c.cut( [ 0, 0, -1 ], 1 )
    print( c.vertex_coords() )
    print( c.bounded() )

    c.cut( [ 0, 0, +1 ], 1 )
    print( c.vertex_coords() )
    print( c.bounded() )

test_base_3D()

# cut( [ +1, 0, 0 ] )
# cut( [ 0, +1, 0 ] )
# cut( [ 0, 0, +1 ] )
# cut( [ -1, 0, 0 ] )
# cut( [ 0, -1, 0 ] )
# cut( [ 0, 0, -1 ] )

# print( cell )

# vo = module.VtkOutput()
# cell.display_vtk( vo )
# vo.save( "pd.vtk" )
