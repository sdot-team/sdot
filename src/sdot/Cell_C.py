"""THE CANONICAL CELL: the simple polytope of `Cell_N`, in any dimension `>= 2`, that a power diagram cuts with CANONICAL PREDICATES
( `PowerDiagram( predicates = "canonical" )`, and always for `cell_complex()` ).

Each vertex is named by its generators -- the seed of its cell and its `d` cuts -- and every decision about it ( where it is, which
plane removes it ) is a function of those generators alone, the same in every cell that has it: the cells of a diagram then agree
exactly on what they share. A tie ( a plane through a vertex, as on a grid of seeds ) is broken by a symbolic perturbation of the
weights; the edges of zero length it makes are marked in `vertex_ties` ( a bit per neighbor ), which the cell complex contracts.
Kernel side: `cell/LocalN.h` with `CANON`, `cell/Canonical.h`. The cut is in doubles, whatever `kernel_dtype` says.
"""

from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar

from .Cell_N import Cell_N
from .CellScratch import words_of


class Cell_C( Cell_N ):
    vertex_positions : RealTensor[ "num_vertex", "dim_axis" ]
    vertex_cuts      : IntTensor [ "num_vertex", "dim_axis", dict( size = 32 ) ]
    vertex_nbrs      : IntTensor [ "num_vertex", "dim_axis", dict( size = 32 ) ]
    cut_ids          : IntTensor [ "num_cut", dict( size = 32 ) ]
    vertex_ties      : IntTensor [ "num_vertex", dict( size = 32 ) ]

    num_vertex       : Axis[ "nb_vertices" ]
    num_cut          : Axis[ "nb_cuts" ]
    dim_axis         : Axis[ "nb_dims" ]
    nb_vertices      : ShapeVar
    nb_cuts          : ShapeVar
    nb_dims          : CtShapeVar

    _GEOMETRY = Cell_N._GEOMETRY + ( "vertex_ties", )

    def __init__( self, nb_dims = None, init_as_unbounded = True, batch_axes = None, kernel_dtype = None ):
        super().__init__( nb_dims, init_as_unbounded, batch_axes, kernel_dtype = "FP64" )

    def scratch_words( self, cap, fp_size ):
        """`LocalN< double, d, true >::words_for( cap )`: those of `Cell_N` in doubles, and the generators"""
        d = self.dim
        return super().scratch_words( cap, 64 ) + ( 6 * d + 1 ) * words_of( 8, cap ) + 4 * words_of( 4, cap )
