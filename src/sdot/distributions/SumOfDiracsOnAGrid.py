import numpy as np

import loom
from loom.tensor import Axis, CtShapeVar, RealTensor, ShapeVar
from loom.util import ComputedAttribute
from loom.tensor.host import to_host

from .Distribution import Distribution
from .SumOfDiracs import SumOfDiracs
from .Image import _pixel_frame


class SumOfDiracsOnAGrid( Distribution ):
    """
        Sum of weighted diracs at the CENTRES of the cells of a regular grid: no coordinates are stored, only the grid.

        The dirac of multi-index `k` ( C order: the first axis the slowest, that of `weights` ) is at
        `origin + frame^T ( k + 1/2 )`, the rows of the FIELD `frame` being the edges of ONE cell -- the convention of `Image`:
        `SumOfDiracsOnAGrid( shape )` puts a dirac at the centre of each pixel of an `Image` of that shape.

        What the constructor takes is said for the WHOLE grid, as for `Image`: by default the grid fits in the unit square
        ( cube... ) at `origin`, its longest side of length 1, the proportions kept. `frame` gives the edges of the whole grid
        ( rows, or their lengths for an axis-aligned grid ), `cell_frame` those of one cell ( the field ).

        It does all a `SumOfDiracs` does, except move its diracs. The 2D and 3D solver on the CPU takes a path of its own for a
        grid ( `PowerDiagram_Grid`: no positions are made ); the others read `positions`, computed when they are asked for.
    """

    # a dirac source for `SdotPlan1d` ( which normalizes it first: a `SumOfDiracs` in 1D )
    _is_dirac_source = True

    nb_diracs        : ShapeVar
    nb_dims          : CtShapeVar

    num_dirac        : Axis[ "nb_diracs" ]
    dim              : Axis[ "nb_dims" ]
    dir              : Axis[ "nb_dims" ]

    origin           : RealTensor[ "dim" ]
    frame            : RealTensor[ "dir", "dim" ]
    weights          : RealTensor[ "num_dirac" ]

    current_mass     : ComputedAttribute[ RealTensor, ( "weights" ) ]

    def __init__( self, shape, weights = None, frame = None, cell_frame = None, origin = None, target_mass = 1.0, **kwargs ):
        """`shape`: the number of cells along each axis. `weights`: `None` ( the same for all ), or one per cell, flat
        ( C order ) or of shape `shape`. `frame`: the edges of the WHOLE grid ( `[ d, d ]`, or `[ d ]` lengths ),
        `cell_frame` those of ONE cell. Neither: the longest side of the grid is 1. `origin`: its corner ( 0 by default )."""
        shape = tuple( int( s ) for s in np.atleast_1d( shape ) )
        d = len( shape )
        if frame is not None and cell_frame is not None:
            raise ValueError( "SumOfDiracsOnAGrid: `frame` ( the whole grid ) or `cell_frame` ( one cell ), not both" )
        if cell_frame is None:
            cell_frame = _pixel_frame( frame, shape )
        if origin is None:
            origin = np.zeros( d )
        if weights is not None:
            weights = getattr( weights, "raw", weights )
            ops = loom.ops( weights ) if not isinstance( weights, ( list, tuple ) ) else None
            weights = np.asarray( weights, dtype = float ).reshape( -1 ) if ops is None else ops.reshape( weights, ( -1, ) )
            if weights.shape[ 0 ] != int( np.prod( shape ) ):
                raise ValueError( f"SumOfDiracsOnAGrid: { weights.shape[ 0 ] } weights for a grid of { int( np.prod( shape ) ) } cells" )
        #: the number of cells along each axis
        self.shape = shape
        self.__base_init__( nb_diracs = int( np.prod( shape ) ), nb_dims = d, origin = origin, frame = cell_frame,
                            weights = weights, target_mass = target_mass, **kwargs )

    @property
    def grid( self ):
        """`( origin, cell_frame, shape )` on the host"""
        return Grid( np.asarray( to_host( self.origin.value ), dtype = float ).reshape( -1 ),
                     np.asarray( to_host( self.frame.value ), dtype = float ).reshape( len( self.shape ), -1 ), self.shape )

    @property
    def positions( self ):
        """the positions of the diracs, `[ nb_diracs, d ]`, COMPUTED ( in the framework of the frame ) -- for what asks for them"""
        idx = _centres( self.shape )
        frame, origin = self.frame.value, self.origin.value
        if isinstance( frame, np.ndarray ) and isinstance( origin, np.ndarray ):
            res = idx @ frame + origin[ None, : ]
        else:
            ops = loom.ops( frame )
            # ( on the device of the frame: Torch puts a new array on its default one )
            idx = frame.new_tensor( idx ) if hasattr( frame, "new_tensor" ) else ops.array( idx, ops.dtype_of( frame ) )
            res = ops.matmul( idx, frame ) + origin[ None, : ]
        return RealTensor[ self.num_dirac, self.dim ]( res )

    def as_sum_of_diracs( self ):
        """the same diracs, as a `SumOfDiracs` ( positions computed )"""
        res = SumOfDiracs( positions = self.positions.raw, weights = self.weights.value if self.weights.is_defined else None,
                           target_mass = self.target_mass.value if self.target_mass.is_defined else None )
        return res

    def normalized_version( self, nb_dims = None ):
        """the same grid, its weights scaled to `target_mass` -- still a grid: no positions are made. In 1D, the diracs on their
        positions: `SdotPlan1d` has no path of its own for a grid"""
        if len( self.shape ) == 1:
            return self.as_sum_of_diracs().normalized_version( nb_dims )
        if not self.target_mass.is_defined:
            return self
        mass = self.mass
        weights = self.target_mass / mass * self.weights if self.weights.is_defined else \
                  RealTensor[ *self.batch_axes, self.num_dirac ].full( self.target_mass / int( np.prod( self.shape ) ) )
        return SumOfDiracsOnAGrid( self.shape, weights = weights, cell_frame = self.frame.value, origin = self.origin.value,
                                   target_mass = self.target_mass, current_mass = self.target_mass, batch_axes = self.batch_axes )

    def add_to_viz( self, viz, color = "#d62728", **kwargs ):
        """The diracs, as points."""
        viz.add_points( np.asarray( to_host( self.positions.raw ) ), color = color )
        return viz

    def _update_current_mass( self ):
        if self.weights.is_defined:
            self.current_mass = self.weights.sum( axis = self.num_dirac )
        else:
            self.current_mass = int( np.prod( self.shape ) )


class Grid:
    """The geometry of a `SumOfDiracsOnAGrid`, on the host: the dirac `k` is at `origin + cell_frame^T ( k + 1/2 )`."""

    def __init__( self, origin, cell_frame, shape ):
        self.origin, self.cell_frame, self.shape = origin, cell_frame, tuple( shape )

    @property
    def shape_nd( self ):
        """`( nb_diracs, d )`: the shape the positions would have"""
        return ( int( np.prod( self.shape ) ), len( self.shape ) )

    @property
    def axis_aligned( self ):
        return bool( np.all( self.cell_frame == np.diag( np.diag( self.cell_frame ) ) ) )

    def __repr__( self ):
        return f"Grid( shape = { self.shape }, origin = { self.origin.tolist() }, cell_frame = { self.cell_frame.tolist() } )"


def _centres( shape ):
    """`k + 1/2` for each multi-index `k`, in C order, `[ prod( shape ), d ]`"""
    return np.stack( np.meshgrid( *[ np.arange( s ) + 0.5 for s in shape ], indexing = "ij" ), -1 ).reshape( -1, len( shape ) )
