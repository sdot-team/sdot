# from .Cell import Cell, INFINITE, BOUNDARY, _make_full_spaces, _make_hypercubes, _measures
from .aggregate import batch_variant_of
from .drivers.driver import driver
from typing import TYPE_CHECKING
from . import Cell
import numpy


@batch_variant_of( Cell.Cell )
class BatchOfCells:
    """
    Batch of Cell objects. Fields have an extra leading batch_size_Cell axis.
    Constructors take arrays with a leading batch dimension.
    """

    if TYPE_CHECKING:
        def __default_init__( self, *args, **kwargs ): ...

        @property
        def batch_axes_dict( self ) -> dict[ str, int ]: ...

        vertex_positions  : numpy.ndarray
        vertex_indices    : numpy.ndarray
        edge_indices      : numpy.ndarray
        cut_planes        : numpy.ndarray
        cut_ids           : numpy.ndarray
        is_fully_bounded   : numpy.ndarray

        batch_axes        : list

        max_of_nb_vertices: int
        max_of_nb_edges   : int
        max_of_nb_cuts    : int

        nb_vertices       : numpy.array
        nb_edges          : numpy.array
        nb_cuts           : numpy.array

        dim               : int

    @staticmethod
    def make_full_spaces( dim: int, batch_size: int ):
        return Cell._make_full_spaces( Cell, dim, { "batch_size_Cell": batch_size } )

    @classmethod
    def make_aligned_hypercubes( cls, min_coords, max_coords = None, cut_id = Cell.BOUNDARY ):
        """
        min_coords: (batch_size, dim)
        max_coords: (batch_size, dim), defaults to ones
        cut_id:     (batch_size,) or scalar
        """
        min_coords = driver.array( min_coords )
        max_coords = driver.array( max_coords ) if max_coords is not None else driver.ones( list( min_coords.shape ) )

        batch_size = min_coords.shape[ 0 ]
        dim        = min_coords.shape[ 1 ]

        diff  = max_coords - min_coords                            # (batch_size, dim)
        eye   = driver.array( numpy.eye( dim ) )[ None, :, : ]    # (1, dim, dim)
        diag  = eye * diff[ :, None, : ]                          # (batch_size, dim, dim)
        frame = driver.concatenate(
            [ min_coords[ :, None, : ] ] + [ diag[ :, r:r+1, : ] for r in range( dim ) ],
            axis = 1
        )                                                           # (batch_size, dim+1, dim)

        cut_id = driver.array( cut_id, dtype = int )
        if cut_id.ndim == 0:
            cut_id = driver.repeat( cut_id, batch_size )

        return cls.make_hypercubes( frame, cut_id = cut_id )

    @classmethod
    def make_hypercubes( cls, frame, cut_id = Cell.BOUNDARY ):
        """
        frame:  (batch_size, dim+1, dim)  — row 0 is origin, rows 1..dim are edge vectors
        cut_id: (batch_size,)
        """
        frame  = driver.array( frame )
        cut_id = driver.array( cut_id, dtype = int )
        return _make_hypercubes( cls, frame, cut_id, { 'batch_size_Cell': frame.shape[ 0 ] } )

    @property
    def measures( self ) -> any:
        return _measures( self, self.batch_axes_dict )
