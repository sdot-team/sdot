from sdot.object_with_tensors import object_with_tensors, TensorField
from .driver import driver, Return, Mutable, Dyn
from typing import TYPE_CHECKING
import numpy

# constant
INFINITE = -2
BOUNDARY = -1


@object_with_tensors
class Cell:
    """

    """

    vertex_positions = TensorField( Dyn( "nb_vertices" ), "dim" )
    vertex_indices   = TensorField( Dyn( "nb_vertices" ), "dim", dtype = int ) # vertex index -> sorted cut indices
    edge_indices     = TensorField( Dyn( "nb_edges" ), "dim + 1", dtype = int ) # edge index -> vertex indices (vertex on each side) + cut_indices
    cut_planes       = TensorField( Dyn( "nb_cuts" ), "dim + 1" )
    cut_ids          = TensorField( Dyn( "nb_cuts" ), dtype = int )

    is_fully_closed  = TensorField( dtype = int )

    if TYPE_CHECKING:
        def __default_init__( self, *args, **kwargs ): ...
        nb_vertices_capacity: int
        nb_edges_capacity: int
        nb_cuts_capacity: int
        dim: int


    def __init__( self, dim, nb_vertices_capacity = 50, nb_edges_capacity = 50, nb_cuts_capacity = 50 ):
        self.vertex_positions = driver.empty( [ nb_vertices_capacity, dim ] )
        self.cut_planes = driver.empty( [ nb_cuts_capacity, dim + 1 ] )
        self.cut_ids = driver.empty( [ nb_cuts_capacity ], dtype = driver.itype )

        self.is_fully_closed = 0
        self.nb_vertices = 0
        self.nb_edges = 0
        self.nb_cuts = 0

        if dim != 2:
            self.large_vertex_indices = driver.empty( [ nb_vertices_capacity, dim ], dtype = driver.int_type )
            self.large_edge_indices = driver.empty( [ nb_edges_capacity, dim + 1 ], dtype = driver.int_type )

        #     driver.call( "make_empty_cell", "sdot/cell/Cell.h", cell = Mutable( self ) )
        #     return

        # # assuming tensors
        # self.__default_init__( *args, **kwargs )


    @staticmethod
    def aligned_hypercube( min_coords_or_dim = None, max_coords = None, dim = None, bnd = BOUNDARY ):
        if isinstance( min_coords_or_dim, int ):
            assert max_coords is None
            assert dim is None
            min_coords = driver.zeros( [ min_coords_or_dim ] )
            dim = min_coords_or_dim
        elif min_coords_or_dim is None:
            assert dim is not None
            min_coords = driver.zeros( [ dim ] )
        else:
            min_coords = driver.array( min_coords_or_dim )
            dim = min_coords.shape[ 0 ]

        if max_coords is None:
            max_coords = driver.ones( [ dim ] )
        else:
            max_coords = driver.array( max_coords )

        diff = max_coords - min_coords
        diag = driver.array( numpy.eye( dim ) ) * diff
        frame = driver.stack( [ min_coords ] + [ diag[ r ] for r in range( dim ) ], axis = 0 )

        return Cell.hypercube( frame, bnd = bnd )

    @staticmethod
    def hypercube( frame, bnd = BOUNDARY ):
        frame = driver.array( frame )
        assert frame is not None
        return Cell( frame.shape[ 1 ], lambda cell: cpp_binding( "make_hypercube", "sdot/cell/Cell.h" )( Output( cell ), frame, bnd ) )

    @staticmethod
    def aligned_simplex( dim, bnd = BOUNDARY ):
        return Cell( dim, lambda cell: cpp_binding( "make_aligned_simplex", "sdot/cell/Cell.h" )( Output( cell ), bnd ) )

    # @property
    # def vertex_positions( self ):
    #     assert self.large_vertex_positions is not None
    #     return self.large_vertex_positions[ : self.nb_vertices, : ]

    # @property
    # def vertex_indices( self ):
    #     if self.dim == 2:
    #         nb = int( self.nb_vertices )
    #         k = numpy.arange( nb )
    #         kp = ( k + nb - 1 ) % nb
    #         return numpy.stack( [ numpy.minimum( k, kp ), numpy.maximum( k, kp ) ], axis = 1 )

    #     if self.large_vertex_indices is None:
    #         return None

    #     return self.large_vertex_indices[ : self.nb_vertices, : ]

    # @property
    # def edge_indices( self ):
    #     if self.dim == 2:
    #         nb = int( self.nb_vertices )
    #         k = numpy.arange( nb )
    #         return numpy.stack( [ k, ( k + 1 ) % nb, k ], axis = 1 )

    #     if self.large_edge_indices is None:
    #         return None

    #     return self.large_edge_indices[ : self.nb_edges, : ]

    # @property
    # def cut_planes( self ):
    #     if self.large_cut_planes is None:
    #         return None
    #     return self.large_cut_planes[ : self.nb_cuts, : ]

    # @property
    # def cut_ids( self ):
    #     return self.cut_ids[ : self.nb_cuts, : ]

    @property
    def faces( self ) -> list:
        res = cpp_binding( "faces", "sdot/cell/faces.h" )( self )
        assert res is not None
        return res

    @property
    def measure( self ) -> any:
        return cpp_binding( "measure", "sdot/cell/measure.h" )( Return( driver.empty( [] ) ), self )

    def cut( self, cut_dir_or_plane, cut_off = None, cut_id = BOUNDARY ):
        cut_plane = driver.t1( cut_dir_or_plane )
        if cut_off is not None:
            cut_off = driver.t0( cut_off )
            cut_plane = driver.hstack( [ cut_plane, driver.expand_dims( cut_off, 0 ) ] )
        cpp_binding( "cut", "sdot/cell/cut.h" )( Output( self ), cut_plane, cut_id )

    def cpp_class_name( self, driver ):
        return f"Cell<{ driver.normalized_dtype },{ self.dim },Cpu>"

    @classmethod
    def cpp_class_name_for( cls, **kwargs ):
        dim = kwargs.get( 'dim', -1 )
        return f"Cell<TF,{ dim },Cpu>"

    def plot( self, plotter = None, offset = None ):
        import pyvista

        # use our own plotter ?
        if plotter is None:
            plotter = pyvista.Plotter( theme = pyvista.plotting.themes.DarkTheme() )
            if self.dim == 2:
                plotter.view_xy()
            self.plot( plotter, offset )
            plotter.reset_camera()
            plotter.show()
            return

        #
        pts = self.vertex_positions # [ num_vertex, dim ]
        if pts is None:
            return

        #
        dim = pts.shape[ 1 ]
        if dim < 3:
            pts = driver.hstack( [ pts ] + [ driver.zeros( [ pts.shape[ 0 ], 1 ] ) ] * ( 3 - dim ) )
        elif dim > 3:
            pts = pts[ :, :3 ]

        #
        if offset is not None:
            offset = driver.array( offset )
            assert offset is not None
            pts += offset

        #
        faces = []
        for face in self.faces:
            faces.append( len( face ) )
            faces += face
        if pts.shape[ 0 ]:
            plotter.add_mesh( pyvista.PolyData( driver.to_numpy( pts ), faces = faces ), show_edges = True )


BatchOfCell = Cell.BatchVersion
