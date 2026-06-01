from .aggregate import aggregate, Workspace, Tensor, Return
from typing import TYPE_CHECKING, Self, cast
from .drivers.driver import driver, FfiCode
import numpy

# constant
INFINITE = -2
BOUNDARY = -1


@aggregate
class Cell:
    """

    """

    # data
    vertex_positions  : Tensor( "nb_vertices[]", "dim", ct_variables = [ "dim" ] )
    vertex_indices    : Tensor( "nb_vertices[]", "dim", dtype = int ) # vertex index -> sorted cut indices
    edge_indices      : Tensor( "nb_edges[]", "dim + 1", dtype = int ) # edge index -> vertex indices (vertex on each side) + cut_indices
    cut_planes        : Tensor( "nb_cuts[]", "dim + 1" )
    cut_ids           : Tensor( "nb_cuts[]", dtype = int )

    is_fully_closed   : Tensor( dtype = int )


    if TYPE_CHECKING:
        def __default_init__( self, *args, **kwargs ): ...

        BatchVersion: Self
        batch_axes_dict: dict
        batch_axes: list

        max_of_nb_vertices: int
        max_of_nb_edges: int
        max_of_nb_cuts: int

        nb_vertices: numpy.array
        nb_edges: numpy.array
        nb_cuts: numpy.array

        dim: int

    # ---------------------------------- ctors ----------------------------------
    @classmethod
    def unbounded( cls, dim ):
        return cast( cls, driver.call( "p.cell.init_as_unbounded();", cell = Return( cls, **cls._return_parameters( dim ) ) ) )

    @classmethod
    def aligned_hypercube( cls, min_coords_or_dim = None, max_coords = None, dim = None, cut_id = BOUNDARY ):
        is_batch = hasattr( cls, 'BatchItemVersion' )

        if is_batch:
            # batched: min_coords is (batch_size, dim), max_coords is (batch_size, dim)
            min_coords = driver.array( min_coords_or_dim )
            max_coords = driver.array( max_coords ) if max_coords is not None else driver.ones( list( min_coords.shape ) )
            assert min_coords is not None
            assert max_coords is not None

            batch_size = min_coords.shape[ 0 ]
            dim        = min_coords.shape[ 1 ]

            diff  = max_coords - min_coords                        # (batch_size, dim)
            eye   = driver.array( numpy.eye( dim ) )[ None, :, : ] # (1, dim, dim)
            diag  = eye * diff[ :, None, : ]                       # (batch_size, dim, dim)
            frame = driver.concatenate( [ min_coords[ :, None, : ] ] + [ diag[ :, r:r+1, : ] for r in range( dim ) ], axis = 1 )

            cut_id = driver.array( cut_id, dtype = int )
            if cut_id.ndim == 0:
                cut_id = driver.repeat( cut_id, ( batch_size ) )

            return cls.hypercube( frame, cut_id = cut_id )
        else:
            if isinstance( min_coords_or_dim, int ):
                assert max_coords is None and dim is None
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

            assert max_coords is not None and min_coords is not None and dim is not None

            diff  = max_coords - min_coords
            diag  = cast( numpy.array, driver.array( numpy.eye( dim ) ) ) * diff
            frame = driver.stack( [ min_coords ] + [ diag[ r ] for r in range( dim ) ], axis = 0 )

            return cls.hypercube( frame, cut_id = cut_id )

    @classmethod
    def hypercube( cls, frame, cut_id : any = BOUNDARY ):
        batch_sizes = [ frame.shape[ 0 ] ] if hasattr( cls, 'BatchItemVersion' ) else []
        cut_id = driver.array( cut_id, dtype = int )
        frame = driver.array( frame )
        assert frame is not None
        dim = frame.shape[ -1 ]

        return cast( cls, driver.call(
            FfiCode(
                header = """
                struct InitAsHypercube {
                    HD void operator()( auto batch_indices, auto &&cell, auto &&frame, auto &&cut_id ) const {
                        cell( batch_indices ).init_as_hypercube( frame( batch_indices ), cut_id( batch_indices ) );
                    }
                };
                """,
                fwd = """
                    // run_parallel( cartesian_product_ranges( p.cell.batch_sizes() ), [] ( auto batch_indices, auto &&cell, auto &&frame, auto &&cut_id ) {
                    //     cell( batch_indices ).init_as_hypercube( frame( batch_indices ), cut_id( batch_indices ) );
                    // }, p.cell, p.frame, p.cut_id );
                    run_parallel( cartesian_product_ranges( p.cell.batch_sizes() ), InitAsHypercube{}, p.cell, p.frame, p.cut_id );
                """,
                bwd = """
                    //run_parallel( cartesian_product_ranges( p.cell.batch_sizes() ), [&] ( auto batch_indices ) mutable {
                    //    p.cell( batch_indices ).init_as_hypercube_bwd( p.frame( batch_indices ), p, batch_indices );
                    //} );
                """,
            ),
            cell = Return( cls, **cls._return_parameters( dim, batch_sizes ) ),
            cut_id = cut_id,
            frame = frame,
        ) )

    @classmethod
    def _return_parameters( cls, dim, batch_sizes = [] ):
        kw = dict(
            max_of_nb_vertices = 64,
            max_of_nb_edges = 64,
            max_of_nb_cuts = 64,
            dim = dim,
        )
        if batch_sizes:
            kw[ 'batch_size_Cell' ] = batch_sizes[ 0 ]
        return kw

    @property
    def measure( self ) -> any:
        max_nb_cuts = self.nb_cuts.max()

        max_of_nb_map_items = Cell._max_of_nb_map_items( self.dim, max_nb_cuts )
        max_nb_threads = min( driver.nb_threads(), self.batch_size_Cell )

        return driver.call(
            FfiCode(
                name = "measure",
                header = """
                    struct MeasureFunctor {
                        HD auto max_gpu_threads( auto &&map_items, auto &&.../* nb_map_items, outputs, max_nb_cuts, batch_of_cells */ ) const {
                            return PI( map_items.shape( Ct<int,0>() ) );
                        }

                        HD void per_thread( const auto &thread_info, const auto &/* batch_indices */, auto &&cont, auto &&map_items, auto &&nb_map_items, auto &&outputs, PI max_nb_cuts, auto &&batch_of_cells ) const {
                            auto item_map = recursive_map_of_unique_sorted_indices( Ct<int,decltype(batch_of_cells.dim)::value-1>(), map_items( thread_info.global_id() ), nb_map_items( thread_info.global_id() ), max_nb_cuts );
                            cont( outputs, batch_of_cells, item_map );
                        }

                        HD void operator()( const auto &batch_index, auto outputs, auto batch_of_cells, auto item_map ) const {
                            outputs( batch_index ) = batch_of_cells( batch_index ).measure( item_map );
                        }
                    };
                """,
                fwd = """
                    run_parallel( cartesian_product_ranges( p.batch_of_cells.batch_sizes() ), MeasureFunctor{}, p.map_items, p.nb_map_items, p.output, p.max_nb_cuts, p.batch_of_cells );
                """,
                bwd = """
                    run_parallel( cartesian_product_ranges( p.batch_of_cells.batch_sizes() ), MeasureFunctor{}, p.map_items, p.nb_map_items, p.output, p.max_nb_cuts, p.batch_of_cells );
                """,
            ),
            map_items = Workspace(
                Tensor( "max_nb_threads", "nb_map_items[ max_nb_threads ]", dtype = int ),
                max_of_nb_map_items = max_of_nb_map_items,
                max_nb_threads = max_nb_threads
            ),
            output = Return( Tensor( *self.batch_axes ), **self.batch_axes_dict ),
            max_nb_cuts = max_nb_cuts,
            batch_of_cells = self
        )

    @staticmethod
    def _max_of_nb_map_items( dim, nb_cuts = None ):
        if nb_cuts is None:
            nb_cuts = 256
        res = 0
        if dim >= 2:
            res += nb_cuts
        if dim >= 3:
            res += nb_cuts * nb_cuts
        if dim >= 4:
            for _ in range( 3, dim ):
                res += nb_cuts * nb_cuts
        return res


    def cut( self, cut_dir_or_plane, cut_off = None, cut_id = BOUNDARY ):
        cut_plane = driver.t1( cut_dir_or_plane )
        if cut_off is not None:
            cut_off = driver.t0( cut_off )
            cut_plane = driver.hstack( [ cut_plane, driver.expand_dims( cut_off, 0 ) ] )
        cpp_binding( "cut", "sdot/cell/cut.h" )( Output( self ), cut_plane, cut_id )

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
