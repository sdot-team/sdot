from .aggregate import aggregate, Workspace, Tensor, Return, Mutable, Conditional
from .drivers.driver import driver, FfiCodeCustom, FfiCodeParallel
from typing import TYPE_CHECKING, cast, overload
import numpy

# constant
INFINITE = -2
BOUNDARY = -1


@aggregate
class Cell:
    """
        2D -> vertex_indices and edge_indices are undefined

        A Cell stays a Cell after batching/vmap: batch axes are leading tensor dimensions
        (tracked in the instance `batch_axes`), never a distinct type. The `_make_*` / `_measures`
        helpers below take a `batch_axes` dict { axis_name: size }; {} means a single cell.
    """

    # base attributes
    is_fully_bounded  : Tensor( dtype = int )
    vertex_positions  : Tensor( "nb_vertices[]", "dim", ct_variables = [ "dim" ] )
    cut_planes        : Tensor( "nb_cuts[]", "dim + 1" )
    cut_ids           : Tensor( "nb_cuts[]", dtype = int )

    # nD attributes when n > 2
    # vertex_indices    : Conditional( lambda a: a.dim > 2, Tensor( "nb_vertices[]", "dim + 1", dtype = int ) ) # vertex index -> sorted cut indices
    # edge_indices      : Conditional( lambda a: a.dim > 2, Tensor( "nb_edges[]", "dim + 1", dtype = int ) ) # edge index -> vertex indices (vertex on each side) + cut_indices
    vertex_indices    : Tensor( "nb_vertices[]", "dim + 1", dtype = int ) # vertex index -> sorted cut indices
    edge_indices      : Tensor( "nb_edges[]", "dim + 1", dtype = int ) # edge index -> vertex indices (vertex on each side) + cut_indices

    # generated attributes
    if TYPE_CHECKING:
        def __default_init__( self, *args, **kwargs ): ...

        max_of_nb_vertices: int
        max_of_nb_edges: int
        max_of_nb_cuts: int

        nb_vertices: numpy.array
        nb_edges: numpy.array
        nb_cuts: numpy.array

        dim: int

    # --------------------------------------- ctor: make_full_space ------------------------------------
    @staticmethod
    def make_full_space( dim: int, batch_axes: dict = None ):
        return Cell._make_full_spaces( dim, batch_axes or {} )

    # ---------------------------------- ctor: make_aligned_hypercube ----------------------------------
    @staticmethod
    @overload
    def make_aligned_hypercube( dim: int, *, cut_id: int = BOUNDARY ) -> 'Cell':
        """Unit hypercube [0, 1]^dim."""
        ...

    @staticmethod
    @overload
    def make_aligned_hypercube( max_coords, *, cut_id: int = BOUNDARY ) -> 'Cell':
        """Axis-aligned hypercube from the origin [0,...,0] to max_coords."""
        ...

    @staticmethod
    @overload
    def make_aligned_hypercube( min_coords, max_coords, *, cut_id: int = BOUNDARY ) -> 'Cell':
        """Axis-aligned hypercube from min_coords to max_coords."""
        ...

    @staticmethod
    def make_aligned_hypercube( *args, **kwargs ) -> 'Cell':
        cut_id = kwargs.get( 'cut_id', BOUNDARY )

        if len( args ) == 1 and isinstance( args[ 0 ], int ):
            # make_aligned_hypercube(dim) — unit cube [0,1]^dim
            dim        = args[ 0 ]
            min_coords = driver.zeros( [ dim ] )
            max_coords = driver.ones( [ dim ] )
        elif len( args ) == 1:
            # make_aligned_hypercube(max_coords) — origin to max_coords
            max_coords = driver.array( args[ 0 ] )
            dim        = max_coords.shape[ 0 ]
            min_coords = driver.zeros( [ dim ] )
        elif len( args ) == 2:
            # make_aligned_hypercube(min_coords, max_coords)
            min_coords = driver.array( args[ 0 ] )
            max_coords = driver.array( args[ 1 ] )
            dim        = min_coords.shape[ 0 ]
        else:
            raise TypeError( f"make_aligned_hypercube: unexpected arguments {args!r}" )

        diff  = max_coords - min_coords
        diag  = cast( numpy.array, driver.array( numpy.eye( dim ) ) ) * diff
        frame = driver.stack( [ min_coords ] + [ diag[ r ] for r in range( dim ) ], axis = 0 )

        return Cell.make_hypercube( frame, cut_id = cut_id )

    # ---------------------------------- ctor: make_hypercube ----------------------------------
    @staticmethod
    def make_hypercube( frame, cut_id = BOUNDARY, batch_axes: dict = None ):
        """
        frame: (dim+1, dim)  — row 0 is origin, rows 1..dim are edge vectors
               with batch_axes, leading dimensions describe the batch (e.g. (N, dim+1, dim))
        """
        frame  = driver.array( frame )
        cut_id = driver.array( cut_id, dtype = int )
        return Cell._make_hypercubes( frame, cut_id, batch_axes or {} )

    # ---------------------------------- measure ----------------------------------
    @property
    def measure( self ) -> any:
        return Cell._measures( self, self.batch_axes_dict )

    # ---------------------------------- cut ----------------------------------
    def cut( self, cut_plane, cut_offset = None, cut_id = BOUNDARY ):
        cut_plane = driver.array( cut_plane )

        if cut_offset is not None:
            cut_offset = driver.t0( cut_offset )
            cut_plane = driver.hstack( [ cut_plane, driver.expand_dims( cut_offset, 0 ) ] )

        driver.call(
            FfiCodeCustom(
                name = "cut",
                header = """
                    struct CutFunctor {
                        template<class BI,class P> HD void operator()( const BI &batch_index, P &&p ) const {
                            p.cell( batch_index ).cut(  );
                        }
                    };
                """,
                fwd_code = """
                    run_parallel( cartesian_product_ranges( p.cell.batch_sizes() ), CutFunctor{}, p );
                """,
            ),
            cell = Mutable( self ),
            cut_planes     = cut_plane,
        )

    def plot( self, plotter = None, offset = None ):
        import pyvista

        if plotter is None:
            plotter = pyvista.Plotter( theme = pyvista.plotting.themes.DarkTheme() )
            if self.dim == 2:
                plotter.view_xy()
            self.plot( plotter, offset )
            plotter.reset_camera()
            plotter.show()
            return

        pts = self.vertex_positions # [ num_vertex, dim ]
        if pts is None:
            return

        dim = pts.shape[ 1 ]
        if dim < 3:
            pts = driver.hstack( [ pts ] + [ driver.zeros( [ pts.shape[ 0 ], 1 ] ) ] * ( 3 - dim ) )
        elif dim > 3:
            pts = pts[ :, :3 ]

        if offset is not None:
            offset = driver.array( offset )
            assert offset is not None
            pts += offset

        faces = []
        for face in self.faces:
            faces.append( len( face ) )
            faces += face
        if pts.shape[ 0 ]:
            plotter.add_mesh( pyvista.PolyData( driver.to_numpy( pts ), faces = faces ), show_edges = True )

    # --------------------------------------------------------------------------------------
    # Internal FFI helpers. `batch_axes` is a { axis_name: size } dict of the leading (batch)
    # dimensions; {} means a single cell. Batch axes are leading tensor dims (not struct
    # members), so their sizes are read from shapes at runtime — Cell stays Cell.
    # --------------------------------------------------------------------------------------

    @staticmethod
    def _batch_size_exprs( batch_axes: dict ) -> list:
        """C++ expressions for the batch sizes — read from a representative field's leading dims."""
        return [ f"p.cell.is_fully_bounded.shape( Ct<int,{ i }>() )" for i in range( len( batch_axes ) ) ]

    @staticmethod
    def _return_parameters_for( dim: int, batch_axes: dict ) -> dict:
        """Axis variables to allocate a new (possibly batched) Cell output."""
        kw = dict(
            max_of_nb_vertices = 64,
            max_of_nb_edges = 64,
            max_of_nb_cuts = 64,
            dim = dim,
        )
        if batch_axes:
            kw[ "batch_axes" ] = list( batch_axes.keys() )
        kw.update( batch_axes )
        return kw

    @staticmethod
    def _make_full_spaces( dim: int, batch_axes: dict ):
        return cast( Cell, driver.call(
            FfiCodeParallel(
                batch_axes = Cell._batch_size_exprs( batch_axes ),
                fwd_body = "p.cell( batch_index ).init_as_unbounded();",
                name = "make_full_spaces"
            ),
            cell = Return( Cell, **Cell._return_parameters_for( dim, batch_axes ) ) )
        )

    @staticmethod
    def _make_hypercubes( frames, cut_ids, batch_axes: dict ):
        dim = frames.shape[ -1 ]
        return cast( Cell, driver.call(
            FfiCodeParallel(
                batch_axes = Cell._batch_size_exprs( batch_axes ),
                fwd_body = "p.cell( batch_index ).init_as_hypercube( p.frames( batch_index ), p.cut_ids( batch_index ) );",
                bwd_body = "p.cell( batch_index ).init_as_hypercube_bwd( p.frames( batch_index ), p, batch_index );",
                name = "make_hypercubes"
            ),
            cell = Return( Cell, **Cell._return_parameters_for( dim, batch_axes ) ),
            cut_ids = cut_ids,
            frames = frames,
        ) )

    @staticmethod
    def _max_nb_map_items( dim: int, nb_cuts: int = None ):
        if nb_cuts is None:
            nb_cuts = 256
        res = 0
        res += nb_cuts * ( dim >= 2 )
        res += nb_cuts * nb_cuts * ( dim >= 3 )
        return res

    @staticmethod
    def _measures( cell_obj, batch_axes: dict ):
        # Workspace and kernel strides must be sized from a *static* allocation bound,
        # never from runtime data: under vmap/grad the actual `nb_cuts` is a traced value
        # (and would leak a tracer into shapes / the dylib cache key). `max_of_nb_cuts` is
        # the compile-time capacity of the cut buffers, known regardless of the driver.
        max_nb_cuts = cell_obj.max_of_nb_cuts
        dim = cell_obj.dim

        max_of_nb_map_items = Cell._max_nb_map_items( dim, max_nb_cuts )

        # `map_items` is per-thread scratch indexed by `thread_info.global_id()`, so it needs one
        # row per concurrent worker. The runtime worker count is min( nb_items, hardware_threads ),
        # and `nb_items` includes the vmap/batch axes that are only prepended *after* this point
        # (by the driver's batching rule) — so it is not visible here. We therefore size the
        # workspace by the hardware thread pool, which always bounds global_id() on CPU and GPU.
        max_nb_threads = driver.nb_threads( nb_local_bytes_per_thread = max_of_nb_map_items * 8 )

        per_thread_args = []
        per_thread = ""
        args = {}
        pre = "Void item_map; "
        if dim != 2:
            per_thread_args = [ "item_map" ]
            per_thread = """
                constexpr int dim = decltype( p.cell.dim )::value;
                auto item_map = recursive_map_of_unique_sorted_indices( Ct<int,dim-1>(), p.map_items( thread_info.global_id() ), p.nb_map_items( thread_info.global_id() ), p.max_nb_cuts );
                cont( p, item_map );
            """
            args = {
                "map_items" : Workspace(
                    Tensor( "max_nb_threads", "nb_map_items[ max_nb_threads ]", dtype = int ),
                    max_of_nb_map_items = max_of_nb_map_items,
                    max_nb_threads = max_nb_threads,
                )
            }
            pre = ""

        return driver.call(
            FfiCodeParallel( name = "measure",
                batch_axes = Cell._batch_size_exprs( batch_axes ),
                per_thread_args = per_thread_args,
                per_thread = per_thread,
                fwd_body = pre + "p.output( batch_index ) = p.cell( batch_index ).measure( item_map );",
                bwd_body = pre + "p.cell( batch_index ).measure_bwd( item_map, p, batch_index );",
            ),
            output = Return( Tensor( *batch_axes.keys() ), **batch_axes ),
            max_nb_cuts = max_nb_cuts,
            cell = cell_obj,
            **args,
        )
