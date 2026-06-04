from .aggregate import aggregate, Workspace, Tensor, Return, Mutable, Conditional
from .drivers.driver import driver, FfiCode
from typing import TYPE_CHECKING, cast
import numpy

# constant
INFINITE = -2
BOUNDARY = -1


@aggregate
class Cell:
    """
        2D -> vertex_indices and edge_indices are undefined
    """

    # base attributes
    is_fully_bounded  : Tensor( dtype = int )
    vertex_positions  : Tensor( "nb_vertices[]", "dim", ct_variables = [ "dim" ] )
    cut_planes        : Tensor( "nb_cuts[]", "dim + 1" )
    cut_ids           : Tensor( "nb_cuts[]", dtype = int )

    # nD attributes when n > 2
    vertex_indices    : Conditional( lambda a: a.dim > 2, Tensor( "nb_vertices[]", "dim + 1", dtype = int ) ) # vertex index -> sorted cut indices
    edge_indices      : Conditional( lambda a: a.dim > 2, Tensor( "nb_edges[]", "dim + 1", dtype = int ) ) # edge index -> vertex indices (vertex on each side) + cut_indices

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

    # ---------------------------------- ctors ----------------------------------
    @staticmethod
    def make_full_space( dim: int ):
        return _make_full_spaces( Cell, dim, {} )

    @staticmethod
    def make_aligned_hypercube( cls, min_coords_or_dim = None, max_coords = None, dim = None, cut_id = BOUNDARY ):
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

        return cls.make_hypercube( frame, cut_id = cut_id )

    @classmethod
    def make_hypercube( cls, frame, cut_id = BOUNDARY ):
        """
        frame: (dim+1, dim)  — row 0 is origin, rows 1..dim are edge vectors
        """
        frame  = driver.array( frame )
        cut_id = driver.array( cut_id, dtype = int )
        return _make_hypercubes( cls, frame, cut_id, {} )

    @property
    def measure( self ) -> any:
        return _measures( self, self.batch_axes_dict )

    def cut( self, cut_plane, cut_offset = None, cut_id = BOUNDARY ):
        cut_plane = driver.array( cut_plane )

        if cut_offset is not None:
            cut_offset = driver.t0( cut_offset )
            cut_plane = driver.hstack( [ cut_plane, driver.expand_dims( cut_offset, 0 ) ] )

        driver.call(
            FfiCode(
                name = "cut",
                header = """
                    struct CutFunctor {
                        HD void operator()( const auto &batch_index, auto &&p ) const {
                            p.batch_of_cells( batch_index ).cut(  );
                        }
                    };
                """,
                fwd = """
                    run_parallel( cartesian_product_ranges( p.batch_of_cells.batch_sizes() ), CutFunctor{}, p );
                """,
            ),
            batch_of_cells = Mutable( self ),
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


# ---------------------------------------------------------------------------
# Shared FFI helpers — called by Cell, BatchOfCells, and future nested variants.
# Each function takes a `batch_axes` dict { axis_name: size } describing the
# leading output dimensions; empty dict means single-cell (scalar output).
# ---------------------------------------------------------------------------

def _make_full_spaces( cls, dim: int, batch_axes: dict[ str, int ] ):
    """  """
    return cast( cls, driver.call( "p.cell.init_as_unbounded();", cell = Return( Cell, **_return_parameters_for( dim, batch_axes ) ) ) )

def _return_parameters_for( dim: int, batch_axes: dict[ str, int ] ) -> dict:
    """ axes variables to create a new Cell """
    kw = dict(
        max_of_nb_vertices = 64,
        max_of_nb_edges = 64,
        max_of_nb_cuts = 64,
        dim = dim,
    )
    kw.update( batch_axes )
    return kw

_HYPERCUBE_FFICODE = FfiCode(
    name = "init_as_hypercube",
    header = """
    struct InitAsHypercube {
        HD void operator()( auto batch_indices, auto &&p ) const {
            p.cell( batch_indices ).init_as_hypercube( p.frame( batch_indices ), p.cut_id( batch_indices ) );
        }
    };
    struct InitAsHypercubeBwd {
        HD void operator()( auto batch_indices, auto &&p ) const {
            p.cell( batch_indices ).init_as_hypercube_bwd( p.frame( batch_indices ), p, batch_indices );
        }
    };
    """,
    fwd = """
        run_parallel( cartesian_product_ranges( p.cell.batch_sizes() ), InitAsHypercube{}, p );
    """,
    bwd = """
        run_parallel( cartesian_product_ranges( p.cell.batch_sizes() ), InitAsHypercubeBwd{}, p );
    """,
)




def _make_hypercubes( cls, frame, cut_id, batch_axes: dict ):
    """Shared hypercube constructor used by all Cell variants."""
    dim = frame.shape[ -1 ]
    return cast( cls, driver.call(
        _HYPERCUBE_FFICODE,
        cell   = Return( cls, **_return_parameters_for( dim, batch_axes ) ),
        cut_id = cut_id,
        frame  = frame,
    ) )


def _max_nb_map_items( dim: int, nb_cuts: int = None ) -> int:
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


def _measures( cell_obj, batch_axes: dict ):
    """
    Shared measure implementation used by all Cell variants.

    batch_axes: ordered { axis_name: size } for the leading output dimensions,
                e.g. {} for a single Cell, {'batch_size_Cell': N} for BatchOfCells,
                {'batch_size_PD': M, 'batch_size_Cell': N} for a nested batch.
    """
    from math import prod

    dim         = cell_obj.dim
    max_nb_cuts = cell_obj.nb_cuts.max()

    max_of_nb_map_items = _max_nb_map_items( dim, max_nb_cuts )
    max_nb_threads      = min( driver.nb_threads(), max( 1, prod( batch_axes.values(), start = 1 ) ) )

    args = {}
    if dim != 2:
        args[ "map_items" ] = Workspace(
            Tensor( "max_nb_threads", "nb_map_items[ max_nb_threads ]", dtype = int ),
            max_of_nb_map_items = max_of_nb_map_items,
            max_nb_threads      = max_nb_threads,
        )

    return driver.call(
        FfiCode(
            name = "measure",
            header = """
                struct MeasureFunctor {
                    """ + """
                    HD auto max_gpu_threads( auto &&map_items, auto &&.../* nb_map_items, outputs, max_nb_cuts, batch_of_cells */ ) const {
                        return PI( map_items.shape( Ct<int,0>() ) );
                    }
                    """ * ( dim != 2 ) + """

                    HD void per_thread( const auto &thread_info, const auto &/* batch_indices */, auto &&cont, auto &&p, auto &&...args ) const {
                        constexpr int dim = decltype( p.batch_of_cells.dim )::value;
                        if constexpr( dim != 2 ) {
                            auto item_map = recursive_map_of_unique_sorted_indices( Ct<int,dim-1>(), p.map_items( thread_info.global_id() ), p.nb_map_items( thread_info.global_id() ), p.max_nb_cuts );
                            cont( item_map, p, args... );
                        } else {
                            cont( Void{}, p, args... );
                        }
                    }

                    HD void operator()( const auto &batch_index, auto &&item_map, auto &&p, Void ) const {
                        p.batch_of_cells( batch_index ).measure_bwd( item_map, p, batch_index );
                    }

                    HD void operator()( const auto &batch_index, auto &&item_map, auto &&p ) const {
                        p.output( batch_index ) = p.batch_of_cells( batch_index ).measure( item_map );
                    }
                };
            """,
            fwd = """
                run_parallel( cartesian_product_ranges( p.batch_of_cells.batch_sizes() ), MeasureFunctor{}, p );
            """,
            bwd = """
                run_parallel( cartesian_product_ranges( p.batch_of_cells.batch_sizes() ), MeasureFunctor{}, p, Void{} );
            """,
        ),
        output         = Return( Tensor( *batch_axes.keys() ), **batch_axes ),
        max_nb_cuts    = max_nb_cuts,
        batch_of_cells = cell_obj,
        **args,
    )
