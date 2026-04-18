from sdot.object_with_tensors import object_with_tensors, TensorField
from .cpp_binding import cpp_binding, Output, Return
from typing import TYPE_CHECKING
from .driver import driver
# from .CtInt import CtInt
import numpy

# constant
INFINITE = -2
BOUNDARY = -1


@object_with_tensors
class Cell:
    """

    """

    large_vertex_positions = TensorField( "vertex_capacity", "dim" )
    large_vertex_indices = TensorField( "vertex_capacity", "dim", dtype = int ) # vertex index -> sorted cut indices
    large_edge_indices = TensorField( "edge_capacity", "dim + 1", dtype = int ) # edge index -> vertex indices (vertex on each side) + cut_indices
    large_cut_planes = TensorField( "cut_capacity", "dim + 1" )
    large_cut_ids = TensorField( "cut_capacity", dtype = int )

    is_fully_closed = TensorField( dtype = int )
    nb_vertices = TensorField( dtype = int )
    nb_edges = TensorField( dtype = int )
    nb_cuts = TensorField( dtype = int )

    if TYPE_CHECKING:
        vertex_capacity: int
        edge_capacity: int
        cut_capacity: int
        dim: int


    def __init__( self, dim, ctor = None ):
        vertex_capacity = 32
        edge_capacity = 32
        cut_capacity = 32

        self.large_vertex_positions = driver.empty( [ vertex_capacity, dim ] )
        self.large_cut_planes = driver.empty( [ cut_capacity, dim + 1 ] )
        self.large_cut_ids = driver.empty( [ cut_capacity ], dtype = driver.int_type )

        self.is_fully_closed = 0
        self.nb_vertices = 0
        self.nb_edges = 0
        self.nb_cuts = 0

        if dim != 2:
            self.large_vertex_indices = driver.empty( [ vertex_capacity, dim ], dtype = driver.int_type )
            self.large_edge_indices = driver.empty( [ edge_capacity, dim + 1 ], dtype = driver.int_type )

        if ctor is None:
            cpp_binding( "make_empty_cell", "sdot/cell/Cell.h" )( Output( self ) )
        else:
            ctor( self )

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

    @property
    def vertex_positions( self ):
        assert self.large_vertex_positions is not None
        return self.large_vertex_positions[ : self.nb_vertices, : ]

    @property
    def vertex_indices( self ):
        if self.dim == 2:
            nb = int( self.nb_vertices )
            k = numpy.arange( nb )
            kp = ( k + nb - 1 ) % nb
            return numpy.stack( [ numpy.minimum( k, kp ), numpy.maximum( k, kp ) ], axis = 1 )

        if self.large_vertex_indices is None:
            return None

        return self.large_vertex_indices[ : self.nb_vertices, : ]

    @property
    def edge_indices( self ):
        if self.dim == 2:
            nb = int( self.nb_vertices )
            k = numpy.arange( nb )
            return numpy.stack( [ k, ( k + 1 ) % nb, k ], axis = 1 )

        if self.large_edge_indices is None:
            return None

        return self.large_edge_indices[ : self.nb_edges, : ]

    @property
    def cut_planes( self ):
        if self.large_cut_planes is None:
            return None
        return self.large_cut_planes[ : self.nb_cuts, : ]

    @property
    def cut_ids( self ):
        return self.cut_ids[ : self.nb_cuts, : ]

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

    def cpp_class_name( self ):
        return f"Cell<{ driver.normalized_dtype },{ self.dim },Cpu>"

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


    # def __repr__( self ) -> str:
    #     return self._instance.__repr__()

    # def _checked_instance( self, dim = None ):
    #     if self._instance is not None:
    #         return self._instance
    #     if dim is None:
    #         raise RuntimeError( "dim cell" )
    #     self._instance = self._checked_bindings( dim ).Cell( dim )
    #     return self._instance

    # def _checked_bindings( self, dim ):
    #     if self._bindings is not None:
    #         return self._bindings
    #     ct_dim = dim if dim <= 4 else -1
    #     dylib_name = f"Cell_{ ct_dim }_{ driver.normalized_dtype }"
    #     self._bindings = driver.import_bindings( dylib_name, lambda: self._binding_code( ct_dim ) )
    #     return self._bindings


    # def _binding_code( self, ct_dim ):
    #     return driver.cpp_src( { "SDOT_CT_DIM": ct_dim }, """
    #         #include <sdot/nanobind_wrappers.h>
    #         #include <sdot/support/Tensor.h>
    #         #include <nanobind/stl/string.h>
    #         #include <nanobind/stl/vector.h>
    #         #include <nanobind/stl/tuple.h>
    #         #include <sdot/geometry/Cell.h>
    #         #include <sstream>
    #         #include <span>

    #         namespace nb = nanobind;
    #         using namespace sdot;

    #         using NbArch = nanobind::device::cpu;
    #         using Arch = ArchFor<NbArch>::type;
    #         using TF = SDOT_SCALAR_TYPE;

    #         using AF = nb::ndarray<const TF,NbArch>;
    #         using MF = nb::ndarray<TF,NbArch>;

    #         static constexpr int ct_dim = SDOT_CT_DIM;
    #         using CellType = Cell<TF,ct_dim,Arch>;
    #         using Pt = CellType::Pt;

    #         static Pt to_Pt( const auto &na ) {
    #             return std::span<const TF>( na.data(), na.size() );
    #         }

    #         NB_MODULE( SDOT_BINDING_NAME, m ) {
    #             m.def( "axis_aligned_hypercube", []( AF min_max ) {
    #                 auto t = tensor_view_2( min_max );
    #                 return CellType::axis_aligned_hypercube(
    #                     to_Pt( t.row( 0 ) ),
    #                     to_Pt( t.row( 1 ) )
    #                 );
    #             } );
    #             nb::class_<CellType>( m, "Cell" )
    #                 .def( "__init__", []( CellType *self, int dim ) {
    #                     new ( self ) CellType(
    #                         dim
    #                     );
    #                 } )
    #                 .def( "__repr__", []( const CellType &b ) -> std::string {
    #                     std::ostringstream ss;
    #                     ss << b;
    #                     return ss.str();
    #                 } )
    #                 .def( "cut", []( CellType &self, const AF dir, TF dot ) {
    #                     self.cut( to_Pt( dir ), dot, { self.dim } );
    #                 } )
    #                 .def( "dim", []( CellType &self ) {
    #                     return self.dim;
    #                 } )
    #                 .def( "vertices", []( const CellType &self ) {
    #                     Tensor<TF,2,Cpu> res( Shape(), { Values(), self.nb_vertices(), self.dim } );
    #                     self.for_each_vertex( [&]( Pt pos, PI index ) {
    #                         for( PI d = 0; d < self.dim; ++d )
    #                             res( index, d ) = pos[ d ];
    #                     } );
    #                     return to_ndarray_2d( res );
    #                 } )
    #                 .def( "faces", []( const CellType &self ) {
    #                     std::vector<std::vector<PI>> res; // [ num_face ][ num_vertex ]
    #                     self.for_each_face( [&]( std::vector<PI> &&c ) {
    #                         res.push_back( std::move( c ) );
    #                     } );
    #                     return res;
    #                 } )
    #                 .def( "cuts", []( const CellType &self ) {
    #                     std::vector<std::tuple<nanobind::ndarray<nanobind::numpy,TF>,TF>> res;
    #                     self.for_each_cut( [&]( Pt dir, TF dot, const auto & ) {
    #                         res.push_back( { to_ndarray_1d( dir ), dot } );
    #                     } );
    #                     return res;
    #                 } )
    #                 // .def( "write_vtk", []( const CellType &b, std::string filename ) {
    #                 //     VtkOutput vo;
    #                 //     b.display_vtk( vo );
    #                 //     vo.save( filename );
    #                 // } )
    #             ;
    #         }
    #     """ )

