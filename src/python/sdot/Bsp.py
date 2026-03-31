from .driver import driver
from .Cell import Cell

from pathlib import Path
import dask.array as da  # type: ignore[import-untyped]
import numpy
import dask

class BspConfig:
    def __init__( self, max_points_per_cell, max_points_per_node ):
        self.max_points_per_cell = max_points_per_cell
        self.max_points_per_node = max_points_per_node
        self.use_cov = False


class IntermediateNode:
    """
        split_dir
        split_dot
        children
        cell
    """
    def __init__( self, indices, positions, nb_points: int, dim: int, cell: Cell, config: BspConfig, min_max_pts ):
        self.cell = cell

        # find the cut
        if config.use_cov:
            avg, cov = self._avg_and_cov( positions, nb_points )
            eig_system = numpy.linalg.eig( cov )
            num_eigval = numpy.argmax( eig_system.eigenvalues )
            eigval = eig_system.eigenvalues[ num_eigval ]
            if eigval == 0:
                raise RuntimeError( "TODO: all the points in the same place" )

            self.split_dir = eig_system.eigenvectors[ num_eigval ]
        else:
            ( ( loc_min_max_pts, avg ), ) = dask.compute( (
                da.max( positions, axis = 0 ) - da.min( positions, axis = 0 ),
                ( da.max( positions, axis = 0 ) + da.min( positions, axis = 0 ) ) / 2
            ) )
            d = loc_min_max_pts.argmax()
            eigval = loc_min_max_pts[ d ]
            self.split_dir = numpy.array( [ d == i for i in range( dim ) ] )

        proj = positions @ self.split_dir.T

        # 1D histogram
        avg_proj = numpy.dot( self.split_dir, avg )
        split_beg = avg_proj - 5 * eigval
        split_end = avg_proj + 5 * eigval
        hist, bins = da.compute( *da.histogram( proj, 256, range = ( split_beg, split_end ) ) )

        # split_dot
        cut_index = numpy.searchsorted( numpy.cumsum( hist ), hist.sum() / 2 )
        # if abs( new_nb_points_list[ 0 ] - new_nb_points_list[ 1 ] ) / hist.sum() >= 0.2:
        #     raise RuntimeError( "TODO: loop to find the cut for bad distributions" )
        self.split_dot = bins[ cut_index ]

        # children
        self.children = []
        def add_child( filter, sgn ):
            new_positions = positions[ filter ]

            cuts_and_maximums = []
            for cut_dir, _ in cell.cuts:
                cuts_and_maximums.append( ( cut_dir, da.max( da.dot( new_positions, cut_dir ) ) ) )
            cuts_and_maximums.append( ( sgn * self.split_dir, da.max( da.dot( new_positions, sgn * self.split_dir ) ) ) )

            new_nb_points, cuts_and_maximums = dask.compute( filter.sum(), cuts_and_maximums )

            new_cell = Cell.axis_aligned_hypercube( min_max_pts )
            for cut_dir, cut_dot in cuts_and_maximums:
                new_cell.cut( cut_dir, cut_dot )

            self.children.append( make_node( indices[ filter ], new_positions, new_nb_points, dim, new_cell, config, min_max_pts ) )

        filter_0 = ( positions @ self.split_dir ) > self.split_dot
        add_child( filter_0, - 1 )
        add_child( da.logical_not( filter_0 ), + 1 )

    def get_all_the_nodes( self, all_the_nodes, final_nodes ):
        res = len( all_the_nodes )
        all_the_nodes.append( self )

        self.child_indices = [ child.get_all_the_nodes( all_the_nodes, final_nodes ) for child in self.children ]

        return res

    def plot_rec( self, plotter, depth = 0 ):
        self.cell.plot( plotter, offset = [ 0, 0, depth / 3 ] )

        for child in self.children:
            child.plot_rec( plotter, depth + 1 )

    def _avg_and_cov( self, positions, nb_points ):
        avg = da.average( positions, axis = 0 )
        nrm = positions - avg[ None, : ]

        cov = da.dot( nrm.T, nrm ) # TODO: do not compute the symetric part
        cov /= nb_points

        return da.compute( avg, cov )

@dask.delayed
def lazy_shape( a ):
    return a.shape

class FinalNode:
    def __init__( self, indices, positions, cell: Cell, config: BspConfig, min_max_pts ):
        self.positions = positions
        self.indices = indices
        self.cell = cell

        self.child_indices = [ 0, 0 ]
        self.split_dir = []
        self.split_dot = 0

    def plot_rec( self, plotter, depth = 0 ):
        self.cell.plot( plotter, offset = [ 0, 0, depth / 3 ] )

        pts = self.positions.compute()
        if pts.shape[ 1 ] < 3:
            pts = driver.hstack( [ pts ] + [ driver.zeros( [ pts.shape[ 0 ], 1 ] ) ] * ( 3 - pts.shape[ 1 ] ) )
        plotter.add_points( driver.to_numpy( pts ) )

    def get_all_the_nodes( self, all_the_nodes, final_nodes ):
        res = len( all_the_nodes )
        all_the_nodes.append( self )
        final_nodes.append( res )

        return res


def make_node( indices, positions, nb_points: int, dim: int, cell: Cell, config: BspConfig, min_max_pts ):
    if nb_points > config.max_points_per_node:
        return IntermediateNode( indices, positions, nb_points, dim, cell, config, min_max_pts )
    return FinalNode( indices, positions, cell, config, min_max_pts )


class Bsp:
    """
    Prop: Bsp contient toujours une liste de LocalBsp

    On pourrait dire que la première partie en dask renvoie une liste avec pour chaque noeud
    - new_cut_dir, new_cut_off,
    - max_sps[ num_dir, num_cut ]
    - child_indices

    Pour chaque noeud final, on pourra récupérer
    - positions et indices (vide s'il y a des enfants)

    """

    def __init__( self, positions, max_points_per_cell = 30, max_points_per_node = 1e6 ):
        min_max_pts, ( nb_points, dim ) = dask.compute(
            da.stack( [ da.min( positions, axis = 0 ), da.max( positions, axis = 0 ) ] ),
            lazy_shape( positions )
        )

        config = BspConfig( max_points_per_cell, max_points_per_node )
        indices = da.arange( nb_points, dtype = driver.numpy_itype )
        cell = Cell.axis_aligned_hypercube( min_max_pts )

        root = make_node( indices, positions, nb_points, dim, cell, config, min_max_pts )

        # import pyvista
        # plotter = pyvista.Plotter( theme = pyvista.plotting.themes.DarkTheme() )
        # root.plot_rec( plotter )
        # plotter.show()

        all_the_nodes = []
        final_nodes = []
        root.get_all_the_nodes( all_the_nodes, final_nodes )

        for i in range( len( final_nodes ) ):
            all_the_nodes[ final_nodes[ i ] ].child_indices[ 1 ] = i

        node_summary = []
        for node in all_the_nodes:
            node_summary.append( ( node.child_indices, node.cell._instance ) )

        self.items = []
        class_binding = self._class_binding( dim )
        for node_index in final_nodes:
            self.items.append( dask.delayed( self._make_item )( class_binding, node_summary, node_index, all_the_nodes[ node_index ].positions, all_the_nodes[ node_index ].indices, config ) )
        self.items = dask.persist( *self.items )

    def _make_item( self, class_binding, node_summary, node_index: int, positions, indices, config ):
        positions = driver.t2( positions )
        indices = driver.t1( indices )

        # node_summary,
        return class_binding( node_summary, node_index, positions, indices, config.max_points_per_cell )

    def _class_binding( self, dim ):
        ct_dim = dim if dim <= 4 else -1
        dylib_name = f"bsp_{ driver.normalized_dtype }_{ ct_dim }_{ driver.normalized_device_type }"

        def src_func():
            return driver.cpp_src( { "SDOT_CT_DIM": ct_dim }, """
                #include <sdot/nanobind_wrappers.h>
                #include <sdot/geometry/Bsp.h>

                #include <nanobind/stl/vector.h>
                #include <nanobind/stl/string.h>
                #include <nanobind/stl/tuple.h>
                #include <sstream>
                #include <span>

                namespace nb = nanobind;
                using namespace sdot;

                using NA = nanobind::device::SDOT_NANOBIND_ARCH;
                using TF = SDOT_SCALAR_TYPE;

                using Arch = ArchFor<NA>::type;
                using NF = nb::ndarray<const TF,NA>;
                using NI = nb::ndarray<const PI,NA>;
                using MF = nb::ndarray<TF,NA>;

                static constexpr int ct_dim = SDOT_CT_DIM;
                using CellType = Cell<TF,2,Arch>;
                using BspType = Bsp<PI,TF,2,Arch>;
                using Pt = BspType::Pt;

                using NodeSummary = std::vector<std::tuple<
                    std::vector<PI>, // child_indices or [ 0, ext_index ]
                    CellType
                >>; //

                static Pt to_Pt( const auto &na ) {
                    return std::span<TF>( na.data(), na.size() );
                }

                NB_MODULE( SDOT_BINDING_NAME, m ) {
                    nb::class_<BspType>( m, "Bsp" )
                        .def( "__init__", []( BspType *self, const NodeSummary &node_summary, PI node_index, NF positions, NI indices, PI max_points_per_cell ) {
                            new ( self ) BspType(
                                node_summary,
                                node_index,
                                tensor_view_2( positions ),
                                tensor_view_1( indices ),
                                max_points_per_cell
                            );
                        } )
                        .def( "__repr__", []( const BspType &b ) -> std::string {
                            std::ostringstream ss;
                            ss << b;
                            return ss.str();
                        } )
                        .def( "nb_points", []( const BspType &b ) {
                            return b.nb_points;
                        } )
                        .def( "write_vtk", []( const BspType &b, std::string filename ) {
                            VtkOutput vo;
                            b.display_vtk( vo );
                            vo.save( filename );
                        } )
                    ;
                }
            """ )

        # get the binding
        geometry_dir = Path( __file__ ).parents[ 2 ] / "cpp" / "sdot" / "geometry"
        bnd = driver.import_bindings( dylib_name, src_func, [
            geometry_dir / "SimpleSquareMatrix_eigen.cpp",
            geometry_dir / "VtkOutput.cpp",
        ] )

        # make an instance
        return bnd.Bsp


    def write_vtk( self, filename: str ):
        self._class_binding.write_vtk( filename )

