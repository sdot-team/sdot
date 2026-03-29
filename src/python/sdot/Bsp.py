from .driver import driver
from pathlib import Path
import dask.array as da  # type: ignore[import-untyped]
import numpy
import dask

class Bsp:
    """

    """

    def __init__( self, points, max_points_per_cell = 30, max_points_per_bsp = 1e6 ):
        min_max_pts = da.stack( [ da.min( points, axis = 0 ), da.max( points, axis = 0 ) ] ).compute()
        splits, all_the_paths = self._base_splits( points, max_points_per_bsp )
        self._class_binding = self._get_class_binding( min_max_pts.shape[ 1 ] )

        self.items = []
        for split in splits:
            self.items.append( dask.delayed( self._make_item )(
                all_the_paths,
                min_max_pts,
                split[ "indices" ],
                split[ "points" ],
                split[ "path" ],
                max_points_per_cell
            ) )

    def _make_item( self, all_the_paths, min_max_pts, indices, points, path, max_points_per_bsp ):
        all_the_paths = driver.t3( all_the_paths )
        min_max_pts = driver.t2( min_max_pts )
        indices = driver.t1( indices )
        points = driver.t2( points )
        path = driver.t2( path )

        return self._class_binding( all_the_paths, min_max_pts, indices, points, path, max_points_per_bsp )

    def _get_class_binding( self, dim ):
        ct_dim = dim if dim <= 4 else -1
        dylib_name = f"bsp_{ driver.normalized_dtype }_{ ct_dim }_{ driver.normalized_device_type }"

        def src_func():
            return driver.cpp_src( """
                #include "../../../../../cpp/geometry/Bsp.h"
                #include "../../nanobind_wrappers.h"
                #include <nanobind/stl/string.h>
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
                using BspType = Bsp<PI,TF,2,Arch>;
                using Pt = BspType::Pt;

                static Pt to_Pt( const auto &na ) {
                    return std::span<TF>( na.data(), na.size() );
                }

                NB_MODULE( SDOT_BINDING_NAME, m ) {
                    nb::class_<BspType>( m, "Bsp" )
                        .def( "__init__", []( BspType *self, NF all_the_paths, NF min_max_pts, NI indices, NF points, NF path, PI max_points_per_cell ) {
                            new ( self ) BspType(
                                tensor_view_3( all_the_paths ),
                                tensor_view_2( min_max_pts ),
                                tensor_view_1( indices ),
                                tensor_view_2( points ),
                                tensor_view_2( path ),
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
            """, { "SDOT_CT_DIM": ct_dim } )

        # get the binding
        geometry_dir = Path( __file__ ).parents[ 2 ] / "cpp" / "geometry"
        bnd = driver.import_bindings( dylib_name, src_func, [
            geometry_dir / "SimpleSquareMatrix_eigen.cpp",
            geometry_dir / "VtkOutput.cpp",
        ] )

        # make an instance
        return bnd.Bsp


    def write_vtk( self, filename: str ):
        self._class_binding.write_vtk( filename )

    def _avg_and_cov( self, points, nb_points ):
        avg = da.average( points, axis=0 )
        nrm = points - avg[ None, : ]
        cov = da.dot( nrm.T, nrm ) # TODO: remove the symetric part
        cov /= nb_points
        return da.compute( avg, cov )

    def _base_splits( self, points, max_points_per_bsp = 1e8, min_split = 1 ):
        """
        split points until it fits into the hardware
        """

        # a list of point to test and split if necessary
        too_large_splits = [ {
            "nb_points": points.shape[ 0 ],
            "indices": da.arange( 0, points.shape[ 0 ], chunks = points.chunks[ 0 ] ),
            "points": points,
            "path": []
        } ]

        # all_the_paths[ num_path, num_cut, split_dir + split_dot +  ]
        all_the_paths = []
        splits = []
        while len( too_large_splits ):
            split = too_large_splits.pop()

            split_nb_points = split[ "nb_points" ]
            split_indices = split[ "indices" ]
            split_points = split[ "points" ]
            split_path = split[ "path" ]
            if split_nb_points <= max_points_per_bsp:
                all_the_paths.append( split_path )
                splits.append( split )
                continue

            # split_dir
            avg, cov = self._avg_and_cov( split_points, split_nb_points )
            eig_system = numpy.linalg.eig( cov )
            num_eigval = numpy.argmax( eig_system.eigenvalues )
            eigval = eig_system.eigenvalues[ num_eigval ]
            if eigval == 0:
                raise RuntimeError( "TODO: all the points in the same place" )

            split_dir = eig_system.eigenvectors[ num_eigval ]
            proj = split_points @ split_dir.T

            # 1D histogram
            avg_proj = numpy.dot( split_dir, avg )
            split_beg = avg_proj - 5 * eigval
            split_end = avg_proj + 5 * eigval
            hist, bins = da.compute( *da.histogram( proj, 256, range = ( split_beg, split_end ) ) )

            # split_dot
            cut_index = numpy.searchsorted( numpy.cumsum( hist ), hist.sum() / 2 )
            # if abs( new_nb_points_list[ 0 ] - new_nb_points_list[ 1 ] ) / hist.sum() >= 0.2:
            #     raise RuntimeError( "TODO: loop to find the cut for bad distributions" )
            split_dot = bins[ cut_index ]

            # "left" part
            filter_0 = ( split_points @ split_dir ) > split_dot
            new_points_0 = split_points[ filter_0 ]
            max_sps_0 = da.max( da.dot( new_points_0, split_dir ) )

            too_large_splits.append( {
                "nb_points": hist[ : cut_index ].sum(),
                "indices": split_indices[ filter_0 ],
                "points": new_points_0,
                "path": split_path + [ [ *split_dir, split_dot, 0, max_sps_0 ] ]
            } )

            # "right" part
            filter_1 = da.logical_not( filter_0 )
            new_points_1 = split_points[ filter_1 ]
            max_sps_1 = da.max( da.dot( new_points_0, - split_dir ) )

            too_large_splits.append( {
                "nb_points": hist[ cut_index : ].sum(),
                "indices": split_indices[ filter_1 ],
                "points": new_points_1,
                "path": split_path + [ [ *split_dir, split_dot, 1, max_sps_1 ] ]
            } )

        return splits, numpy.array( all_the_paths )

