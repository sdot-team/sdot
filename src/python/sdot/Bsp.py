from .driver import driver
from pathlib import Path

class Bsp:
    """

    """

    def __init__( self, all_the_paths, min_max_pts, indices, points, path, max_points_per_cell = 30 ):
        all_the_paths = driver.t3( all_the_paths )
        min_max_pts = driver.t2( min_max_pts )
        indices = driver.t1( indices )
        points = driver.t2( points )
        path = driver.t2( path )

        dim = min_max_pts.shape[ 1 ]
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
        self._instance = bnd.Bsp(
            all_the_paths,
            min_max_pts,
            indices,
            points,
            path,
            max_points_per_cell
        )

    def write_vtk( self, filename: str ):
        self._instance.write_vtk( filename )
