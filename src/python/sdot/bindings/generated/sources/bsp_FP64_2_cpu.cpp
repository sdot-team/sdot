#line 44 "/Users/hugo.leclerc/sdot_with_interfaces/src/python/sdot/Bsp.py"
#include "../../../../../cpp/geometry/Bsp.h"
#include "../../nanobind_wrappers.h"
#include <nanobind/stl/string.h>
#include <sstream>
#include <span>

namespace nb = nanobind;
using namespace sdot;

using NA = nanobind::device::cpu;
using TF = FP64;

using Arch = ArchFor<NA>::type;
using NF = nb::ndarray<const TF,NA>;
using NI = nb::ndarray<const PI,NA>;
using MF = nb::ndarray<TF,NA>;

static constexpr int ct_dim = 2;
using BspType = Bsp<PI,TF,2,Arch>;
using Pt = BspType::Pt;

static Pt to_Pt( const auto &na ) {
    return std::span<TF>( na.data(), na.size() );
}

NB_MODULE( bsp_FP64_2_cpu, m ) {
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
