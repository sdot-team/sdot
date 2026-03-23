#include "../../../cpp/geometry/Bsp.h"
#include <nanobind/stl/string.h>
#include "nanobind_wrappers.h"
#include <sstream>
#include <span>

namespace nb = nanobind;
using namespace sdot;

// static std::span<const std::byte> as_view( nb::bytes b ) {
//     return { reinterpret_cast<const std::byte *>( b.data() ), b.size() };
// }

// static nb::bytes serialize( const auto &obj ) {
//     std::vector<std::byte> buf;
//     zpp::bits::out out{ buf };
//     out( obj ).or_throw();
//     return nb::bytes( reinterpret_cast<const char *>( buf.data() ), buf.size() );
// }

// template<class T>
// static T deserialize( const nb::bytes &bytes ) {
//     T res{};
//     zpp::bits::in in{ as_view( bytes ) };
//     in( res ).or_throw();
//     return res;
// }

template<class TF,int ct_dim = -1>
struct LocBspBindings {
    using BspType = Bsp<PI,TF,2>;
    using NA = nb::ndarray<TF>;
    using Pt = BspType::Pt;

    static Pt to_Pt( const auto &na ) {
        return std::span<TF>( na.data(), na.size() );
    }

    static void add_mod( auto &m ) {
        static std::string m0 = std::string( "Bsp_" ) + type_name( CtType<TF>() );
        static std::string r_avg = std::string( "reduction_of_avg_data_" ) + type_name( CtType<TF>() );
        static std::string r_cov = std::string( "reduction_of_cov_data_" ) + type_name( CtType<TF>() );

        nb::class_<BspType>( m, m0.c_str() )
            .def( "__init__", []( BspType *self, nb::ndarray<const TF> all_the_paths, nb::ndarray<const PI> indices, nb::ndarray<const TF> points, nb::ndarray<const TF> path, PI max_points_per_cell ) {
                new ( self ) BspType(
                    tensor_view_3( all_the_paths ),
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
        ;

    }
};

NB_MODULE( sdot_bsp_bindings, m ) {
    LocBspBindings<FP32>::add_mod( m );
    LocBspBindings<FP64>::add_mod( m );
}
