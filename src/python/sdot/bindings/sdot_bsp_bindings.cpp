#include "../../../cpp/geometry/Bsp.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <sstream>
#include <span>

namespace nb = nanobind;
using namespace sdot;

static std::span<const std::byte> as_view( nb::bytes b ) {
    return { reinterpret_cast<const std::byte *>( b.data() ), b.size() };
}

template<class TF>
static auto tensor_view_2( const nb::ndarray<TF> &v ) {
    std::array<PI,2> extent{ v.shape( 0 ), v.shape( 1 ) };
    return TensorView<const TF,2>( v.data(), extent );
}

static nb::bytes serialize( const auto &obj ) {
    std::vector<std::byte> buf;
    zpp::bits::out out{ buf };
    out( obj ).or_throw();
    return nb::bytes( reinterpret_cast<const char *>( buf.data() ), buf.size() );
}

template<class T>
static T deserialize( const nb::bytes &bytes ) {
    T res{};
    zpp::bits::in in{ as_view( bytes ) };
    in( res ).or_throw();
    return res;
}

template<class TF,int ct_dim = -1>
struct LocBspBindings {
    using BspType = Bsp<PI,TF,2>;
    using NA = nb::ndarray<TF>;

    static void add_mod( auto &m ) {
        static std::string m0 = std::string( "Bsp_" ) + type_name( CtType<TF>() );
        static std::string r_avg = std::string( "reduction_of_avg_data_" ) + type_name( CtType<TF>() );
        static std::string r_cov = std::string( "reduction_of_cov_data_" ) + type_name( CtType<TF>() );

        nb::class_<BspType>( m, m0.c_str() )
            .def( nb::init<sdot::PI>() )
            .def( "__repr__", []( const BspType &b ) {
                std::ostringstream ss;
                ss << b;
                return ss.str();
            } )
            .def( "avg_data_for", []( BspType &b, NA positions ) -> nb::bytes {
                return serialize( b.avg_data_for( tensor_view_2( positions ) ) );
            } )
            .def( "cov_data_for", []( BspType &b, NA positions ) -> nb::bytes {
                return serialize( b.cov_data_for( tensor_view_2( positions ) ) );
            } )
            .def( "avg_reduction", []( nb::bytes a, nb::bytes b ) -> nb::bytes {
                auto va = deserialize<typename BspType::AvgData>( a );
                auto vb = deserialize<typename BspType::AvgData>( b );
                va += vb;
                return serialize( va );
            } )
            .def( "cov_reduction", []( nb::bytes a, nb::bytes b ) -> nb::bytes {
                auto va = deserialize<typename BspType::CovData>( a );
                auto vb = deserialize<typename BspType::CovData>( b );
                va += vb;
                return serialize( va );
            } )
        ;

    }
};

NB_MODULE( sdot_bsp_bindings, m ) {
    LocBspBindings<FP32>::add_mod( m );
    LocBspBindings<FP64>::add_mod( m );
}
