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

template<class Vec>
static auto to_ndarray_1d( Vec &&vec ) {
    using T = std::decay_t<Vec>;
    auto *ptr = new T( std::move( vec ) );
    nb::capsule owner( ptr, []( void *p ) noexcept { delete static_cast<T *>( p ); } );
    return nb::ndarray<nb::numpy, typename T::value_type>( ptr->data(), { ptr->size() }, owner );
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
            .def( nb::init<sdot::PI,sdot::PI>() )
            .def( "__repr__", []( const BspType &b ) {
                std::ostringstream ss;
                ss << b;
                return ss.str();
            } )
            .def( "nb_points", []( const BspType &b ) {
                return b.nb_points;
            } )
            .def( "avg_data_for", []( BspType &b, const NA &positions ) -> nb::bytes {
                return serialize( b.avg_data_for( tensor_view_2( positions ) ) );
            } )
            .def( "cov_data_for", []( BspType &b, const NA &positions, const nb::bytes &avg ) -> nb::bytes {
                return serialize( b.cov_data_for(
                    tensor_view_2( positions ),
                    deserialize<std::vector<typename BspType::AvgData>>( avg )
                ) );
            } )
            .def( "avg_reduction", []( const nb::bytes &a, const nb::bytes &b ) -> nb::bytes {
                return serialize( BspType::avg_reduction(
                    deserialize<std::vector<typename BspType::AvgData>>( a ),
                    deserialize<std::vector<typename BspType::AvgData>>( b )
                ) );
            } )
            .def( "cov_reduction", []( const nb::bytes &a, const nb::bytes &b ) -> nb::bytes {
                return serialize( BspType::cov_reduction(
                    deserialize<std::vector<typename BspType::CovData>>( a ),
                    deserialize<std::vector<typename BspType::CovData>>( b )
                ) );
            } )
            .def( "split_dir", []( nb::bytes cov ) {
                auto dir = BspType::split_dir( deserialize<std::vector<typename BspType::CovData>>( cov ) );
                return to_ndarray_1d( dir );
            } )
        ;

    }
};

NB_MODULE( sdot_bsp_bindings, m ) {
    LocBspBindings<FP32>::add_mod( m );
    LocBspBindings<FP64>::add_mod( m );
}
