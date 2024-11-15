#include <tl/support/type_info/type_name.h>
#include <tl/support/string/to_string.h>
#include <sdot/poom/PoomVec.h>
#include "config.h"
 
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifndef SDOT_CONFIG_size_0
#define SDOT_CONFIG_size_0 2
#endif

using namespace sdot;

static constexpr int nd = SDOT_CONFIG_size_0;
using TF = SDOT_CONFIG_scalar_type;
using IT = Vec<TF,nd>;

using Array_TF = pybind11::array_t<TF, pybind11::array::c_style>;

PYBIND11_MODULE( SDOT_CONFIG_module_name, m ) {
    pybind11::class_<PoomVec<IT>>( m, PD_STR( PoomVec ) )
        .def( "__repr__", []( const PoomVec<IT> &vec ) { return to_string( vec ); } )
        
        .def( "as_ndarray", []( PoomVec<IT> &vec ) {
            Vec<PI,2> shape{ vec.size(), PI( nd ) };
            Array_TF res( shape );
            vec.get_values_by_chuncks( [&]( CstSpanView<IT> sv ) {
                for( PI i = sv.beg_index(); i < sv.end_index(); ++i )
                    for( PI d = 0; d < nd; ++d )
                        res.mutable_at( i, d ) = sv[ i ][ d ];
            } );
            return res;
        } )

        .def( "dtype", []( PoomVec<IT> &vec ) { return type_name<TF>(); } )
        .def( "shape", []( PoomVec<IT> &vec ) { return std::vector{ vec.size(), PI( nd ) }; } )

        .def( "self_add", []( PoomVec<IT> &self, PoomVec<IT> &that ) { self.operator+=( that ); } )
        .def( "self_sub", []( PoomVec<IT> &self, PoomVec<IT> &that ) { self.operator-=( that ); } )
        // .def( "self_div", []( PoomVec<IT> &self, IT that ) { self.operator/=( that ); } )
        ;

    // "named" ctor
    m.def( "make_PoomVec_from_ndarray", []( const Array_TF &inp ) -> PoomVec<IT> {
        return CstSpan<IT>( reinterpret_cast<const IT *>( inp.data() ), inp.shape( 0 ) );
    } );
}
