#include <tl/support/type_info/type_name.h>
#include <tl/support/string/to_string.h>
#include <sdot/poom/PoomVec.h>
#include "config.h"
 
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace sdot;

using TF = SDOT_CONFIG_scalar_type;

using Array_TF = pybind11::array_t<TF, pybind11::array::c_style>;

PYBIND11_MODULE( SDOT_CONFIG_module_name, m ) { // py::module_local()
    pybind11::class_<PoomVec<TF>>( m, PD_STR( PoomVec ) )
        // .def( pybind11::init<>() )
        .def( "__repr__", []( const PoomVec<TF> &vec ) { return to_string( vec ); } )
        
        .def( "as_ndarray", []( PoomVec<TF> &vec ) {
            Vec<PI,1> shape{ vec.size() };
            Array_TF res( shape );
            vec.get_values_by_chuncks( [&]( CstSpanView<TF> sv ) {
                for( PI i = sv.beg_index(); i < sv.end_index(); ++i )
                    res.mutable_at( i ) = sv[ i ];
            } );
            return res;
        } )

        .def( "dtype", []( PoomVec<TF> &vec ) { return type_name<TF>(); } )
        .def( "shape", []( PoomVec<TF> &vec ) { return std::vector{ vec.size() }; } )

        .def( "self_add", []( PoomVec<TF> &self, PoomVec<TF> &that ) { self.operator+=( that ); } )
        .def( "self_sub", []( PoomVec<TF> &self, PoomVec<TF> &that ) { self.operator-=( that ); } )
        .def( "self_div", []( PoomVec<TF> &self, TF that ) { self.operator/=( that ); } )
        ;

    // "named" ctor
    m.def( "make_PoomVec_from_ndarray", []( const Array_TF &inp ) -> PoomVec<TF> {
        return CstSpan<TF>( reinterpret_cast<const TF *>( inp.data() ), inp.size() );
    } );

}
