#include <tl/support/string/to_string.h>
#include <sdot/poom/PoomVec.h>
#include "config.h"
 
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifndef SDOT_CONFIG_scalar_type
#define SDOT_CONFIG_scalar_type FP64
#endif

#ifndef SDOT_CONFIG_item_type
#define SDOT_CONFIG_item_type FP64
#endif

using namespace sdot;

using ScalarType = SDOT_CONFIG_scalar_type;
using ItemType = SDOT_CONFIG_item_type;
using Array = pybind11::array_t<ScalarType, pybind11::array::c_style>;

// Pt Pt_from_Array( const Array &array ) {
//     Pt res;
//     if ( array.size() < nb_dims )
//         throw pybind11::value_error( "array is not large enough" );
//     for( PI d = 0; d < nb_dims; ++d )
//        res[ d ] = array.at( d );
//     return res;
// }

// Vec<Pt> VecPt_from_Array( const Array &array ) {
//     Vec<Pt> res;
//     if ( array.shape( 1 ) < nb_dims )
//         throw pybind11::value_error( "array is not large enough" );
//     res.resize( array.shape( 0 ) );
//     for( PI r = 0; r < res.size(); ++r )
//         for( PI d = 0; d < nb_dims; ++d )
//            res[ r ][ d ] = array.at( r, d );
//     return res;
// }

// template<class T,int s>
// static auto Array_from_VecPt( const Vec<Vec<T,s>> &v ) {
//     Vec<PI,2> shape{ v.size(), PI( s ) };
//     pybind11::array_t<T, pybind11::array::c_style> res( shape );
//     for( PI i = 0; i < v.size(); ++i )
//         for( PI d = 0; d < s; ++d )
//             res.mutable_at( i, d ) = v[ i ][ d ];
//     return res;
// }

PYBIND11_MODULE( SDOT_CONFIG_module_name, m ) { // py::module_local()
    pybind11::class_<PoomVec<ItemType>>( m, PD_STR( VtkOutput ) )
        // .def( pybind11::init<>() )
        .def( "__repr__", []( const PoomVec<ItemType> &cell ) { return to_string( cell ); } )
        ;

    m.def( "make_PoomVec_from_ndarray", []( const Array &inp ) -> PoomVec<ItemType> {
        return CstSpan<ItemType>( inp.data(), inp.size() );
    } );

}
