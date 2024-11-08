#include <tl/support/type_info/type_name.h>
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

using Array_TF = pybind11::array_t<ScalarType, pybind11::array::c_style>;
using Array_PI = pybind11::array_t<PI, pybind11::array::c_style>;

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

template<class T>
Array_TF poom_vec_to_ndarray( PoomVec<T> &vec ) {
    Vec<PI,1> shape{ vec.size() };
    pybind11::array_t<T, pybind11::array::c_style> res( shape );
    vec.get_values_by_chuncks( [&]( CstSpanView<ItemType> sv ) {
        for( PI i = sv.beg_index(); i < sv.end_index(); ++i )
            res.mutable_at( i ) = sv[ i ];
    } );
    return res;
}

template<class T,int s>
Array_TF poom_vec_to_ndarray( PoomVec<Vec<T,s>> &vec ) {
    Vec<PI,2> shape{ vec.size(), PI( s ) };
    pybind11::array_t<T, pybind11::array::c_style> res( shape );
    vec.get_values_by_chuncks( [&]( CstSpanView<Vec<T,s>> sv ) {
        for( PI i = sv.beg_index(); i < sv.end_index(); ++i )
            for( int d = 0; d < s; ++d )
                res.mutable_at( i, d ) = sv[ i ][ d ];
    } );
    return res;
}

template<class T>
Str poom_vec_dtype( PoomVec<T> &vec ) {
    return type_name<T>();
}

template<class T,int s>
Str poom_vec_dtype( PoomVec<Vec<T,s>> &vec ) {
    return type_name<T>();
}

template<class T>
Array_PI poom_vec_shape( PoomVec<T> &vec ) {
    Vec<PI,1> shape{ 1 };
    Array_PI res( shape );
    res.mutable_at( 0 ) = vec.size();
    return res;
}

template<class T,int s>
Array_PI poom_vec_shape( PoomVec<Vec<T,s>> &vec ) {
    Vec<PI,1> shape{ 2 };
    Array_PI res( shape );
    res.mutable_at( 0 ) = vec.size();
    res.mutable_at( 1 ) = s;
    return res;
}

PYBIND11_MODULE( SDOT_CONFIG_module_name, m ) { // py::module_local()
    pybind11::class_<PoomVec<ItemType>>( m, PD_STR( VtkOutput ) )
        // .def( pybind11::init<>() )
        .def( "__repr__", []( const PoomVec<ItemType> &vec ) { return to_string( vec ); } )
        .def( "as_ndarray", []( PoomVec<ItemType> &vec ) { return poom_vec_to_ndarray( vec ); } )
        .def( "dtype", []( PoomVec<ItemType> &vec ) { return poom_vec_dtype( vec ); } )
        .def( "shape", []( PoomVec<ItemType> &vec ) { return poom_vec_shape( vec ); } )
        ;

    m.def( "make_PoomVec_from_ndarray", []( const Array_TF &inp ) -> PoomVec<ItemType> {
        return CstSpan<ItemType>( reinterpret_cast<const ItemType *>( inp.data() ), inp.shape( 0 ) );
    } );

}
