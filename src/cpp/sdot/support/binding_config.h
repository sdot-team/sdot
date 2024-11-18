#pragma once

#include <tl/support/containers/Vec.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifndef SDOT_CONFIG_module_name
#define SDOT_CONFIG_module_name smurf
#endif

#ifndef SDOT_CONFIG_suffix
#define SDOT_CONFIG_suffix smurf
#endif

#ifndef SDOT_CONFIG_scalar_type
#define SDOT_CONFIG_scalar_type FP64
#endif

#ifndef SDOT_CONFIG_nb_dims
#define SDOT_CONFIG_nb_dims 3
#endif

#ifndef SDOT_CONFIG_arch
#define SDOT_CONFIG_arch YoArch
#endif

// helper for PD_NAME
#define PD_CONCAT_MACRO_( A, B ) A##_##B
#define PD_CONCAT_MACRO( A, B ) PD_CONCAT_MACRO_( A, B)

#define PD_STR_CONCAT_MACRO_( A, B ) #A "_" #B
#define PD_STR_CONCAT_MACRO( A, B ) PD_STR_CONCAT_MACRO_( A, B )

/// concatenation with $SDOT_CONFIG_suffix
#define PD_NAME( EXPR ) PD_CONCAT_MACRO( EXPR, SDOT_CONFIG_suffix )

/// string with $SDOT_CONFIG_suffix
#define PD_STR( EXPR ) PD_STR_CONCAT_MACRO( EXPR, SDOT_CONFIG_suffix )

namespace sdot { struct SDOT_CONFIG_arch {}; }

template<class T,int nb_dims> static Vec<Vec<T,nb_dims>> vec_of_svec_from_array( const pybind11::array_t<T, pybind11::array::c_style> &array ) {
    Vec<Vec<T,nb_dims>> res;

    if ( array.shape( 1 ) < nb_dims )
        throw pybind11::value_error( "array is not large enough" );

    res.resize( array.shape( 0 ) );
    for( PI r = 0; r < res.size(); ++r )
        for( PI d = 0; d < nb_dims; ++d )
           res[ r ][ d ] = array.at( r, d );
    
    return res;
}

template<class T,int nb_dims> static Vec<T,nb_dims> svec_from_array( const pybind11::array_t<T, pybind11::array::c_style> &array ) {
    Vec<T,nb_dims> res;
    
    if ( array.size() < nb_dims )
        throw pybind11::value_error( "array is not large enough" );
    
    for( PI d = 0; d < nb_dims; ++d )
       res[ d ] = array.at( d );
    
    return res;
}

template<class T,int n_dims>
static auto array_from_vec( const Vec<Vec<T,n_dims>> &v ) {
    Vec<PI,2> shape{ v.size(), PI( n_dims ) };
    pybind11::array_t<T, pybind11::array::c_style> res( shape );

    for( PI i = 0; i < v.size(); ++i )
        for( PI d = 0; d < n_dims; ++d )
            res.mutable_at( i, d ) = v[ i ][ d ];

    return res;
}

template<class T,int nb_dims>
static auto array_from_vec( const Vec<T,nb_dims> &v ) {
    Vec<PI,1> shape{ v.size() };

    pybind11::array_t<T, pybind11::array::c_style> res( shape );
    for( PI i = 0; i < v.size(); ++i )
        res.mutable_at( i ) = v[ i ];

    return res;
}
