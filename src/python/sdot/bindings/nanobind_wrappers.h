#pragma once

#include "../../../cpp/support/TensorView.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace sdot {


// to sdot type
// nanobind strides are in elements (DLPack convention), not bytes — multiply by sizeof(TF)
template<class TF>
static auto tensor_view_1( const nanobind::ndarray<TF> &v ) {
    std::array<SI,1> strides{ v.stride( 0 ) * SI( sizeof( TF ) ) };
    std::array<PI,1> extent { v.shape( 0 ) };
    return TensorView<TF,1>( v.data(), extent, strides );
}

template<class TF>
static auto tensor_view_2( const nanobind::ndarray<TF> &v ) {
    std::array<SI,2> strides{ v.stride( 0 ) * SI( sizeof( TF ) ), v.stride( 1 ) * SI( sizeof( TF ) ) };
    std::array<PI,2> extent { v.shape( 0 ), v.shape( 1 ) };
    return TensorView<TF,2>( v.data(), extent, strides );
}

template<class TF>
static auto tensor_view_3( const nanobind::ndarray<TF> &v ) {
    std::array<SI,3> strides{ v.stride( 0 ) * SI( sizeof( TF ) ), v.stride( 1 ) * SI( sizeof( TF ) ), v.stride( 2 ) * SI( sizeof( TF ) ) };
    std::array<PI,3> extent { v.shape( 0 ), v.shape( 1 ), v.shape( 2 ) };
    return TensorView<TF,3>( v.data(), extent, strides );
}


// to nanobind types
template<class Vec>
static auto to_ndarray_1d( Vec &&vec ) {
    using T = std::decay_t<Vec>;
    auto *ptr = new T( std::move( vec ) );
    nanobind::capsule owner( ptr, []( void *p ) noexcept { delete static_cast<T *>( p ); } );
    return nanobind::ndarray<nanobind::numpy, typename T::value_type>( ptr->data(), { ptr->size() }, owner );
}

template<class Mat>
static auto to_ndarray_2d( Mat &&vec ) {
    using T = std::decay_t<Mat>;
    auto *ptr = new T( std::move( vec ) );
    nanobind::capsule owner( ptr, []( void *p ) noexcept { delete static_cast<T *>( p ); } );
    return nanobind::ndarray<nanobind::numpy, typename T::value_type>( ptr->data(), { ptr->size( 0 ), ptr->size( 1 ) }, owner );
}

} // namespace sdot

