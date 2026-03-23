#pragma once

#include "../../../cpp/support/TensorView.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace sdot {


// to sdot type
template<class TF>
static auto tensor_view_1( const nanobind::ndarray<TF> &v ) {
    std::array<PI,1> extent { v.shape( 0 ) };
    std::array<PI,1> strides{ PI( v.stride( 0 ) ) };
    return TensorView<TF,1>( v.data(), extent, strides );
}

template<class TF>
static auto tensor_view_2( const nanobind::ndarray<TF> &v ) {
    std::array<PI,2> extent { v.shape( 0 ), v.shape( 1 ) };
    std::array<PI,2> strides{ PI( v.stride( 0 ) ), PI( v.stride( 1 ) ) };
    return TensorView<TF,2>( v.data(), extent, strides );
}

template<class TF>
static auto tensor_view_3( const nanobind::ndarray<TF> &v ) {
    std::array<PI,3> extent { v.shape( 0 ), v.shape( 1 ), v.shape( 2 ) };
    std::array<PI,3> strides{ PI( v.stride( 0 ) ), PI( v.stride( 1 ) ), PI( v.stride( 2 ) ) };
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

