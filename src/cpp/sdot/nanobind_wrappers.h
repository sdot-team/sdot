#pragma once

#include "support/common_macros.h"
#include "support/vector_map.h"
#include "support/TensorView.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <optional>

namespace sdot {

// nanobind arch -> sdot arch
T_T struct ArchFor {};

template<> struct ArchFor<nanobind::device::cpu> { using type = Cpu; };

#ifdef __CUDACC__
template<> struct ArchFor<nanobind::device::cuda> { using type = Cuda; };
#endif

// to sdot type versions (non-optional) ---------------------------------------------------------------
template<class TF,class TA>
static auto tensor_view_0( const nanobind::ndarray<TF,TA> &v ) {
    ASSERT( v.ndim() == 0 );
    std::array<SI,0> strides{};
    std::array<PI,0> extent{};
    return TensorView<TF,0,typename ArchFor<TA>::type>( (TF *)v.data(), extent, strides );
}

template<class TF,class TA>
static auto tensor_view_1( const nanobind::ndarray<TF,TA> &v ) {
    ASSERT( v.ndim() == 1 );
    std::array<SI,1> strides{ v.stride( 0 ) * SI( sizeof( TF ) ) };
    std::array<PI,1> extent{ v.shape( 0 ) };
    return TensorView<TF,1,typename ArchFor<TA>::type>( (TF *)v.data(), extent, strides );
}

template<class TF,class TA>
static auto tensor_view_2( const nanobind::ndarray<TF,TA> &v ) {
    ASSERT( v.ndim() == 2 );
    std::array<SI,2> strides{ v.stride( 0 ) * SI( sizeof( TF ) ), v.stride( 1 ) * SI( sizeof( TF ) ) };
    std::array<PI,2> extent{ v.shape( 0 ), v.shape( 1 ) };
    return TensorView<TF,2,typename ArchFor<TA>::type>( (TF *)v.data(), extent, strides );
}

template<class TF,class TA>
static auto tensor_view_3( const nanobind::ndarray<TF,TA> &v ) {
    ASSERT( v.ndim() == 3 );
    std::array<SI,3> strides{ v.stride( 0 ) * SI( sizeof( TF ) ), v.stride( 1 ) * SI( sizeof( TF ) ), v.stride( 2 ) * SI( sizeof( TF ) ) };
    std::array<PI,3> extent { v.shape( 0 ), v.shape( 1 ), v.shape( 2 ) };
    return TensorView<TF,3,typename ArchFor<TA>::type>( (TF *)v.data(), extent, strides );
}

// to sdot type versions (optional — None → invalid TensorView with _ptr==nullptr) --------------------
template<class TF,class TA>
static auto tensor_view_0( const std::optional<nanobind::ndarray<TF,TA>> &v ) {
    using TV = TensorView<TF,0,typename ArchFor<TA>::type>;
    return v ? tensor_view_0( *v ) : TV::make_invalid( 0 );
}

template<class TF,class TA>
static auto tensor_view_1( const std::optional<nanobind::ndarray<TF,TA>> &v ) {
    using TV = TensorView<TF,1,typename ArchFor<TA>::type>;
    return v ? tensor_view_1( *v ) : TV::make_invalid( 1 );
}

template<class TF,class TA>
static auto tensor_view_2( const std::optional<nanobind::ndarray<TF,TA>> &v ) {
    using TV = TensorView<TF,2,typename ArchFor<TA>::type>;
    return v ? tensor_view_2( *v ) : TV::make_invalid( 2 );
}

template<class TF,class TA>
static auto tensor_view_3( const std::optional<nanobind::ndarray<TF,TA>> &v ) {
    using TV = TensorView<TF,3,typename ArchFor<TA>::type>;
    return v ? tensor_view_3( *v ) : TV::make_invalid( 3 );
}

// vectors of views -----------------------------------------------------------------------------------
template<class TF,class TA>
static auto tensor_views_1( const std::vector<std::optional<nanobind::ndarray<TF,TA>>> &v ) {
    return vector_map( v, []( const auto &i ) { return tensor_view_1( i ); } );
}

template<class TF,class TA>
static auto tensor_views_2( const std::vector<std::optional<nanobind::ndarray<TF,TA>>> &v ) {
    return vector_map( v, []( const auto &i ) { return tensor_view_2( i ); } );
}

template<class TF,class TA>
static auto tensor_views_3( const std::vector<std::optional<nanobind::ndarray<TF,TA>>> &v ) {
    return vector_map( v, []( const auto &i ) { return tensor_view_3( i ); } );
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
