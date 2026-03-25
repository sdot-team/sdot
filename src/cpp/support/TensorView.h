#pragma once

#include "common_macros.h"
#include "common_types.h"
#include "ASSERT.h"
#include "Arch.h"

#ifdef __CUDACC__
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/cub.cuh>
#endif

#include <algorithm>
#include <ostream>
#include <array>

namespace sdot {
template<class T,int ct_rank,class Arch>
class Tensor;

/// view on strided data (strides in bytes, handles non-contiguous arrays)
template<class T,int ct_rank,class Arch>
class TensorView {
public:
    using       Strides           = std::array<SI,ct_rank>;  ///< byte strides
    using       Shape             = std::array<PI,ct_rank>;
    using       RawPtr            = std::conditional_t<std::is_const_v<T>,const std::byte*,std::byte*>;

    /* */       TensorView        ( T *data, Shape shape, Strides strides ) : _strides( strides ), _shape( shape ), _ptr( reinterpret_cast<RawPtr>( data ) ) {}
    /* */       TensorView        ( T *data, Shape shape ) : _strides( contiguous_strides( shape ) ), _shape( shape ), _ptr( reinterpret_cast<RawPtr>( data ) ) {}
    /* */       TensorView        ( T *data, PI size ) : _shape{ size }, _strides{ sizeof( T ) }, _ptr( reinterpret_cast<RawPtr>( data ) ) {}

    static auto contiguous_strides( const Shape &ext ) -> Strides;

    T&          operator()        ( PI i0, PI i1, PI i2 ) const { return *reinterpret_cast<T *>( _ptr + i0 * _strides[ 0 ] + i1 * _strides[ 1 ] + i2 * _strides[ 2 ] ); }
    T&          operator()        ( PI i0, PI i1 ) const { return *reinterpret_cast<T *>( _ptr + i0 * _strides[ 0 ] + i1 * _strides[ 1 ] ); }
    T&          operator()        ( PI i0 ) const { return *reinterpret_cast<T *>( _ptr + i0 * _strides[ 0 ] ); }
    T&          operator()        () const { return *reinterpret_cast<T *>( _ptr ); }

    T&          operator[]        ( PI i0 ) const { return *reinterpret_cast<T *>( _ptr + i0 * _strides[ 0 ] ); }

    auto        strides           () const { return _strides; }
    SI          stride            ( PI d ) const { return _strides[ d ]; }
    SI          shape             ( PI d ) const { return _shape[ d ]; }
    auto        shape             () const { return _shape; }
    bool        empty             () const { return ct_rank == 0 ? false : std::none_of( _shape.begin(), _shape.end(), []( auto a ) { return a != 0; } ); }
    PI          rank              () const { return ct_rank; }
    PI          size              ( PI d ) const { return _shape[ d ]; }
    PI          size              () const { ASSERT( rank() == 1 ); return size( 0 ); }

    T*          data              () const { return reinterpret_cast<T *>( _ptr ); }

    T_U auto    sum_along_axis_1  () const -> Tensor<U,1,Arch>;
    void        with_cpu_version  ( auto &&func ) const;
    auto        squeeze           ( PI axis ) const -> TensorView<T,ct_rank-1,Arch>;
    auto        row               ( PI index ) const -> TensorView<T,ct_rank-1,Arch>;

private:
    Shape       _shape;   ///<
    Strides     _strides;  ///< byte strides
    RawPtr      _ptr;      ///<
};

} // namespace sdot

template<class T,int ct_rank>
std::ostream &operator<<( std::ostream &os, const sdot::TensorView<T,ct_rank,sdot::Cpu> &p );

#ifdef __CUDACC__
template<class T,int ct_rank>
std::ostream &operator<<( std::ostream &os, const sdot::TensorView<T,ct_rank,sdot::Cuda> &p );
#endif

#include "TensorView.cxx"
