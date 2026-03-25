#pragma once

#include "common_types.h"
#include "ASSERT.h"
#include "TODO.h"
#include "Arch.h"

#ifdef __CUDACC__
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
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
    using  Strides   = std::array<SI,ct_rank>;  ///< byte strides
    using  Extent    = std::array<PI,ct_rank>;
    using  RawPtr    = std::conditional_t<std::is_const_v<T>,const std::byte*,std::byte*>;

    /* */  TensorView( T *data, Extent ext, Strides str ) : extent( ext ), strides( str ), ptr( reinterpret_cast<RawPtr>( data ) ) {}
    /* */  TensorView( T *data, Extent ext ) : extent( ext ), strides( contiguous_strides( ext ) ), ptr( reinterpret_cast<RawPtr>( data ) ) {}
    /* */  TensorView( T *data, PI size ) : extent{ size }, strides{ sizeof( T ) }, ptr( reinterpret_cast<RawPtr>( data ) ) {}

    T&     operator()( PI i0, PI i1, PI i2 ) const { return *reinterpret_cast<T *>( ptr + i0 * strides[ 0 ] + i1 * strides[ 1 ] + i2 * strides[ 2 ] ); }
    T&     operator()( PI i0, PI i1 ) const { return *reinterpret_cast<T *>( ptr + i0 * strides[ 0 ] + i1 * strides[ 1 ] ); }
    T&     operator()( PI i0 ) const { return *reinterpret_cast<T *>( ptr + i0 * strides[ 0 ] ); }
    T&     operator()() const { return *reinterpret_cast<T *>( ptr ); }

    T&     operator[]( PI i0 ) const { return *reinterpret_cast<T *>( ptr + i0 * strides[ 0 ] ); }

    bool   empty     () const { return ct_rank == 0 ? false : std::none_of( extent.begin(), extent.end(), []( auto a ) { return a != 0; } ); }

    //autp   strides   () const { return strides[ d ]; }
    SI     stride    ( PI d ) const { return strides[ d ]; }

    PI     size      ( PI d ) const { return extent[ d ]; }
    PI     size      () const { ASSERT( rank() == 1 ); return size( 0 ); }
    PI     rank      () const { return ct_rank; }

    T*     data      () const { return reinterpret_cast<T *>( ptr ); }

    auto   squeeze   ( PI axis ) const -> TensorView<T,ct_rank-1,Arch> {
        std::array<SI,ct_rank-1> nstrides;
        std::array<PI,ct_rank-1> nextent;
        for( PI i = 0; i < ct_rank-1; ++i ) {
            nstrides[ i ] = strides[ i + ( i >= axis ) ];
            nextent[ i ] = extent[ i + ( i >= axis ) ];
        }
        return { reinterpret_cast<T *>( ptr ), nextent, nstrides };
    }

    auto   row       ( PI index ) const -> TensorView<T,ct_rank-1,Arch> {
        std::array<SI,ct_rank-1> nstrides;
        std::array<PI,ct_rank-1> nextent;
        for( PI i = 1; i < ct_rank; ++i ) {
            nstrides[ i - 1 ] = strides[ i ];
            nextent[ i - 1 ] = extent[ i ];
        }
        return { reinterpret_cast<T *>( ptr + index * strides[ 0 ] ), nextent, nstrides };
    }

    static Strides contiguous_strides( const Extent &ext ) {
        Strides s;
        s[ ct_rank - 1 ] = sizeof( T );
        for( int i = ct_rank - 2; i >= 0; --i )
            s[ i ] = s[ i + 1 ] * ext[ i + 1 ];
        return s;
    }

    template<class TF>
    Tensor<TF,1,Arch> sum_axis_0() const {
        Tensor<TF,1,Arch> res( sdot::Extent(), { extent[ 0 ] } );

        // cuda
        #ifdef __CUDACC__
        if constexpr ( std::is_same_v<Arch,Cuda> ) {
            static_assert( std::is_same_v<std::decay_t<TF>,std::decay_t<T>>, "for now we support only homogeneous types for CUDA" );
            static_assert( ct_rank == 2, "CUDA sum_axis_0 only supports rank-2 for now" );

            // columns within a row must be contiguous
            ASSERT( strides[ 1 ] == (SI)sizeof( T ) );

            const int pitch( strides[ 0 ] / sizeof( T ) );  // row stride in elements (>= ncols if padding)
            const int nrows( extent[ 0 ] );
            const int ncols( extent[ 1 ] );

            // segment [r] = [ r*pitch, r*pitch + ncols )
            auto row_idx = thrust::counting_iterator<int>( 0 );
            auto off_b = thrust::make_transform_iterator( row_idx, [ pitch ] __device__ __host__ ( int r ) { return r * pitch; } );
            auto off_e = thrust::make_transform_iterator( row_idx, [ pitch, ncols ] __device__ __host__ ( int r ) { return r * pitch + ncols; } );

            // first pass: call measures workspace size
            size_t tmp_sz = 0;
            cub::DeviceSegmentedReduce::Sum( nullptr, tmp_sz, data(), res.data(), nrows, off_b, off_e );

            // second run the reduction
            thrust::device_vector<char> tmp( tmp_sz );
            cub::DeviceSegmentedReduce::Sum( static_cast<void *>( thrust::raw_pointer_cast( tmp.data() ) ), tmp_sz, data(), res.data(), nrows, off_b, off_e );

            return res;
        }
        #endif

        // cpu
        TODO;
        return res;
    }

private:
    Strides strides;  ///< byte strides
    Extent  extent;   ///<
    RawPtr  ptr;      ///<
};

} // namespace sdot

template<class T,int ct_rank>
std::ostream &operator<<( std::ostream &os, const sdot::TensorView<T,ct_rank,sdot::Cpu> &p ) {
    if constexpr( ct_rank == 0 )
        return os << p();
    else if constexpr ( ct_rank == 1 ) {
        for( sdot::PI i = 0; i < p.size(); ++i )
            os << ( i ? ", " : "" ) << p[ i ];
        return os;
    } else {
        for( sdot::PI i = 0; i < p.size( 0 ); ++i )
            os << "\n" << p.row( i );
        return os;
    }
}

#ifdef __CUDACC__
template<class T,int ct_rank>
std::ostream &operator<<( std::ostream &os, const sdot::TensorView<T,ct_rank,sdot::Cuda> &p ) {
    using NT = std::remove_const_t<T>;

    // plain C arrays so that device lambda indexing uses built-in [] (not std::array::operator[])
    struct { sdot::PI ext[ ct_rank ]; sdot::SI str[ ct_rank ]; } es;
    sdot::PI size = 1;
    std::array<sdot::PI,ct_rank> ext;
    for( int d = 0; d < ct_rank; ++d ) {
        es.ext[ d ] = ext[ d ] = p.size( d );
        es.str[ d ] = p.stride( d );
        size *= es.ext[ d ];
    }

    // gather strided GPU elements into a flat contiguous GPU buffer
    const auto *src = reinterpret_cast<const std::byte *>( p.data() );
    thrust::device_vector<NT> dev( size );
    thrust::transform(
        thrust::counting_iterator<sdot::PI>( 0 ),
        thrust::counting_iterator<sdot::PI>( size ),
        dev.begin(),
        [src, es] __device__ ( sdot::PI i ) -> NT {
            sdot::SI off = 0;
            for ( int d = ct_rank - 1; d >= 0; --d ) {
                off += sdot::SI( i % es.ext[ d ] ) * es.str[ d ];
                i   /= es.ext[ d ];
            }
            return *reinterpret_cast<const T *>( src + off );
        } );

    // copy to host and print via CPU TensorView (contiguous)
    std::vector<NT> host( size );
    thrust::copy( dev.begin(), dev.end(), host.begin() );
    return os << sdot::TensorView<NT,ct_rank,sdot::Cpu>( host.data(), ext );
}
#endif
