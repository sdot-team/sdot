#pragma once

#include "StrideIterator.h"
#include "PointFactory.h"
#include "TensorView.h"
#include "ASSERT.h"
#include "TODO.h"

#include <algorithm>

namespace sdot {

#define UTP template<class T,int ct_rank,class Arch>
#define DTP TensorView<T,ct_rank,Arch>

UTP DTP::TensorView( T *data, Sizes sizes, Strides strides ) : _strides( strides ), _sizes( sizes ), _ptr( reinterpret_cast<Ptr>( data ) ) {
}

UTP DTP::TensorView( T *data, Sizes sizes ) : _strides( contiguous_strides( sizes ) ), _sizes( sizes ), _ptr( reinterpret_cast<Ptr>( data ) ) {
}

UTP DTP::TensorView( T *data, PI size ) : _sizes{ size }, _strides{ sizeof( T ) }, _ptr( reinterpret_cast<Ptr>( data ) ) {
}

UTP DTP::TensorView() : _strides( ct_rank >= 0 ? ct_rank : 0 ), _sizes( ct_rank >= 0 ? ct_rank : 0 ), _ptr( nullptr ) {
}

UTP T& DTP::operator()( const auto &indices, auto ...rem ) const requires ( requires { indices.size(); } ) {
    if ( indices.size() )
        return operator()( indices[ 0 ], indices.without_index( 0 ), rem... );
    return operator()( rem... );
}

UTP T& DTP::operator()( PI index, auto ...rem ) const {
    return row( index )( rem... );
}

UTP T& DTP::operator()() const {
    ASSERT( rank() == 0 );
    return *reinterpret_cast<T *>( _ptr );
}

UTP T& DTP::operator[]( const auto &index ) const {
    return operator()( index );
}

UTP DTP::Strides DTP::strides() const {
    return _strides;
}

UTP SI DTP::stride( PI d ) const {
    return _strides[ d ];
}

UTP DTP::Sizes DTP::sizes() const {
    return _sizes;
}

UTP PI DTP::size( PI d ) const {
    return _sizes[ d ];
}

UTP PI DTP::size() const {
    ASSERT( rank() == 1 );
    return size( 0 );
}

UTP bool DTP::empty() const {
    return rank() == 0 ? _ptr == nullptr : std::none_of( _sizes.begin(), _sizes.end(), []( auto a ) { return a != 0; } );
}

UTP PI DTP::rank() const {
    return ct_rank >= 0 ? ct_rank : _sizes.size();
}

UTP T* DTP::data() const {
    return reinterpret_cast<T *>( _ptr );
}

UTP auto DTP::begin() const {
    ASSERT( rank() == 1 );
    return StrideIterator<T>{ _ptr, _strides[ 0 ], 0 };
}

UTP auto DTP::end() const {
    return StrideIterator<T>{ _ptr, _strides[ 0 ], _sizes[ 0 ] };
}

UTP auto DTP::squeeze( PI axis, PI index ) const {
    constexpr int new_ct_rank = ct_rank >= 0 ? ct_rank - 1 : -1;
    return TensorView<T,new_ct_rank,Arch>{ reinterpret_cast<T *>( _ptr + _strides[ axis ] * index ), _sizes.without_index( axis ), _strides.without_index( axis ) };
}

UTP auto DTP::row( PI index ) const {
    return squeeze( 0, index );
}

UTP void DTP::for_each_index( auto &&func, PI sub ) const {
    PointFactory<PI,ct_rank,Arch> pf( rank() );
    Point<PI,ct_rank,Arch> index = pf.zeros();
    while ( true ) {
        func( index );

        PI n = rank() - 1;
        while ( ++index[ n ] == size( n ) - sub ) {
            if ( n == 0 )
                return;
            index[ n ] = 0;
            --n;
        }
    }
}

UTP auto DTP::contiguous_strides( const Sizes &ext ) -> Strides {
    Strides s;
    s[ ct_rank - 1 ] = sizeof( T );
    for( int i = ct_rank - 2; i >= 0; --i )
        s[ i ] = s[ i + 1 ] * ext[ i + 1 ];
    return s;
}

UTP template<class TF>
Tensor<TF,1,Arch> DTP::sum_along_axis_1() const {
    Tensor<TF,1,Arch> res( sdot::Shape(), { _sizes[ 0 ] } );

    // cuda
    #ifdef __CUDACC__
    if constexpr ( std::is_same_v<Arch,Cuda> ) {
        static_assert( std::is_same_v<std::decay_t<TF>,std::decay_t<T>>, "for now we support only homogeneous types for CUDA" );
        static_assert( ct_rank == 2, "CUDA sum_axis_0 only supports rank-2 for now" );

        // columns within a row must be contiguous
        ASSERT( _strides[ 1 ] == (SI)sizeof( T ) );

        const int pitch( _strides[ 0 ] / sizeof( T ) );  // row stride in elements (>= ncols if padding)
        const int nrows( _sizes[ 0 ] );
        const int ncols( _sizes[ 1 ] );

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

#ifdef __CUDACC__
// Functor at namespace scope so CUDA/cudafe can properly mangle the type in Thrust typedefs
// (local structs with __device__ members cause "template argument N is invalid" in cudafe)
template<class T, int ct_rank>
struct TensorViewIndexer {
    __device__ const T &operator()( sdot::PI i ) const {
        sdot::SI off = 0;
        for ( int d = ct_rank - 1; d >= 0; --d ) {
            off += sdot::SI( i % ext[ d ] ) * str[ d ];
            i /= ext[ d ];
        }
        return *reinterpret_cast<const T *>( src + off );
    }

    const std::byte *src;
    sdot::PI ext[ ct_rank ];
    sdot::SI str[ ct_rank ];
};

UTP void DTP::with_cpu_version( auto &&func ) const {
    TensorViewIndexer<T,ct_rank> es;
    es.src = reinterpret_cast<const std::byte *>( data() );

    PI size = 1;
    for( int d = 0; d < ct_rank; ++d ) {
        es.str[ d ] = stride( d );
        es.ext[ d ] = shape( d );
        size *= es.ext[ d ];
    }

    // gather strided GPU elements into a flat contiguous GPU buffer
    using NT = std::remove_const_t<T>;
    thrust::device_vector<NT> dev( size );
    thrust::transform( thrust::counting_iterator<sdot::PI>( 0 ), thrust::counting_iterator<sdot::PI>( size ), dev.begin(), es );

    // copy to host and print via CPU TensorView (contiguous)
    std::vector<NT> host( size );
    thrust::copy( dev.begin(), dev.end(), host.begin() );

    sdot::TensorView<NT,ct_rank,sdot::Cpu> tensor( host.data(), shape() );
    return func( tensor );
}
#endif

#undef UTP
#undef DTP

} // namespace sdot


#ifdef __CUDACC__
template<class T,int ct_rank>
std::ostream &operator<<( std::ostream &os, const sdot::TensorView<T,ct_rank,sdot::Cuda> &p ) {
    p.with_cpu_version( [&]( auto t ) {
        os << t;
    } );
    return os;
}
#endif

