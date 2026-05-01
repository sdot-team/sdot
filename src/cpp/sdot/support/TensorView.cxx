#pragma once

#include "StrideIterator.h"
#include "DsVecFactory.h"
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

UTP DTP::TensorView( T *data, PI size ) : _sizes( Values(), size ), _strides( Values(), sizeof( T ) ), _ptr( reinterpret_cast<Ptr>( data ) ) {
}

UTP DTP::TensorView( Rank, PI rank ) : _strides( Size(), rank, 0 ), _sizes( Size(), rank, 0 ), _ptr( nullptr ) {
}

UTP DTP &DTP::get_data_from( const TensorView<T,ct_rank,Arch> &that ) {
    if ( is_contiguous() ) {
        std::memcpy( data(), that.data(), sizeof( T ) * total_size() );
        return *this;
    }
    TODO;
}

UTP void DTP::fill_with( T value ) {
    for_each_index( [&]( auto ...indices ) {
        operator()( indices... ) = value;
    } );
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

UTP auto DTP::partial( auto ...indices ) const {
    PI i = 0;
    T *ptr = data() + ( ( _strides[ i++ ] * indices ) + ... + 0 );
    return TensorView<T,ct_rank-sizeof...(indices),Arch>( ptr, _sizes.template from<sizeof...(indices)>(), _strides.template from<sizeof...(indices)>() );
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

UTP PI DTP::total_size() const {
    PI res = 1;
    for( PI d = 0; d < rank(); ++d )
        res *= size( d );
    return res;
}

UTP PI DTP::size() const {
    ASSERT( rank() == 1 );
    return size( 0 );
}

UTP std::byte DTP::_sentinel{};

UTP auto DTP::make_invalid( PI rank ) {
    TensorView res( Rank(), rank );
    res._ptr = reinterpret_cast<Ptr>( &_sentinel );
    return res;
}

UTP bool DTP::is_invalid() const {
    return _ptr == reinterpret_cast<const Ptr>( &_sentinel );
}

UTP bool DTP::is_valid() const {
    return _ptr != reinterpret_cast<const Ptr>( &_sentinel );
}

UTP bool DTP::empty() const {
    if ( rank() == 0 )
        return _ptr == nullptr;
    for ( PI i = 0; i < rank(); ++i )
        if ( _sizes[ i ] == 0 )
            return true;
    return false;
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

UTP auto DTP::unsqueeze() const {
    // Append a trailing dimension of size 1.
    // The stride for the new axis is sizeof(T): matches contiguous layout when the source is contiguous.
    // For a size-1 axis the stride value is irrelevant for correctness, but we set it consistently.
    constexpr int new_ct_rank = ct_rank >= 0 ? ct_rank + 1 : -1;
    return TensorView<T,new_ct_rank,Arch>( data(), _sizes.with_pushed_value( PI( 1 ) ), _strides.with_pushed_value( SI( sizeof( T ) ) ) );
}

UTP bool DTP::is_contiguous() const {
    // Check that strides match the standard row-major (C-order) contiguous layout.
    SI expected = sizeof( T );
    for ( int i = int( rank() ) - 1; i >= 0; --i ) {
        if ( _strides[ i ] != expected )
            return false;
        expected *= SI( _sizes[ i ] );
    }
    return true;
}

UTP void DTP::for_each_index( auto &&func, PI sub ) const {
    DsVecFactory<PI,ct_rank,Arch> pf( rank() );
    if ( rank() == 0 ) {
        func( pf.zeros() );
        return;
    }
    DsVec<PI,ct_rank,Arch> index = pf.zeros();
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
    Strides s( Size(), ext.size() );
    if ( ext.size() ) {
        s[ ext.size() - 1 ] = sizeof( T );
        for( int i = int( ext.size() ) - 2; i >= 0; --i )
            s[ i ] = s[ i + 1 ] * ext[ i + 1 ];
    }
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

