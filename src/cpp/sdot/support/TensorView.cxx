#pragma once

#include "StrideIterator.h"
// #include "VectorFactory.h"
#include "TensorView.h"
#include "ASSERT.h"
#include "TODO.h"

// #include <algorithm>
#include <cstring>

namespace sdot {

#define UTP template<class TF,class Shape,class Strides>
#define DTP TensorView<TF,Shape,Strides>

UTP HD DTP DTP::make_invalid( Shape shape, Strides strides ) {
    return TensorView( reinterpret_cast<TF *>( sentinel() ), shape, strides );
}

UTP HD DTP::TensorView( TF *data, Shape shape, Strides strides ) : _raw_ptr( reinterpret_cast<RawPtr>( data ) ), _strides( strides ), _shape( shape ) {
}

UTP HD void DTP::get_data_from( const auto &that, const auto &size_to_take ) {
    for_each_index( [&]( auto indices ) {
        operator()( indices ) = that( indices );
    }, 0, size_to_take );
}

UTP HD void DTP::get_data_from( const auto &that ) {
    if ( is_contiguous() )
        std::memcpy( data(), that.data(), sizeof( TF ) * total_size() );
    else
        get_data_from( that, shape() );
}

UTP HD void DTP::with_same_shape( const auto &arch, auto &&func ) const {
    arch.template with_reservation<TF>( total_size(), [&]( TF *data ) {
        auto new_strides = contiguous_strides( _shape );
        using NewStrides = DECAYED_TYPE_OF( new_strides );
        TensorView<TF,Shape,NewStrides> res( data, _shape, new_strides );
        func( res );
    } );
}

UTP HD void DTP::spill_to( TensorView &that ) {
    that.get_data_from( *this );
    _raw_ptr = that._raw_ptr;
}

UTP HD void DTP::fill_with( TF value ) {
    for_each_index( [&]( auto ...indices ) {
        operator()( indices... ) = value;
    } );
}


UTP HD auto DTP::operator()( const auto &indices, auto ...rem ) const requires ( requires { indices.size(); } ) {
    if ( indices.size() )
        return operator()( indices[ Ct<int,0>() ], indices.without_index( Ct<int,0>() ), rem... );
    return operator()( rem... );
}

UTP HD auto DTP::operator()( const auto &index, auto ...rem ) const {
    return row( index )( rem... );
}

UTP HD TF &DTP::operator[]( const auto &index ) const {
    auto selection = operator()( index );
    return selection.item();
}

UTP HD TF &DTP::item() const {
    ASSERT( rank() == 0 );
    return *data();
}

// UTP auto DTP::slice( auto ...indices ) const {
//     auto new_strides = _strides.slice_from( Ci<int,sizeof...( indices )>() );
//     auto new_shape = _shape.slice_from( Ci<int,sizeof...( indices )>() );
//     using NewStrides = DECAYED_TYPE_OF( new_strides );
//     using NewShape = DECAYED_TYPE_OF( new_shape );

//     PI i = 0;
//     TF *ptr = reinterpret_cast<TF *>( _raw_ptr + ( ( _strides[ i++ ] * indices ) + ... + 0 ) );

//     return TensorView<TF,NewShape,NewStrides>( ptr, new_shape, new_strides );
// }

UTP HD Strides DTP::strides() const {
    return _strides;
}

UTP HD SI DTP::stride( auto d ) const {
    return _strides[ d ];
}

UTP HD PI DTP::total_size() const {
    PI res = 1;
    for( PI d = 0; d < rank(); ++d )
        res *= size( d );
    return res;
}

UTP HD auto DTP::size() const {
    ASSERT( rank() == 1 );
    return shape( Ct<int,0>() );
}

UTP HD bool DTP::surely_null() const {
    if ( is_invalid() )
        return true;
    // empty tensor (any dimension == 0)
    if ( _shape.has_value( []( auto size ) { return size < Ct<int,1>(); } ) )
        return true;
    // all strides zero (rank > 0) → surely-null by construction: all elements alias data()[0] == 0
    if ( rank() > 0 && ! _strides.has_value( []( auto s ) { return s != SI(0); } ) )
        return true;
    // single scalar: check value
    if ( ! _shape.has_value( []( auto size ) { return size > Ct<int,1>(); } ) )
        return *data() == 0;
    return false;
}

UTP HD bool DTP::is_invalid() const {
    return _raw_ptr == sentinel();
}

UTP HD bool DTP::is_valid() const {
    return _raw_ptr != sentinel();
}

UTP HD auto DTP::empty() const {
    return _shape.has_value( []( auto size ) { return size == Ct<int,0>(); } );
}

UTP HD auto DTP::rank() const {
    return  _shape.size();
}

UTP HD TF* DTP::data() const {
    return reinterpret_cast<TF *>( _raw_ptr );
}

UTP HD auto DTP::begin() const {
    if constexpr ( ct_rank == 1 || ct_rank == 0 ) {
        return data();
    } else {
        TODO;
    }
}

UTP HD auto DTP::end() const {
    if constexpr ( ct_rank == 0 ) {
        return data() + 1;
    } else if constexpr ( ct_rank == 1 ) {
        return data() + size();
    } else {
        TODO;
    }
}

UTP HD auto DTP::squeeze( auto axis, PI index ) const {
    auto new_shape   = _shape.without_axis( axis );
    auto new_strides = _strides.without_axis( axis );
    using NewShape   = DECAYED_TYPE_OF( new_shape );
    using NewStrides = DECAYED_TYPE_OF( new_strides );
    auto ptr = reinterpret_cast<TF *>( _raw_ptr + _strides[ axis ] * index );
    return TensorView<TF,NewShape,NewStrides>( ptr, new_shape, new_strides );
}

UTP HD auto DTP::row( PI index ) const {
    return squeeze( Ct<int,0>(), index );
}

UTP HD auto DTP::unsqueeze( auto axis ) const {
    // Append a dimension of size 1.
    // The stride for the new axis is sizeof(T): matches contiguous layout when the source is contiguous.
    // For a size-1 axis the stride value is irrelevant for correctness, but we set it consistently.
    // constexpr int new_ct_rank = ct_rank >= 0 ? ct_rank + 1 : -1;
    // return TensorView<T,new_ct_rank,Arch>( data(), _shape.with_pushed_value( PI( 1 ) ), _strides.with_pushed_value( SI( sizeof( T ) ) ) );
    TODO;
}

UTP HD bool DTP::is_contiguous() const {
    // Check that strides match the standard row-major (C-order) contiguous layout.
    SI expected = sizeof( TF );
    for ( int i = int( rank() ) - 1; i >= 0; --i ) {
        if ( _strides[ i ] != expected )
            return false;
        expected *= SI( _shape[ i ] );
    }
    return true;
}

// UTP void DTP::for_each_index( auto &&func, PI sub, Vector<PI,ct_rank,Arch> size_to_take ) const {
//     VectorFactory<PI,ct_rank,Arch> pf( rank() );
//     if ( rank() == 0 ) {
//         func( pf.zeros() );
//         return;
//     }

//     for( PI n = 0; n < rank(); ++n ) {
//         if ( size_to_take[ n ] == 0 )
//             return;
//         if ( size_to_take[ n ] < 0 )
//             size_to_take[ n ] = size( n );
//     }

//     Vector<PI,ct_rank,Arch> index = pf.zeros();
//     while ( true ) {
//         func( index );

//         PI n = rank() - 1;
//         while ( ++index[ n ] == size_to_take[ n ] - sub ) {
//             if ( n == 0 )
//                 return;
//             index[ n ] = 0;
//             --n;
//         }
//     }
// }

UTP HD void DTP::for_each_index( auto &&func ) const {
    _shape.for_each_index( func );
}

// UTP auto DTP::contiguous_strides( const Sizes &ext ) -> Strides {
//     Strides s( Size(), ext.size() );
//     if ( ext.size() ) {
//         s[ ext.size() - 1 ] = sizeof( T );
//         for( int i = int( ext.size() ) - 2; i >= 0; --i )
//             s[ i ] = s[ i + 1 ] * ext[ i + 1 ];
//     }
//     return s;
// }

// UTP template<class TF>
// Tensor<TF,1,Arch> DTP::sum_along_axis_1() const {
//     Tensor<TF,1,Arch> res( sdot::Shape(), { _shape[ 0 ] } );

//     // cuda
//     #ifdef __CUDACC__
//     if constexpr ( std::is_same_v<Arch,Cuda> ) {
//         static_assert( std::is_same_v<std::decay_t<TF>,std::decay_t<T>>, "for now we support only homogeneous types for CUDA" );
//         static_assert( ct_rank == 2, "CUDA sum_axis_0 only supports rank-2 for now" );

//         // columns within a row must be contiguous
//         ASSERT( _strides[ 1 ] == (SI)sizeof( T ) );

//         const int pitch( _strides[ 0 ] / sizeof( T ) );  // row stride in elements (>= ncols if padding)
//         const int nrows( _shape[ 0 ] );
//         const int ncols( _shape[ 1 ] );

//         // segment [r] = [ r*pitch, r*pitch + ncols )
//         auto row_idx = thrust::counting_iterator<int>( 0 );
//         auto off_b = thrust::make_transform_iterator( row_idx, [ pitch ] __device__ __host__ ( int r ) { return r * pitch; } );
//         auto off_e = thrust::make_transform_iterator( row_idx, [ pitch, ncols ] __device__ __host__ ( int r ) { return r * pitch + ncols; } );

//         // first pass: call measures workspace size
//         size_t tmp_sz = 0;
//         cub::DeviceSegmentedReduce::Sum( nullptr, tmp_sz, data(), res.data(), nrows, off_b, off_e );

//         // second run the reduction
//         thrust::device_vector<char> tmp( tmp_sz );
//         cub::DeviceSegmentedReduce::Sum( static_cast<void *>( thrust::raw_pointer_cast( tmp.data() ) ), tmp_sz, data(), res.data(), nrows, off_b, off_e );

//         return res;
//     }
//     #endif

//     // cpu
//     TODO;
//     return res;
// }

// #ifdef __CUDACC__
// // Functor at namespace scope so CUDA/cudafe can properly mangle the type in Thrust typedefs
// // (local structs with __device__ members cause "template argument N is invalid" in cudafe)
// template<class T, int ct_rank>
// struct TensorViewIndexer {
//     __device__ const T &operator()( sdot::PI i ) const {
//         sdot::SI off = 0;
//         for ( int d = ct_rank - 1; d >= 0; --d ) {
//             off += sdot::SI( i % ext[ d ] ) * str[ d ];
//             i /= ext[ d ];
//         }
//         return *reinterpret_cast<const T *>( src + off );
//     }

//     const std::byte *src;
//     sdot::PI ext[ ct_rank ];
//     sdot::SI str[ ct_rank ];
// };

// UTP void DTP::apply_cpu_version( auto &&func ) const {
//     TensorViewIndexer<T,ct_rank> es;
//     es.src = reinterpret_cast<const std::byte *>( data() );

//     PI size = 1;
//     for( int d = 0; d < ct_rank; ++d ) {
//         es.str[ d ] = stride( d );
//         es.ext[ d ] = shape( d );
//         size *= es.ext[ d ];
//     }

//     // gather strided GPU elements into a flat contiguous GPU buffer
//     using NT = std::remove_const_t<T>;
//     thrust::device_vector<NT> dev( size );
//     thrust::transform( thrust::counting_iterator<sdot::PI>( 0 ), thrust::counting_iterator<sdot::PI>( size ), dev.begin(), es );

//     // copy to host and print via CPU TensorView (contiguous)
//     std::vector<NT> host( size );
//     thrust::copy( dev.begin(), dev.end(), host.begin() );

//     sdot::TensorView<NT,ct_rank,sdot::Cpu> tensor( host.data(), shape() );
//     return func( tensor );
// }
// #endif

#undef UTP
#undef DTP

} // namespace sdot
