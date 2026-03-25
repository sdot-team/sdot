#pragma once

#include "TensorView.h"
#include "TODO.h"

namespace sdot {

#define UTP template<class T,int ct_rank,class Arch>
#define DTP TensorView<T,ct_rank,Arch>

UTP auto DTP::squeeze( PI axis ) const -> TensorView<T,ct_rank-1,Arch> {
    std::array<SI,ct_rank-1> nstrides;
    std::array<PI,ct_rank-1> nextent;
    for( PI i = 0; i < ct_rank-1; ++i ) {
        nstrides[ i ] = _strides[ i + ( i >= axis ) ];
        nextent[ i ] = _shape[ i + ( i >= axis ) ];
    }
    return { reinterpret_cast<T *>( _ptr ), nextent, nstrides };
}

UTP auto DTP::row( PI index ) const -> TensorView<T,ct_rank-1,Arch> {
    std::array<SI,ct_rank-1> nstrides;
    std::array<PI,ct_rank-1> nextent;
    for( PI i = 1; i < ct_rank; ++i ) {
        nstrides[ i - 1 ] = _strides[ i ];
        nextent[ i - 1 ] = _shape[ i ];
    }
    return { reinterpret_cast<T *>( _ptr + index * _strides[ 0 ] ), nextent, nstrides };
}

UTP auto DTP::contiguous_strides( const Shape &ext ) -> Strides {
    Strides s;
    s[ ct_rank - 1 ] = sizeof( T );
    for( int i = ct_rank - 2; i >= 0; --i )
        s[ i ] = s[ i + 1 ] * ext[ i + 1 ];
    return s;
}

UTP template<class TF>
Tensor<TF,1,Arch> DTP::sum_along_axis_1() const {
    Tensor<TF,1,Arch> res( sdot::Shape(), { _shape[ 0 ] } );

    // cuda
    #ifdef __CUDACC__
    if constexpr ( std::is_same_v<Arch,Cuda> ) {
        static_assert( std::is_same_v<std::decay_t<TF>,std::decay_t<T>>, "for now we support only homogeneous types for CUDA" );
        static_assert( ct_rank == 2, "CUDA sum_axis_0 only supports rank-2 for now" );

        // columns within a row must be contiguous
        ASSERT( _strides[ 1 ] == (SI)sizeof( T ) );

        const int pitch( _strides[ 0 ] / sizeof( T ) );  // row stride in elements (>= ncols if padding)
        const int nrows( _shape[ 0 ] );
        const int ncols( _shape[ 1 ] );

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
    p.with_cpu_version( [&]( auto t ) {
        os << t;
    } );
    return os;
}
#endif

