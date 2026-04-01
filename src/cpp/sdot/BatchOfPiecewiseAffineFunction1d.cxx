#pragma once

#include "BatchOfPiecewiseAffineGrid1d.h"

namespace sdot {

#define UTP template<class T,class Arch>
#define DTP BatchOfPiecewiseAffineGrid1d<T,Arch>

UTP PI DTP::nb_points() const {
    return xs.size( 1 );
}

UTP PI DTP::nb_rows() const {
    return xs.size( 0 );
}

UTP PiecewiseAffineGrid1d<T,Arch> DTP::row( PI num_batch ) const  {
   return { xs.row( num_batch ), ys.row( num_batch ) };
}

#ifdef __CUDACC__
template<class T,class TF>
struct BatchAffine1dMassFunctor {
    __device__ TF operator()( int idx ) const {
        const int r = idx / n_pairs;
        const int c = idx % n_pairs;
        const TF x0 = static_cast<TF>( *reinterpret_cast<const T*>( xs_ptr + r * xs_s0 + ( c + 0 ) * xs_s1 ) );
        const TF x1 = static_cast<TF>( *reinterpret_cast<const T*>( xs_ptr + r * xs_s0 + ( c + 1 ) * xs_s1 ) );
        const TF y0 = static_cast<TF>( *reinterpret_cast<const T*>( ys_ptr + r * ys_s0 + ( c + 0 ) * ys_s1 ) );
        const TF y1 = static_cast<TF>( *reinterpret_cast<const T*>( ys_ptr + r * ys_s0 + ( c + 1 ) * ys_s1 ) );
        return ( x1 - x0 ) * ( y1 + y0 ) / 2;
    }

    const std::byte* xs_ptr;
    const std::byte* ys_ptr;
    SI xs_s0, xs_s1;
    SI ys_s0, ys_s1;
    int n_pairs;
};
#endif // __CUDACC__

UTP auto DTP::masses() const -> Tensor<TF,1,Arch> {
    const int n_pairs = xs.size( 1 ) - 1;
    const int nrows = xs.size( 0 );

    Tensor<TF,1,Arch> res( sdot::Shape(), { PI( nrows ) } );

    #ifdef __CUDACC__
    if constexpr ( std::is_same_v<Arch,Cuda> ) {
        static_assert( std::is_same_v<std::decay_t<TF>,std::decay_t<T>>, "for now we support only homogeneous types for CUDA" );

        BatchAffine1dMassFunctor<T,TF> functor{
            reinterpret_cast<const std::byte*>( xs.data() ),
            reinterpret_cast<const std::byte*>( ys.data() ),
            xs.stride( 0 ), xs.stride( 1 ),
            ys.stride( 0 ), ys.stride( 1 ),
            n_pairs
        };
        auto trans_iter = thrust::make_transform_iterator( thrust::counting_iterator<int>( 0 ), functor );

        auto row_idx = thrust::counting_iterator<int>( 0 );
        auto off_b = thrust::make_transform_iterator( row_idx, [n_pairs] __device__ __host__ ( int r ) { return r * n_pairs; } );
        auto off_e = thrust::make_transform_iterator( row_idx, [n_pairs] __device__ __host__ ( int r ) { return r * n_pairs + n_pairs; } );

        size_t tmp_sz = 0;
        cub::DeviceSegmentedReduce::Sum( nullptr, tmp_sz, trans_iter, res.data(), nrows, off_b, off_e );

        thrust::device_vector<char> tmp( tmp_sz );
        cub::DeviceSegmentedReduce::Sum( static_cast<void *>( thrust::raw_pointer_cast( tmp.data() ) ), tmp_sz, trans_iter, res.data(), nrows, off_b, off_e );

        return res;
    }
    #endif

    // cpu
    const PI ncols = xs.size( 1 );
    for ( PI r = 0; r < (PI)nrows; ++r ) {
        TF sum = 0;
        for ( PI c = 0; c + 1 < ncols; ++c )
            sum += ( static_cast<TF>( xs( r, c + 1 ) ) - static_cast<TF>( xs( r, c ) ) )
                 * ( static_cast<TF>( ys( r, c + 1 ) ) + static_cast<TF>( ys( r, c ) ) );
        res( r ) = sum / 2;
    }
    return res;
}

#undef UTP
#undef DTP

} // namespace sdot
