#pragma once

#include "Affine1d.h"
#include <stdexcept>
#include <utility>

namespace sdot {

    // --------------------------------------------------- batch dim 0 ---------------------------------------------------
#define UTP template<class T,class Arch>
#define DTP Affine1d<T,Arch>

UTP DTP::Piece DTP::get_first_piece( TF point_scale ) const {
    #ifndef __CUDACC__
    if ( nb_points() < 2 )
        throw std::runtime_error( "For affine function, we need at least 2 points" );
    #endif

    const TF y0 = point_scale * static_cast<TF>( ys[ 0 ] );
    const TF y1 = point_scale * static_cast<TF>( ys[ 1 ] );
    const TF x0 = static_cast<TF>( xs[ 0 ] );
    const TF x1 = static_cast<TF>( xs[ 1 ] );

    const TF mass = ( x1 - x0 ) * ( y1 + y0 ) / 2.0;

    return {
        .index = 1,
        .mass  = mass,
        .x0    = x0,
        .x1    = x1,
        .y0    = y0,
        .y1    = y1,
    };
}

UTP void DTP::get_next_piece( Piece &piece, TF point_scale ) const {
    if ( ++piece.index < nb_points() ) {
        piece.y0 = std::exchange( piece.y1, point_scale * static_cast<TF>( ys[ piece.index ] ) );
        piece.x0 = std::exchange( piece.x1, static_cast<TF>( xs[ piece.index ] ) );
        piece.mass = ( piece.x1 - piece.x0 ) * ( piece.y1 + piece.y0 ) / 2;
        return;
    }

    piece.x0 = piece.x1;
    piece.y0 = piece.y1;
    piece.mass = std::numeric_limits<TF>::max();
}

UTP DTP::TF DTP::take_some_mass( Piece &current_piece, TF point_scale, TF mass_to_take, auto &&on_taken_piece ) const {
    // enough mass in the current piece ?
    if ( mass_to_take <= current_piece.mass ) {
        Piece np = current_piece.take_some_mass( mass_to_take );
        on_taken_piece( np );
        return np.x1;
    }

    // else, use the current piece, get to the next, ...
    mass_to_take -= current_piece.mass;
    on_taken_piece( current_piece );

    while ( true ) {
        get_next_piece( current_piece, point_scale );
        if ( mass_to_take <= current_piece.mass )
           break;

        // full interval
        mass_to_take -= current_piece.mass;
        on_taken_piece( current_piece );
    }

    Piece np = current_piece.take_some_mass( mass_to_take );
    on_taken_piece( np );
    return np.x1;
}

UTP PI DTP::nb_points() const {
    return xs.size();
}

UTP DTP::TF DTP::mass() const {
    TF px0 = static_cast<TF>( xs[ 0 ] );
    TF py0 = static_cast<TF>( ys[ 0 ] );
    TF res = 0;
    for( PI j = 1; j < nb_points(); ++j ) {
        const TF px1 = static_cast<TF>( xs[ j ] );
        const TF py1 = static_cast<TF>( ys[ j ] );
        res += ( py0 + py1 ) * ( px1 - px0 );
        px0 = px1;
        py0 = py1;
    }
    return res / 2;
}

UTP void DTP::get_grad_ys( T ratio, auto grad_y ) const {
    using namespace std;
    for( int j = 0, n = nb_points() - 1; j <= n; ++j ) {
        T x0 = xs[ max( j - 1, 0 ) ];
        T x2 = xs[ min( j + 1, n ) ];
        grad_y[ j ] -= ratio / 2 * ( x2 - x0 );
    }
}

UTP void DTP::accumulate_w2_grad_ys( const Piece &piece, TF dirac_pos, TF potential, TF g_scale, auto grad_ys ) const {
    TF gl = 0, gr = 0;
    piece.integrate_w2_shape_functions( dirac_pos, potential,
        static_cast<TF>( xs[ piece.index - 1 ] ), static_cast<TF>( xs[ piece.index ] ), gl, gr );
    grad_ys[ piece.index - 1 ] += static_cast<T>( g_scale * gl );
    grad_ys[ piece.index ]     += static_cast<T>( g_scale * gr );
}

UTP void DTP::accumulate_linear_grad_ys( const Piece &piece, TF slope, TF offset, TF scale, auto grad_ys ) const {
    TF gl = 0, gr = 0;
    piece.integrate_linear_shape_functions( slope, offset,
        static_cast<TF>( xs[ piece.index - 1 ] ), static_cast<TF>( xs[ piece.index ] ), gl, gr );
    grad_ys[ piece.index - 1 ] += static_cast<T>( scale * gl );
    grad_ys[ piece.index ]     += static_cast<T>( scale * gr );
}

#undef UTP
#undef DTP

// --------------------------------------------------- batch dim 1 ---------------------------------------------------
#define UTP template<class T,class Arch>
#define DTP BatchOfAffine1d<T,Arch>

UTP PI DTP::nb_points() const {
    return xs.size( 1 );
}

UTP PI DTP::nb_rows() const {
    return xs.size( 0 );
}

UTP Affine1d<T,Arch> DTP::row( PI num_batch ) const  {
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
