#pragma once

#include "support/TODO.h"
#include "support/P.h"
#include "DiracSet.h"
#include <algorithm>

namespace sdot {

#define UTP template<class T,class Arch>
#define DTP DiracSet<T,Arch>

UTP PI DTP::nb_diracs() const {
    return xs.size();
}

UTP DTP::TF DTP::mass() const {
    TF res = 0;
    for( PI i = 0; i < nb_diracs(); ++i )
        res += static_cast<TF>( ws[ i ] );
    return res;
}

UTP auto DTP::arg_sort() -> std::vector<PI> const {
    std::vector<PI> res( nb_diracs() );

    for( PI i = 0; i < nb_diracs(); ++i )
        res[ i ] = i;

    std::sort( res.begin(), res.end(), [&]( PI index_a, PI index_b ) {
        return xs[ index_a ] < xs[ index_b ];
    });

    return res;
}

#undef UTP
#undef DTP

#define UTP template<class T,class Arch>
#define DTP BatchOfDiracSet<T,Arch>

UTP PI DTP::nb_diracs() const {
    return xs.size( 1 );
}

UTP PI DTP::nb_rows() const {
    return xs.size( 0 );
}

UTP DiracSet<T,Arch> DTP::row( PI num_batch ) const  {
   return { xs.row( num_batch ), ws.row( num_batch ) };
}

UTP auto DTP::masses() const -> Tensor<TF,1,Arch> {
    return ws.template sum_along_axis_1<TF>();
}

UTP void DTP::arg_sorts( Tensor<PI,2,Arch> &sorted_is, Tensor<std::remove_const_t<T>,2,Arch> &sorted_xs ) const {
    const PI nd = nb_diracs(), nr = nb_rows();

    // declaration of the outputs
    sorted_is = { sdot::Shape(), { nr, nd } };
    sorted_xs = { sdot::Shape(), { nr, nd } };

    // cuda
    #ifdef __CUDACC__
    if constexpr ( std::is_same_v<Arch,Cuda> ) {
        ASSERT( xs.shape( 1 ) == xs.stride( 0 ) / sizeof( T ) );

        // initial values: [0..nd-1, 0..nd-1, ...] — use int to match existing patterns
        Tensor<PI,2,Arch> is( sdot::Shape(), { nr, nd } );
        thrust::transform( thrust::cuda::par, thrust::counting_iterator<PI>( 0 ), thrust::counting_iterator<PI>( nr * nd ), is.data(), [nd] __device__ __host__ ( PI i ) {
            return i % nd;
        } );

        // segment offsets
        auto row_idx = thrust::counting_iterator<int>( 0 );
        auto off_b = thrust::make_transform_iterator( row_idx, [nd] __device__ __host__ ( PI r ) { return ( r + 0 ) * nd; } );
        auto off_e = thrust::make_transform_iterator( row_idx, [nd] __device__ __host__ ( PI r ) { return ( r + 1 ) * nd; } );

        // size query
        size_t tmp_sz = 0;
        cub::DeviceSegmentedSort::SortPairs( nullptr, tmp_sz, xs.data(), sorted_xs.data(), is.data(), sorted_is.data(), nr * nd, nr, off_b, off_e );

        // sort
        thrust::device_vector<char> tmp( tmp_sz );
        void *tmp_ptr = static_cast<void *>( thrust::raw_pointer_cast( tmp.data() ) );
        cub::DeviceSegmentedSort::SortPairs( tmp_ptr, tmp_sz, xs.data(), sorted_xs.data(), is.data(), sorted_is.data(), nr * nd, nr, off_b, off_e );

        return;
    }
    #endif

    // cpu: fill each row with 0..nd-1 then sort by xs value
    // for( PI r = 0; r < nr; ++r ) {
    //     PI *row_ptr = res.data() + r * nd;
    //     for( PI c = 0; c < nd; ++c )
    //         row_ptr[ c ] = c;
    //     std::sort( row_ptr, row_ptr + nd, [&]( PI a, PI b ) {
    //         return xs( r, a ) < xs( r, b );
    //     });
    // }
    TODO;
}

#undef UTP
#undef DTP

} // namespace sdot
