#pragma once

#include "AxisValues.h"
#include <utility>

namespace sdot {

// C-contiguous (row-major) byte strides for a given shape AxisValues, for an element type TF.
// stride[last] = sizeof(TF) ; stride[i] = stride[i+1] * shape[i+1].

auto _contiguous_strides_impl( auto head, auto... tail ) {
    using Strides = AxisValues<typename Shape::TI,Shape::ct_rank>;
    if constexpr ( Shape::ct_rank == 0 ) {
        return Strides( Values() );
    } else {
        SI s[ Shape::ct_rank ];
        s[ Shape::ct_rank - 1 ] = sizeof( TF );
        for ( int i = Shape::ct_rank - 2; i >= 0; --i )
            s[ i ] = s[ i + 1 ] * SI( shape[ i + 1 ] );
        return Strides( Values(), SI( s[ Is ] )... );
    }
}

template<class TF, class Shape>
auto contiguous_strides( const Shape &shape ) {
    return shape.apply_values( [&]( auto ...values ) {
        return _contiguous_strides_impl();
    } );
}

} // namespace sdot
