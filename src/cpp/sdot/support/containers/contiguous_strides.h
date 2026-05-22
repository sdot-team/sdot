#pragma once

#include "Tuple.h"

namespace sdot {

namespace detail::contiguous_strides {
    auto _contiguous_strides_impl() {
        return tuple();
    }
    auto _contiguous_strides_impl( auto head, auto... tail ) {
        auto product = head * ( tail * ... * Ct<int,1>() );
        return concat( tuple( product ), _contiguous_strides_impl( tail... ) );
    }
}

// C-contiguous (row-major) byte strides for a given shape AxisValues, for an element type TF.
// stride[last] = sizeof(TF) ; stride[i] = stride[i+1] * shape[i+1].
template<class TF>
constexpr auto contiguous_strides( const auto &shape ) {
    if constexpr( DECAYED_TYPE_OF( shape.size() )::value ) {
        return shape.apply_values( [&]( auto _, auto ...values ) {
            return detail::contiguous_strides::_contiguous_strides_impl( values..., Ct<int,sizeof( TF )>() );
        } );
    } else
        return tuple();
}

} // namespace sdot
