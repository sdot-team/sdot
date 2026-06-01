#pragma once

#include "ZeroTensor.h"
#include "nb_items.h"

#define UTP template<class TF, class Shape>
#define DTP ZeroTensor<TF, Shape>

namespace sdot {

UTP HD DTP::ZeroTensor( Shape shape ) : _shape( shape ) {
}

UTP HD auto DTP::nb_items() const {
    return product( _shape );
}

UTP HD auto DTP::operator()( auto index, auto ...rem ) const {
    if constexpr ( ct_rank == 0 ) {
        return *this;
    } else if constexpr ( requires { DECAYED_TYPE_OF( index.size() )::value; } ) {
        // tuple/array index: unpack front element and recurse
        if constexpr ( DECAYED_TYPE_OF( index.size() )::value )
            return operator()( index[ Ct<int,0>() ], index.without_index( Ct<int,0>() ), rem... );
        else
            return operator()( rem... );
    } else {
        // scalar index: peel one dimension (value irrelevant for a zero tensor)
        auto sub_shape = _shape.without_index( Ct<int,0>() );
        return ZeroTensor<TF, DECAYED_TYPE_OF( sub_shape )>( sub_shape )( rem... );
    }
}

#undef UTP
#undef DTP

} // namespace sdot
