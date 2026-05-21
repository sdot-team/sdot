#pragma once

#include "TypePromote.h"
#include "Tuple.h"
// #include <utility>

namespace sdot {

// ---- runtime front value -----------------------------------------------------------
#define UTP template<class Head,class... Tail>
#define DTP Tuple<Head,Tail...>

UTP HD DTP::Tuple( Function, auto &&func, auto index ) : head( func( index ) ), tail( Function(), FORWARD( func ), index + Ct<int,1>() ) {
}

UTP HD DTP::Tuple( Function, auto &&func ) : Tuple( Function(), func, Ct<int,0>() ) {
}

UTP HD DTP::Tuple( Values, auto head, auto... tail ) : head( head ), tail( Values(), tail... ) {
}

UTP HD auto DTP::apply_values( auto &&cb ) const {
    return tail.apply_values( [&]( auto ...tail ) {
        return cb( head, tail... );
    } );
}

UTP HD auto DTP::operator[]( auto &&index ) const {
    if constexpr ( requires { DECAYED_TYPE_OF( index )::value; } ) {
        if ( DECAYED_TYPE_OF( index )::value )
            return tail[ index - Ct<int,1>() ];
        else
            return head;
    } else {
        using TR = TypePromote<Head,Tail...>::type;
        if ( index )
            return TR( tail[ index - Ct<int,1>() ] );
        else
            return TR( head );
    }
}

UTP HD auto DTP::size() const {
    return Ct<int,1 + sizeof...( Tail )>();
}

UTP HD auto DTP::without_index( auto index ) const {
    if constexpr ( requires { DECAYED_TYPE_OF( index )::value; } ) {
        if constexpr ( DECAYED_TYPE_OF( index )::value )
            return tail;
        else
            return concat( tuple( head ), tail.without_index( index - Ct<int,1>() ) );
    } else {
        using TR = TypePromote<Head,Tail...>::type;
        return apply_values( [&]( auto ...vals ) {
            TR rt_vals[] = { vals... };
            return Tuple<std::conditional_t<1,TR,Tail>...>( Function(), [&]( auto i ) {
                return rt_vals[ i + ( i >= index ) ];
            } );
        } );
    }
}

#undef UTP
#undef DTP

// ---- rank 0 ------------------------------------------------------------------------
#define UTP // template<>
#define DTP Tuple<>

UTP HD DTP::Tuple( Function, auto &&/*func*/, auto /*index*/ ) {
}

UTP HD DTP::Tuple( Function, auto &&/*func*/ ) {
}

UTP HD DTP::Tuple( const Tuple &/* that */ ) {
}

UTP HD DTP::Tuple( Values ) {
}

UTP HD DTP::Tuple() {

}

UTP HD void DTP::for_each_item( auto &&/* cb */ ) const {
}

UTP HD auto DTP::apply_values( auto &&cb ) const {
    return cb();
}

UTP HD Void DTP::operator[]( auto ) const {
    return {};
}

UTP HD auto DTP::size() const {
    return Ct<int,0>();
}

#undef UTP
#undef DTP

} // namespace sdot
