#pragma once

#include "../common_macros.h"
#include "../common_types.h" // PI
#include "../Ct.h" // PI

namespace sdot {

template<class TI,class TC=PI>
struct Range {
    T_U constexpr void for_each_item_split( PI rel, PI mod, U &&func ) const { for( TC i = rel; i < TC( end ); i += mod ) func( TI( i ) ); }
    T_U constexpr void for_each_item      ( U &&func ) const { for( TC i = 0; i < TC( end ); ++i ) func( i ); }
    constexpr TI   nb_items           () const { return end; }
    T_U constexpr TC   item_at            ( U index ) const { return index; } ///< the index-th item (a Range yields its own index)

    TI             end;               ///<
};

T_T constexpr auto range( T &&end ) {
    return Range<DECAYED_TYPE_OF( end )>{ end };
}

template<class TI,class EC>
HD auto transfer_cost( const EC &/* execution_context */, const Range<TI> &/* arg */ ) {
    return 0_c;
}

} // namespace sdot
