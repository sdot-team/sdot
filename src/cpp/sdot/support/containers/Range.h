#pragma once

#include "../common_macros.h"
#include "../common_types.h" // PI
#include "../Ct.h" // PI

namespace sdot {

template<class TI,class TC=PI>
struct Range {
    constexpr void for_each_item_split( PI rel, PI mod, auto &&func ) const { for( TC i = rel; i < TC( end ); i += mod ) func( TI( i ) ); }
    constexpr void for_each_item      ( auto &&func ) const { for( TC i = 0; i < TC( end ); ++i ) func( i ); }
    constexpr TI   nb_items           () const { return end; }
    constexpr TC   item_at            ( auto index ) const { return index; } ///< the index-th item (a Range yields its own index)

    TI             end;               ///<
};

constexpr auto range( auto &&end ) {
    return Range<DECAYED_TYPE_OF( end )>{ end };
}

template<class TI>
HD auto transfer_cost( const auto &/* execution_context */, const Range<TI> &/* arg */ ) {
    return 0_c;
}

} // namespace sdot
