#pragma once

#include "../common_macros.h"
#include "../common_types.h" // PI

namespace sdot {

template<class TI>
struct Range {
    constexpr void for_each_item      ( auto &&func ) const { for( TI i = 0; i < end; ++i ) func( i ); }
    // strided distribution: thread `rel` of `mod` visits items rel, rel+mod, ... (linear order).
    // O( nb_items / mod ) per thread — no full scan (unlike the generic fallback in for_each_item_split.h).
    constexpr void for_each_item_split( PI rel, PI mod, auto &&func ) const { for( PI i = rel; i < PI( end ); i += mod ) func( TI( i ) ); }
    constexpr TI   item_at            ( PI index ) const { return TI( index ); } ///< the index-th item (a Range yields its own index)
    constexpr TI   nb_items           () const { return end; }

    TI             end;         ///<
};

constexpr auto range( auto &&end ) {
    return Range<DECAYED_TYPE_OF( end )>{ end };
}

} // namespace sdot
