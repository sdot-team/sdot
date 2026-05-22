#pragma once

#include "../common_macros.h"

namespace sdot {

template<class TI>
struct Range {
    void for_each_item( auto &&func ) const { for( TI i = 0; i < end; ++i ) func( i ); }
    TI   nb_items     () const { return end; }

    TI   end;         ///<
};

auto range( auto &&end ) {
    return Range<DECAYED_TYPE_OF( end )>{ end };
}

} // namespace sdot
