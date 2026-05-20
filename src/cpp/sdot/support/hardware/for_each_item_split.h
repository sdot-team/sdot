#pragma once

#include "../common_types.h"
#include "for_each_item.h"

namespace sdot {

void for_each_item_split( auto &&list, PI rel, PI mod, auto &&func ) requires ( requires { list.for_each_item_split( FORWARD( func ) ); } || requires { for_each_item( FORWARD( list ), FORWARD( func ) ); } ) {
    if constexpr ( requires { list.for_each_item_split( FORWARD( func ) ); } ) {
        list.for_each_item_split( FORWARD( func ) );
    } else {
        PI cpt = 0;
        for_each_item( FORWARD( list ), [&]( auto &&item ) {
            if ( cpt++ % mod == rel )
                func( FORWARD( item ) );
        } );
    }
}

} // namespace sdot

