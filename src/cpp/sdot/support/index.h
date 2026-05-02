#pragma once

#include "common_types.h"

namespace sdot {

SI index( const auto &inputs, auto &&func ) {
    for( SI i = 0; i < inputs.size(); ++i )
        if ( func( inputs[ i ] ) )
            return i;
    return -1;
}

SI index( auto &&func ) {
    for( SI i = 0; ; ++i )
        if ( func( i ) )
            return i;
}

} // namespace sdot
