#pragma once

#include "common_macros.h"
#include <vector>

namespace sdot {

auto vector_map( const auto &inputs, auto &&func ) {
    using TR = DECAYED_TYPE_OF( func( *inputs.begin() ) );
    std::vector<TR> res;
    res.reserve( inputs.size() );
    for( const auto &inp : inputs )
        res.push_back( func( inp ) );
    return res;
}

} // namespace sdot
