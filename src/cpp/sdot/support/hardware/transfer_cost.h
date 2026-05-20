#pragma once

#include "for_each_memory_space_of.h"

namespace sdot {

// 2 args version that calls a 3 args version (execution_space, memory_space and function to get nb_bytes if requires)
double transfer_cost( const auto &execution_space, const auto &value ) {
    double res = 0;
    for_each_memory_space_of( value, [&]( auto memory_space, auto &&get_nb_bytes ) {
        res += transfer_cost( execution_space, memory_space, get_nb_bytes );
    } );
    return res;
}

} // namespace sdot
