#pragma once

#include "../display.h"
#include <sstream>

namespace sdot {

std::string to_string( const auto &v ) {
    std::ostringstream ss;
    display( ss, v );
    return ss.str();
}

} // namespace sdot
