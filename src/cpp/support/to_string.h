#pragma once

#include <sstream>

namespace sdot {

std::string to_string( const auto &v ) {
    std::ostringstream ss;
    ss << v;
    return ss.str();
}

} // namespace sdot
