#pragma once

#include "../common_macros.h"

namespace sdot {

//
HD auto apply_values( auto &&value, auto &&func ) requires requires { value.apply_values( FORWARD( func ) ); } {
    return value.apply_values( FORWARD( func ) );
}

} // namespace sdot
