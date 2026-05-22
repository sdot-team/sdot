#pragma once

#include "../common_macros.h"

namespace sdot {

HD void make_accessible( const auto &execution_space, auto &&value, auto &&func ) {
    if constexpr ( requires { value.make_accessible( execution_space, FORWARD( func ) ); } )
        value.make_accessible( execution_space, FORWARD( func ) );
    else
        func( FORWARD( value ) );
}

} // namespace sdot


