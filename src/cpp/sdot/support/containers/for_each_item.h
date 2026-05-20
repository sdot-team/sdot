#pragma once

#include "../common_macros.h" // HD, FORWARD

namespace sdot {

HD void for_each_item( auto &&list, auto &&func ) requires requires { list.for_each_item( FORWARD( func ) ); } {
    list.for_each_item( FORWARD( func ) );
}

} // namespace sdot

