#pragma once

#include "../common_macros.h"

namespace sdot {

//
// SFINAE via trailing return: participates only when value.apply_values( func ) is valid
T_TA HD auto apply_values( T &&value, A &&func ) -> decltype( value.apply_values( FORWARD( func ) ) ) {
    return value.apply_values( FORWARD( func ) );
}

} // namespace sdot
