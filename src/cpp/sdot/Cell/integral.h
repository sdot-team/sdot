#pragma once

#include "../support/common_macros.h"

namespace sdot {

template<class Domain,class Value>
struct Integral;


auto integral( auto &&value, auto &&domain ) {
    using Domain = DECAYED_TYPE_OF( domain );
    using Value = DECAYED_TYPE_OF( value );
    return Integral<Value,Domain>::integral( value, domain );
}

} // namespace sdot
