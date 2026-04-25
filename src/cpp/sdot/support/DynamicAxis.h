#pragma once

#include "common_types.h"
#include <stdexcept>

namespace sdot {

struct DynamicAxis {
    SI*  ptr;
    SI   cap;

    static DynamicAxis input( SI* ptr, SI size ) { *ptr = size; return { ptr, size }; }

    operator SI () const { return *ptr; }

    SI& operator++ () {
        if ( ++( *ptr ) > cap )
            throw std::runtime_error( "DynamicAxis: capacity exceeded" );
        return *ptr;
    }

    SI operator++ ( int ) {
        SI res = *ptr;
        ++( *this );
        return res;
    }
};

} // namespace sdot
