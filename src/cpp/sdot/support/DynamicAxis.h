#pragma once

#include "common_types.h"
#include <stdexcept>

namespace sdot {

struct DynamicAxis {
    /**/         DynamicAxis( PI *ptr, PI size, PI capacity ) : capacity( capacity ), ptr( ptr ) { *ptr = size; }

    operator     PI         () const { return *ptr; }

    PI           operator++ () { if ( ++( *ptr ) > capacity ) overflow(); return *ptr; }
    PI           operator++ ( int ) { PI res = *ptr; ++( *this ); return res; }

    DynamicAxis& operator=  ( PI value ) { if ( value > capacity ) overflow(); *ptr = value; return *this; }

    void         overflow   () { throw std::runtime_error( "DynamicAxis: capacity exceeded" ); }

    PI           capacity;
    PI*          ptr;
};

} // namespace sdot
