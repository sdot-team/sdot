#pragma once

#include "Point.h"

namespace sdot {

#define UTP template<class T,int dim>
#define DTP Point<T,dim>

UTP PI DTP::size() const {
    return data.size();
}

#undef UTP
#undef DTP

// --------------------------------------------------------------------------------------
#define UTP template<class T>
#define DTP Point<T,-1>

UTP PI DTP::size() const {
    return data.size();
}

#undef UTP
#undef DTP

} // namespace sdot

template<class T,int dim>
std::ostream &operator<<( std::ostream &os, const sdot::Point<T,dim> &p ) {
    for( sdot::PI i = 0; i < p.size(); ++i )
        os << ( i ? ", " : "" ) << p[ i ];
    return os;
}

T_T std::ostream &operator<<( std::ostream &os, const std::vector<T> &p ) {
    for( sdot::PI i = 0; i < p.size(); ++i )
        os << ( i ? ", " : "" ) << p[ i ];
    return os;
}
