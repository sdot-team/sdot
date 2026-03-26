#pragma once

#include "Point.h"

namespace sdot {

#define UTP template<class T,int dim,class Arch>
#define DTP Point<T,dim,Arch>

UTP PI DTP::size() const {
    return content.size();
}

#undef UTP
#undef DTP

// ---------------------------------------------------------------------------
#define UTP template<class T,class Arch>
#define DTP Point<T,-1,Arch>

UTP PI DTP::size() const {
    return content.size();
}

#undef UTP
#undef DTP

} // namespace sdot

template<class T,int dim,class Arch>
std::ostream &operator<<( std::ostream &os, const sdot::Point<T,dim,Arch> &p ) {
    for( sdot::PI i = 0; i < p.size(); ++i )
        os << ( i ? ", " : "" ) << p[ i ];
    return os;
}

T_T std::ostream &operator<<( std::ostream &os, const std::vector<T> &p ) {
    for( sdot::PI i = 0; i < p.size(); ++i )
        os << ( i ? ", " : "" ) << p[ i ];
    return os;
}
