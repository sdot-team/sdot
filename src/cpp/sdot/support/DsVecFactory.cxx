#pragma once

#include "DsVecFactory.h"

namespace sdot {

#define UTP template<class T,int ct_dim,class Arch>
#define DTP DsVecFactory<T,ct_dim,Arch>

UTP DTP::DsVecFactory( PI dim ) : _dim( dim ) {
    if constexpr ( ct_dim >= 0 )
        ASSERT( dim == ct_dim );
}

UTP typename DTP::PT DTP::zeros   () const { return PT::zeros   ( _dim ); }
UTP typename DTP::PT DTP::ones    () const { return PT::ones    ( _dim ); }
UTP typename DTP::PT DTP::value_at( PI index, T value ) const { return PT::value_at( _dim, index, value ); }

UTP PI DTP::dim() const { return ct_dim >= 0 ? PI( ct_dim ) : _dim; }

#undef UTP
#undef DTP

} // namespace sdot
