#pragma once

#include "Vector.h"

namespace sdot {

template<class T,int ct_dim,class Arch=Cpu>
class VectorFactory {
public:
    using PT = Vector<T,ct_dim,Arch>;

    /**/  VectorFactory ( PI dim );

    PT    operator()   ( auto ...args ) const { PT res( Reserved(), _dim ); PI cpt = 0; ( ( new ( res.data() + cpt++ ) T( args ) ), ... ); for ( ; cpt < _dim; ++cpt ) new ( res.data() + cpt ) T(); return { Values(),  }; }
    PT    with_func    ( auto &&func ) const { return PT::with_func( _dim, FORWARD( func ) ); }
    PT    value_at     ( PI index, T value ) const;
    PT    zeros        () const;
    PT    ones         () const;

    PI    dim          () const;

    PI    _dim;
};

} // namespace sdot

#include "VectorFactory.cxx"
