#pragma once

// #include "../support/ASSERT.h"
#include "Point.h"

namespace sdot {

//
template<class T,int _dim=-1>
class PointFactory {
public:
    using PT          = Point<T,_dim>;

    /**/  PointFactory( PI /*size*/ ) {}

    PT    operator()  ( auto ...args ) const { PI cpt = 0; PT res; ( ( res[ cpt++ ] = args ), ... ); for( ; cpt < dim(); ++cpt ) res[ cpt ] = 0; return res; }
    PT    with_func   ( auto &&func ) const { PT res; for( PI i = 0; i < dim(); ++i ) res[ i ] = func( i ); return res; }
    PT    value_at    ( PI index, T value ) const { PT res = zeros(); res[ index ] = value; return res; }
    PT    zeros       () const { PT res; for( PI i = 0; i < dim(); ++i ) res[ i ] = 0; return res; }

    PI    dim         () const { return _dim; }
};

//
template<class T>
class PointFactory<T,-1> {
public:
    using PT          = Point<T,-1>;

    /**/  PointFactory( PI size ) : _dim( size ) {}

    PT    operator()  ( auto ...args ) const { PT res( dim() ); PI cpt = 0; ( ( res[ cpt++ ] = args ), ... ); for( ; cpt < dim(); ++cpt ) res[ cpt ] = 0; return res; }
    PT    with_func   ( auto &&func ) const { PT res; for( PI i = 0; i < dim(); ++i ) res[ i ] = func( i ); return res; }
    PT    value_at    ( PI index, T value ) const { PT res = zeros(); res[ index ] = value; return res; }
    PT    zeros       () const { PT res( dim() ); for( PI cpt = 0; cpt < dim(); ++cpt ) res[ cpt ] = 0; return res; }

    PI    dim         () const { return _dim; }

    PI    _dim;
};

} // namespace sdot

#include "PointFactory.cxx"
