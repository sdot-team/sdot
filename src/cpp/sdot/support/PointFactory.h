#pragma once

#include "Point.h"

namespace sdot {

//
template<class T,int ct_dim,class Arch>
class PointFactory {
public:
    using PT          = Point<T,ct_dim,Arch>;

    /**/  PointFactory( PI /*size*/ ) {}

    PT    operator()  ( auto ...args ) const { PI cpt = 0; PT res; ( ( res[ cpt++ ] = args ), ... ); for( ; cpt < dim(); ++cpt ) res[ cpt ] = 0; return res; }
    PT    with_func   ( auto &&func ) const { PT res; for( PI i = 0; i < dim(); ++i ) res[ i ] = func( i ); return res; }
    PT    value_at    ( PI index, T value ) const { PT res = zeros(); res[ index ] = value; return res; }
    PT    zeros       () const { PT res; for( PI i = 0; i < dim(); ++i ) res[ i ] = 0; return res; }
    PT    ones        () const { PT res; for( PI i = 0; i < dim(); ++i ) res[ i ] = 1; return res; }

    PI    dim         () const { return ct_dim; }
};

//
template<class T,class Arch>
class PointFactory<T,-1,Arch> {
public:
    using PT          = Point<T,-1,Arch>;

    /**/  PointFactory( PI size ) : _dim( size ) {}

    PT    operator()  ( auto ...args ) const { PT res( dim() ); PI cpt = 0; ( ( res[ cpt++ ] = args ), ... ); for( ; cpt < dim(); ++cpt ) res[ cpt ] = 0; return res; }
    PT    with_func   ( auto &&func ) const { PT res( dim() ); for( PI i = 0; i < dim(); ++i ) res[ i ] = func( i ); return res; }
    PT    value_at    ( PI index, T value ) const { PT res = zeros(); res[ index ] = value; return res; }
    PT    zeros       () const { PT res( dim() ); for( PI cpt = 0; cpt < dim(); ++cpt ) res[ cpt ] = 0; return res; }
    PT    ones        () const { PT res( dim() ); for( PI cpt = 0; cpt < dim(); ++cpt ) res[ cpt ] = 1; return res; }

    PI    dim         () const { return _dim; }

    PI    _dim;
};

} // namespace sdot

#include "PointFactory.cxx"
