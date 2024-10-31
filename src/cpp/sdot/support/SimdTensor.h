#pragma once

#include <tl/support/containers/Vec.h>
#include <tl/support/operators/ceil.h>
#include <asimd/SimdVec.h>

namespace sdot {

/**
*/
template<class T,int nb_dims>
class SimdTensor {
public:
    using               SimdVec     = asimd::SimdVec<T>;
    using               Point       = Vec<T,nb_dims>;
 
    static constexpr PI simd_size   = SimdVec::size();
 
    /**/                SimdTensor  ();
 
    SimdTensor&         operator=   ( const SimdTensor &that );
 
    PI                  capacity    () const { return _data.size() / nb_dims; }
    PI                  size        () const { return _size; }
 
    PI                  offset      ( PI n, PI d = 0 ) const { return ( n / simd_size ) * simd_size * nb_dims + ( n % simd_size ) + simd_size * d; }
 
    const T&            operator()  ( PI n, PI d = 0 ) const { return _data[ offset( n, d ) ]; }
    T&                  operator()  ( PI n, PI d = 0 ) { return _data[ offset( n, d ) ]; }
 
    const Point         operator[]  ( PI n ) const { Point res; for( PI d = 0; d < nb_dims; ++d ) res[ d ] = operator()( n, d ); return res; }

    T_i Vec<T,i>        nd_at       ( PI n, CtInt<i> ) const { Vec<T,i> res; for( PI d = 0; d < i; ++d ) res[ d ] = operator()( n, d ); return res; }
 
    const T*            data        () const { return _data.data(); }
    T*                  data        () { return _data.data(); }
 
    T_i SimdTensor&     operator<<  ( const Vec<T,i> &p );

    Point               pop_back_val() { return operator[]( --_size ); }
    T_i void            set_item    ( PI index, const Vec<T,i> &p );
    void                reserve     ( PI capa );
    void                resize      ( PI size );
    void                clear       ();

private:
    PI                  _size;
    Vec<T>              _data;     ///< ex for simd_size == 4: x0 x1 x2 x3 y0 y1 y2 y3 x4 x5 ...
};

#define DTP template<class T,int nb_dims>
#define UTP SimdTensor<T,nb_dims>

DTP UTP::SimdTensor() : _size( 0 ) {
    reserve( 128 );
}

DTP UTP &UTP::operator=( const SimdTensor &that ) {
    resize( that.size() );
    for( PI i = 0; i < that.size(); ++i )
        set_item( i, that[ i ] );
    return *this;
}

DTP T_i UTP &UTP::operator<<( const Vec<T,i> &p ) {
    const PI o = _size++;
    reserve( _size );
    set_item( o, p );
    return *this;
}

DTP T_i void UTP::set_item( PI index, const Vec<T,i> &p ) {
    for( PI d = 0; d < i; ++d )
        operator()( index, d ) = p[ d ];
}

DTP void UTP::reserve( PI capa ) {
    const PI wsize = ceil( capa, simd_size ) * nb_dims;
    if ( wsize > _data.size() )
        _data.aligned_resize( wsize, CtInt<simd_size * sizeof( T )>() );
}

DTP void UTP::resize( PI size ) {
    reserve( size );
    _size = size;
}

DTP void UTP::clear() {
    _size = 0;
}

#undef DTP
#undef UTP

} // namespace sdot
