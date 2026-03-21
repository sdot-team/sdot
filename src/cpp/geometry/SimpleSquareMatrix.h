#pragma once

#include "../support/ASSERT.h"
#include "Point.h"

namespace sdot {

// _size
template<class T,int ct_size=-1>
class SimpleSquareMatrix {
public:
    using       Data                    = Point<T,(ct_size>=0?ct_size*ct_size:-1)>;
    using       Vec                     = Point<T,ct_size>;

    /**/        SimpleSquareMatrix      ( PI size ) : _size( size ), data( size * size ) { if ( ct_size >= 0 ) ASSERT( size == ct_size ); }

    const T&    operator()              ( PI r, PI c ) const { return data[ r * size() + c ]; }
    T&          operator()              ( PI r, PI c ) { return data[ r * size() + c ]; }

    auto        without_row_and_col     ( PI r, PI c ) const -> SimpleSquareMatrix<T,(ct_size>0?ct_size-1:-1)>;
    auto        with_replaced_col       ( PI c, const Vec &col ) const -> SimpleSquareMatrix;
    T           determinant             () const;
    Vec         solve                   ( const Vec &vec ) const;
    PI          size                    () const { return ct_size >= 0 ? ct_size : _size; }

    auto        begin                   () const { return data.begin(); }
    auto        begin                   () { return data.begin(); }
    auto        end                     () const { return data.end(); }
    auto        end                     () { return data.end(); }

    PI          _size;
    Data        data;
};

/* T_Td void operator+=( SimpleSquareMatrix<T,d> &a, const SimpleSquareMatrix<T,d> &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] += b[ i ]; }
T_Td void operator-=( SimpleSquareMatrix<T,d> &a, const SimpleSquareMatrix<T,d> &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] -= b[ i ]; }

T_Td SimpleSquareMatrix<T,d> operator+( const SimpleSquareMatrix<T,d> &a, const SimpleSquareMatrix<T,d> &b ) { SimpleSquareMatrix<T,d> res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] + b[ i ]; return res; }
T_Td SimpleSquareMatrix<T,d> operator-( const SimpleSquareMatrix<T,d> &a, const SimpleSquareMatrix<T,d> &b ) { SimpleSquareMatrix<T,d> res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] - b[ i ]; return res; }

T_Td SimpleSquareMatrix<T,d> operator*( const T &a, const SimpleSquareMatrix<T,d> &b ) { SimpleSquareMatrix<T,d> res( b.size() ); for( PI i = 0; i < b.size(); ++i ) res[ i ] = a * b[ i ]; return res; }
T_Td SimpleSquareMatrix<T,d> operator/( const SimpleSquareMatrix<T,d> &a, const T &b ) { SimpleSquareMatrix<T,d> res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] / b; return res; }

T_Td SimpleSquareMatrix<T,d> operator-( const SimpleSquareMatrix<T,d> &a ) { SimpleSquareMatrix<T,d> res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = - a[ i ]; return res; }
 */
} // namespace sdot

T_Td std::ostream &operator<<( std::ostream &os, const sdot::SimpleSquareMatrix<T,d> &p );

#include "SimpleSquareMatrix.cxx"
