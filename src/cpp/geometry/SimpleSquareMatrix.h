#pragma once

#include "../support/ASSERT.h"
#include "Point.h"

namespace sdot {

// _size
template<class T,int ct_size=-1>
class SimpleSquareMatrix {
public:
    using       value_type              = T;
    using       Content                    = Point<T,(ct_size>=0?ct_size*ct_size:-1)>;
    using       Vec                     = Point<T,ct_size>;

    /**/        SimpleSquareMatrix      ( PI size ) : content( size * size ), _size( size ) { if ( ct_size >= 0 ) ASSERT( size == ct_size ); }

    const T&    operator()              ( PI r, PI c ) const { return content[ r * size() + c ]; }
    T&          operator()              ( PI r, PI c ) { return content[ r * size() + c ]; }

    struct EigenSystem {
        Point<T,ct_size>              values;   ///< eigenvalues, ascending order
        SimpleSquareMatrix<T,ct_size> vectors;  ///< row i = eigenvector i
    };

    auto        without_row_and_col     ( PI r, PI c ) const -> SimpleSquareMatrix<T,(ct_size>0?ct_size-1:-1)>;
    auto        with_replaced_col       ( PI c, const Vec &col ) const -> SimpleSquareMatrix;
    EigenSystem eigen_system            () const;
    T           determinant             () const;
    Vec         solve                   ( const Vec &vec ) const;
    PI          size                    ( PI ) const { return size(); }
    PI          size                    () const { return ct_size >= 0 ? ct_size : _size; }

    const T*    data                    () const { return content.data(); }
    T*          data                    () { return content.data(); }

    auto        begin                   () const { return content.begin(); }
    auto        begin                   () { return content.begin(); }
    auto        end                     () const { return content.end(); }
    auto        end                     () { return content.end(); }

    Content     content;
    PI          _size;
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
