#pragma once

#include "../support/DsVec.h"

namespace sdot {

// _size
template<class T,int ct_size,class Arch>
class SimpleSquareMatrix {
public:
    struct             EigenSystem             { DsVec<T,ct_size,Arch> values;   /* ascending order */ SimpleSquareMatrix<T,ct_size,Arch> vectors;  /* row i = eigenvector i */ };
    using              value_type              = T;
    using              Content                 = DsVec<T,(ct_size>=0?ct_size*ct_size:-1),Arch>;
    using              Vec                     = DsVec<T,ct_size,Arch>;

    /**/               SimpleSquareMatrix      ( PI size ) : _content( Size(), size * size ), _size( size ) { if ( ct_size >= 0 ) ASSERT( size == ct_size ); }

    static auto        with_func               ( PI size, auto &&func );

    const T&           operator()              ( PI r, PI c ) const { return _content[ r * size() + c ]; }
    T&                 operator()              ( PI r, PI c ) { return _content[ r * size() + c ]; }


    auto               without_row_and_col     ( PI r, PI c ) const -> SimpleSquareMatrix<T,(ct_size>0?ct_size-1:-1),Arch>;
    auto               with_replaced_col       ( PI c, const Vec &col ) const -> SimpleSquareMatrix;
    EigenSystem        eigen_system            () const;
    T                  determinant             () const;
    SimpleSquareMatrix cholesky                () const;  ///< returns L s.t. *this = L * L^T (H must be SPD)
    Vec                solve_ge                ( Vec b ) const;   ///< Gaussian elimination with partial pivoting; zero pivot → x[p]=0 (handles degenerate cells)
    Vec                solve                   ( const Vec &vec ) const;
    SimpleSquareMatrix inverse                 () const;  ///< Gauss-Jordan on [A | I]; zero pivot row → identity row in result
    PI                 size                    ( PI ) const { return size(); }
    PI                 size                    () const { return ct_size >= 0 ? ct_size : _size; }

    const T*           data                    () const { return _content.data(); }
    T*                 data                    () { return _content.data(); }

    auto               begin                   () const { return _content.begin(); }
    auto               begin                   () { return _content.begin(); }
    auto               end                     () const { return _content.end(); }
    auto               end                     () { return _content.end(); }

    Content            _content;
    PI                 _size;
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

T_TdA std::ostream &operator<<( std::ostream &os, const sdot::SimpleSquareMatrix<T,d,A> &p );

#include "SimpleSquareMatrix.cxx"
