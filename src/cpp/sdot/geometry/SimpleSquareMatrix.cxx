#pragma once

#include "SimpleSquareMatrix.h"
#include <cmath>

namespace sdot {

#define UTP template<class T,int ct_size,class Arch>
#define DTP SimpleSquareMatrix<T,ct_size,Arch>

UTP SimpleSquareMatrix<T,(ct_size>0?ct_size-1:-1),Arch> DTP::without_row_and_col( PI wr, PI wc ) const {
    SimpleSquareMatrix<T,(ct_size>0?ct_size-1:-1),Arch> res( size() - 1 );
    for( PI r = 0; r < res.size(); ++r )
        for( PI c = 0; c < res.size(); ++c )
            res( r, c ) = operator()( r + ( r >= wr ), c + ( c >= wc ) );
    return res;
}

UTP auto DTP::with_func( PI size, auto &&func ) {
    SimpleSquareMatrix<T,ct_size,Arch> res( size );
    for( PI r = 0; r < size; ++r )
        for( PI c = 0; c < size; ++c )
            res( r, c ) = func( r, c );
    return res;
}

UTP DTP DTP::with_replaced_col( PI c, const Vec &col ) const {
    SimpleSquareMatrix res = *this;
    for( PI r = 0; r < size(); ++r )
        res( r, c ) = col[ r ];
    return res;
}

UTP T DTP::determinant() const {
    if ( size() == 1 )
        return operator()( 0, 0 );

    T sgn = 1, res = 0;
    for( PI r = 0; r < size(); ++r, sgn = -sgn )
        res += sgn * operator()( r, 0 ) * without_row_and_col( r, 0 ).determinant();
    return res;
}

UTP DTP DTP::cholesky() const {
    const PI nd = size();
    SimpleSquareMatrix L( nd );
    for ( PI i = 0; i < nd; ++i )
        for ( PI j = 0; j < nd; ++j )
            L( i, j ) = T( 0 );

    for ( PI j = 0; j < nd; ++j ) {
        // diagonal
        T s = operator()( j, j );
        for ( PI k = 0; k < j; ++k ) s -= L( j, k ) * L( j, k );
        L( j, j ) = std::sqrt( s );

        // column below diagonal
        const T inv_ljj = T( 1 ) / L( j, j );
        for ( PI i = j + 1; i < nd; ++i ) {
            T t = operator()( i, j );
            for ( PI k = 0; k < j; ++k ) t -= L( i, k ) * L( j, k );
            L( i, j ) = t * inv_ljj;
        }
    }
    return L;
}

UTP DTP::Vec DTP::solve( const Vec &vec ) const {
    T d = determinant();
    Vec res( size() );
    T sgn = 1;
    for( PI c = 0; c < size(); ++c, sgn = -sgn )
        res[ c ] = with_replaced_col( c, vec ).determinant() / d;
    return res;
}

#undef UTP
#undef DTP

} // namespace sdot


template<class T,int dim,class Arch>
std::ostream &operator<<( std::ostream &os, const sdot::SimpleSquareMatrix<T,dim,Arch> &p ) {
    for( sdot::PI r = 0; r < p.size(); ++r ) {
        os << "\n  ";
        for( sdot::PI c = 0; c < p.size(); ++c )
            os << ( c ? ", " : "" ) << p( r, c );
    }
    return os;
}
