#pragma once

#include "SimpleSquareMatrix.h"

namespace sdot {

#define UTP template<class T,int ct_size>
#define DTP SimpleSquareMatrix<T,ct_size>

UTP SimpleSquareMatrix<T,(ct_size>0?ct_size-1:-1)> DTP::without_row_and_col( PI wr, PI wc ) const {
    SimpleSquareMatrix<T,(ct_size>0?ct_size-1:-1)> res( size() - 1 );
    for( PI r = 0; r < res.size(); ++r )
        for( PI c = 0; c < res.size(); ++c )
            res( r, c ) = operator()( r + ( r >= wr ), c + ( c >= wc ) );
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


template<class T, int dim>
std::ostream &operator<<( std::ostream &os, const sdot::SimpleSquareMatrix<T,dim> &p ) {
    for( sdot::PI r = 0; r < p.size(); ++r ) {
        os << "\n  ";
        for( sdot::PI c = 0; c < p.size(); ++c )
            os << ( c ? ", " : "" ) << p( r, c );
    }
    return os;
}
