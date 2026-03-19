#pragma once

#include "../support/common_macros.h"
#include "../support/common_types.h"
// #include <initializer_list>
#include <ostream>
#include <vector>
#include <array>

namespace sdot {

//
template<class T,int dim=-1>
class Point {
public:
    using    Data                    = std::array<T,dim>;
               
    /**/     Point                   ( PI /*size*/ = dim ) { for( auto &v : data ) v = 0; }

    T_p      operator std::array<T,p>() const { std::array<T,p> res; for( PI i = 0; i < std::min( p, size() ); ++i ) res[ i ] = data[ i ]; for( PI i = size(); i < p; ++i ) res[ i ] = 0; return res; }
    /**/     operator std::vector<T> () const { std::vector<T> res( size() ); for( PI i = 0; i < size(); ++i ) res[ i ] = data[ i ]; return res; }

    const T& operator[]              ( PI index ) const { return data[ index ]; }
    T&       operator[]              ( PI index ) { return data[ index ]; }
              
    PI       size                    () const;
   
    Data     data;
};

//
template<class T>
class Point<T,-1> {
public:
    using    Data                    = std::vector<T>;
        
    /**/     Point                   ( PI size ) : data( size, 0 ) {}

    T_p      operator std::array<T,p>() const { std::array<T,p> res; for( PI i = 0; i < std::min( p, size() ); ++i ) res[ i ] = data[ i ]; for( PI i = size(); i < p; ++i ) res[ i ] = 0; return res; }
    /**/     operator std::vector<T> () const { std::vector<T> res( size() ); for( PI i = 0; i < size(); ++i ) res[ i ] = data[ i ]; return res; }

    const T& operator[]              ( PI index ) const { return data[ index ]; }
    T&       operator[]              ( PI index ) { return data[ index ]; }
              
    PI       size                    () const;

    Data     data;
};

/// scalar product
T_Td T sp( const Point<T,d> &a, const Point<T,d> &b ) { T res = 0; for( PI i = 0; i < a.size(); ++i ) res += a[ i ] * b[ i ]; return res; }

/// euclidean norm
T_Td T norm_2_p2( const Point<T,d> &a ) { return sp( a, a ); }

/// euclidean norm
T_Td T norm_2( const Point<T,d> &a ) { using namespace std; return sqrt( norm_2_p2( a ) ); }

T_Td Point<T,d> normalized( const Point<T,d> &a ) { return a / norm_2( a ); }

//
T_Td void operator+=( Point<T,d> &a, const Point<T,d> &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] += b[ i ]; }
T_Td void operator-=( Point<T,d> &a, const Point<T,d> &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] -= b[ i ]; }

T_Td Point<T,d> operator+( const Point<T,d> &a, const Point<T,d> &b ) { Point<T,d> res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] + b[ i ]; return res; }
T_Td Point<T,d> operator-( const Point<T,d> &a, const Point<T,d> &b ) { Point<T,d> res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] - b[ i ]; return res; }

T_Td Point<T,d> operator*( const T &a, const Point<T,d> &b ) { Point<T,d> res( b.size() ); for( PI i = 0; i < b.size(); ++i ) res[ i ] = a * b[ i ]; return res; }
T_Td Point<T,d> operator/( const Point<T,d> &a, const T &b ) { Point<T,d> res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] / b; return res; }

T_Td Point<T,d> operator-( const Point<T,d> &a ) { Point<T,d> res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = - a[ i ]; return res; }

} // namespace sdot

T_Td std::ostream &operator<<( std::ostream &os, const sdot::Point<T,d> &p );
T_T std::ostream &operator<<( std::ostream &os, const std::vector<T> &p );

#include "Point.cxx"
