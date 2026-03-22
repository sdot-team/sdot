#pragma once

#include "../support/common_macros.h"
#include "../support/common_types.h"
#include <zpp_bits.h>
#include <ostream>
#include <vector>
#include <array>
#include <span>

namespace sdot {

// dim >= 0
template<class T,int dim=-1>
class Point {
public:
    using    value_type              = T;
    using    serialize               = zpp::bits::members<1>;
    using    Content                 = std::array<T,dim>;

    /**/     Point                   ( std::initializer_list<T> values ) { auto iter = values.begin(); for( PI i = 0; i < std::min( values.size(), size() ); ++i ) content[ i ] = *( iter++ ); for( PI i = values.size(); i < size(); ++i ) content[ i ] = 0; }
    // T_d   Point                   ( const Point<T,d> &values ) : data( values.begin(), values.end() ) {}
    // /**/  Point                   ( const Point &values ) : data( values.begin(), values.end() ) {}
    /**/     Point                   ( std::span<T> values ) { for( PI i = 0; i < std::min( values.size(), size() ); ++i ) content[ i ] = values[ i ]; for( PI i = values.size(); i < size(); ++i ) content[ i ] = 0; }
    /**/     Point                   ( PI /*size*/ = dim ) { for( auto &v : content ) v = 0; }

    T_p      operator std::array<T,p>() const { std::array<T,p> res; for( PI i = 0; i < std::min( p, size() ); ++i ) res[ i ] = content[ i ]; for( PI i = size(); i < p; ++i ) res[ i ] = 0; return res; }
    /**/     operator std::vector<T> () const { std::vector<T> res( size() ); for( PI i = 0; i < size(); ++i ) res[ i ] = content[ i ]; return res; }
    /**/     operator std::span<T>   () const { return std::span<T>( const_cast<T *>( content.data() ), content.size() ); }

    const T& operator[]              ( PI index ) const { return content[ index ]; }
    T&       operator[]              ( PI index ) { return content[ index ]; }

    bool     operator<               ( const std::span<T> &that ) const { return content < that.data; }
    bool     operator<               ( const Point &that ) const { return content < that.content; }

    auto     with_pushed_value       ( T value ) const { Point<T,dim+1> res( dim + 1 ); for( PI i = 0; i < size(); ++i ) res[ i ] = content[ i ]; res[ size() ] = value; return res; }
    auto     without_index           ( PI ind_to_remove ) const { Point<T,dim-1> res( dim-1 ); for( PI i = 0; i < ind_to_remove; ++i ) res[ i ] = content[ i ]; for( PI i = ind_to_remove + 1; i < size(); ++i ) res[ i - 1 ] = content[ i ]; return res; }
    PI       size                    () const;

    const T* data                    () const { return content.data(); }
    T*       data                    () { return content.data(); }

    auto     begin                   () const { return content.begin(); }
    auto     begin                   () { return content.begin(); }
    auto     end                     () const { return content.end(); }
    auto     end                     () { return content.end(); }


    Content  content;
};

//
template<class T>
class Point<T,-1> {
public:
    using       value_type              = T;
    using       serialize               = zpp::bits::members<1>;
    using       Content                    = std::vector<T>;

    /**/        Point                   ( std::initializer_list<T> values ) : content( values ) {}
    // T_d      Point                   ( const Point<T,d> &values ) : data( values.begin(), values.end() ) {}
    // /**/     Point                   ( const Point &values ) : data( values.begin(), values.end() ) {}
    /**/        Point                   ( std::span<T> values ) : content( values.begin(), values.end() ) {}
    /**/        Point                   ( PI size = 0 ) : content( size, 0 ) {}

    T_p         operator std::array<T,p>() const { std::array<T,p> res; for( PI i = 0; i < std::min( p, size() ); ++i ) res[ i ] = content[ i ]; for( PI i = size(); i < p; ++i ) res[ i ] = 0; return res; }
    /**/        operator std::vector<T> () const { std::vector<T> res( size() ); for( PI i = 0; i < size(); ++i ) res[ i ] = content[ i ]; return res; }
    /**/        operator std::span<T>   () const { return std::span<T>( const_cast<T *>( content.data() ), content.size() ); }

    const T&    operator[]              ( PI index ) const { return content[ index ]; }
    T&          operator[]              ( PI index ) { return content[ index ]; }

    bool        operator<               ( const std::span<T> &that ) const { return content < that.data; }
    bool        operator<               ( const Point &that ) const { return content < that.content; }

    auto        with_pushed_value       ( T value ) const { Point<T,-1> res( size() + 1 ); for( PI i = 0; i < size(); ++i ) res[ i ] = content[ i ]; res[ size() ] = value; return res; }
    auto        without_index           ( PI ind_to_remove ) const { Point<T,-1> res( size() - 1 ); for( PI i = 0; i < ind_to_remove; ++i ) res[ i ] = content[ i ]; for( PI i = ind_to_remove + 1; i < size(); ++i ) res[ i - 1 ] = content[ i ]; return res; }
    PI          size                    () const;

    const T*    data                    () const { return content.data(); }
    T*          data                    () { return content.data(); }

    auto        begin                   () const { return content.begin(); }
    auto        begin                   () { return content.begin(); }
    auto        end                     () const { return content.end(); }
    auto        end                     () { return content.end(); }

    Content     content;
};

/// scalar product
T_Td T dot( const Point<T,d> &a, const Point<T,d> &b ) { T res = 0; for( PI i = 0; i < a.size(); ++i ) res += a[ i ] * b[ i ]; return res; }

/// euclidean norm
T_Td T norm_2_p2( const Point<T,d> &a ) { return dot( a, a ); }

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
