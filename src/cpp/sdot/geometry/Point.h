#pragma once

#include "../support/common_macros.h"
#include "../support/TensorView.h"
#include <zpp_bits.h>
#include <vector>
#include <array>
#include <span>

namespace sdot {

// dim >= 0
template<class T,int ct_dim,class Arch>
class Point {
public:
    using    value_type              = T;
    using    serialize               = zpp::bits::members<1>;
    using    Content                 = std::array<T,ct_dim>;

    /**/     Point                   ( std::initializer_list<T> values ) { auto iter = values.begin(); for( PI i = 0; i < std::min( values.size(), size() ); ++i ) content[ i ] = *( iter++ ); for( PI i = values.size(); i < size(); ++i ) content[ i ] = 0; }
    T_U      Point                   ( TensorView<U,1,Arch> values ) { for( PI i = 0; i < std::min( values.size(), size() ); ++i ) content[ i ] = values[ i ]; for( PI i = values.size(); i < size(); ++i ) content[ i ] = 0; }
    // T_d   Point                   ( const Point<T,d,A> &values ) : data( values.begin(), values.end() ) {}
    // /**/  Point                   ( const Point &values ) : data( values.begin(), values.end() ) {}
    /**/     Point                   ( std::span<const T> values ) { for( PI i = 0; i < std::min( values.size(), size() ); ++i ) content[ i ] = values[ i ]; for( PI i = values.size(); i < size(); ++i ) content[ i ] = 0; }
    /**/     Point                   ( std::span<T> values ) { for( PI i = 0; i < std::min( values.size(), size() ); ++i ) content[ i ] = values[ i ]; for( PI i = values.size(); i < size(); ++i ) content[ i ] = 0; }
    /**/     Point                   ( PI /*size*/ = ct_dim ) { for( auto &v : content ) v = 0; }

    T_Up     operator std::array<U,p>() const { std::array<U,p> res; for( PI i = 0; i < std::min( p, size() ); ++i ) res[ i ] = content[ i ]; for( PI i = size(); i < p; ++i ) res[ i ] = 0; return res; }
    T_U      operator std::vector<U> () const { std::vector<U> res( size() ); for( PI i = 0; i < size(); ++i ) res[ i ] = content[ i ]; return res; }
    /**/     operator std::span<T>   () const { return std::span<T>( const_cast<T *>( content.data() ), content.size() ); }

    const T& operator[]              ( PI index ) const { return content[ index ]; }
    T&       operator[]              ( PI index ) { return content[ index ]; }

    bool     operator<               ( const std::span<T> &that ) const { return content < that.data; }
    bool     operator<               ( const Point &that ) const { return content < that.content; }

    auto     with_pushed_value       ( T value ) const { Point<T,ct_dim+1,Arch> res( ct_dim + 1 ); for( PI i = 0; i < size(); ++i ) res[ i ] = content[ i ]; res[ size() ] = value; return res; }
    auto     without_index           ( PI ind_to_remove ) const { Point<T,ct_dim-1,Arch> res( ct_dim-1 ); for( PI i = 0; i < ind_to_remove; ++i ) res[ i ] = content[ i ]; for( PI i = ind_to_remove + 1; i < size(); ++i ) res[ i - 1 ] = content[ i ]; return res; }
    PI       size                    () const;

    const T* data                    () const { return content.data(); }
    T*       data                    () { return content.data(); }

    auto     begin                   () const { return content.begin(); }
    auto     begin                   () { return content.begin(); }
    auto     end                     () const { return content.end(); }
    auto     end                     () { return content.end(); }

    PI       arg_max                 () const { PI res = 0; for( PI i = 1; i < size(); ++i ) if ( content[ res ] < content[ i ] ) res = i; return res; }

    Content  content;
};

//
template<class T,class Arch>
class Point<T,-1,Arch> {
public:
    using       value_type              = T;
    using       serialize               = zpp::bits::members<1>;
    using       Content                 = std::vector<T>;

    /**/        Point                   ( std::initializer_list<T> values ) : content( values ) {}
    // T_d      Point                   ( const Point<T,d,A> &values ) : data( values.begin(), values.end() ) {}
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

    auto        with_pushed_value       ( T value ) const { Point<T,-1,Arch> res( size() + 1 ); for( PI i = 0; i < size(); ++i ) res[ i ] = content[ i ]; res[ size() ] = value; return res; }
    auto        without_index           ( PI ind_to_remove ) const { Point<T,-1,Arch> res( size() - 1 ); for( PI i = 0; i < ind_to_remove; ++i ) res[ i ] = content[ i ]; for( PI i = ind_to_remove + 1; i < size(); ++i ) res[ i - 1 ] = content[ i ]; return res; }
    PI          size                    () const;

    const T*    data                    () const { return content.data(); }
    T*          data                    () { return content.data(); }

    auto        begin                   () const { return content.begin(); }
    auto        begin                   () { return content.begin(); }
    auto        end                     () const { return content.end(); }
    auto        end                     () { return content.end(); }

    PI          arg_max                 () const { PI res = 0; for( PI i = 1; i < size(); ++i ) if ( content[ res ] < content[ i ] ) res = i; return res; }

    Content     content;
};

/// scalar product
T_TdA T dot( const Point<T,d,A> &a, const Point<T,d,A> &b ) { T res = 0; for( PI i = 0; i < a.size(); ++i ) res += a[ i ] * b[ i ]; return res; }

/// euclidean norm
T_TdA T norm_2_p2( const Point<T,d,A> &a ) { return dot( a, a ); }

/// euclidean norm
T_TdA T norm_2( const Point<T,d,A> &a ) { using namespace std; return sqrt( norm_2_p2( a ) ); }

T_TdA Point<T,d,A> normalized( const Point<T,d,A> &a ) { return a / norm_2( a ); }

//
T_TdA void operator+=( Point<T,d,A> &a, const Point<T,d,A> &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] += b[ i ]; }
T_TdA void operator-=( Point<T,d,A> &a, const Point<T,d,A> &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] -= b[ i ]; }
T_TdA void operator/=( Point<T,d,A> &a, const auto &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] /= b; }

T_TdA Point<T,d,A> operator+( const Point<T,d,A> &a, const Point<T,d,A> &b ) { Point<T,d,A> res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] + b[ i ]; return res; }
T_TdA Point<T,d,A> operator-( const Point<T,d,A> &a, const Point<T,d,A> &b ) { Point<T,d,A> res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] - b[ i ]; return res; }

T_TdA Point<T,d,A> operator*( const T &a, const Point<T,d,A> &b ) { Point<T,d,A> res( b.size() ); for( PI i = 0; i < b.size(); ++i ) res[ i ] = a * b[ i ]; return res; }
T_TdA Point<T,d,A> operator/( const Point<T,d,A> &a, const T &b ) { Point<T,d,A> res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] / b; return res; }

T_TdA Point<T,d,A> operator+( const Point<T,d,A> &a ) { Point<T,d,A> res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = + a[ i ]; return res; }
T_TdA Point<T,d,A> operator-( const Point<T,d,A> &a ) { Point<T,d,A> res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = - a[ i ]; return res; }

} // namespace sdot

T_TdA std::ostream &operator<<( std::ostream &os, const sdot::Point<T,d,A> &p );
T_Up std::ostream &operator<<( std::ostream &os, const std::array<U,p> &v );
T_T std::ostream &operator<<( std::ostream &os, const std::vector<T> &p );

#include "Point.cxx"
