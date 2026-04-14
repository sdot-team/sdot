#pragma once

#include "common_types.h"
#include "Arch.h"
#include <span>

#ifdef USE_ZPP
#include <zpp_bits.h>
#endif

namespace sdot {

// dim >= 0. Known size
template<class T,int ct_size,class Arch=Cpu>
class Point {
public:
    using        Content                 = std::conditional_t<(ct_size>=0),std::array<T,PI(ct_size)>,std::vector<T>>;
    using        value_type              = T;
    #ifdef       USE_ZPP
    using        serialize               = zpp::bits::members<1>;
    #endif

    T_U          Point                   ( const std::initializer_list<U> &values );
    /**/         Point                   ( const auto &values ) requires( requires { values.size(); } );
    /**/         Point                   ( PI size, auto &&value );
    /**/         Point                   ( PI size = ct_size );

    T_Up         operator std::array<U,p>() const;
    T_U          operator std::vector<U> () const;
    /**/         operator std::span<T>   () const;

    const T&     operator[]              ( PI index ) const;
    T&           operator[]              ( PI index );

    bool         operator<               ( const std::span<T> &that ) const;
    bool         operator<               ( const Point &that ) const;

    auto         with_pushed_value       ( T value ) const;
    auto         without_index           ( PI ind_to_remove ) const;
    PI           size                    () const;

    const T*     data                    () const;
    T*           data                    ();

    auto         begin                   () const;
    auto         begin                   ();
    auto         end                     () const;
    auto         end                     ();

    PI           arg_max                 () const;


    friend Point normalized              ( const Point &a ) { return a / norm_2( a ); }
    friend T     norm_2_p2               ( const Point &a ) { return dot( a, a ); }
    friend T     norm_2                  ( const Point &a ) { using namespace std; return sqrt( norm_2_p2( a ) ); }

    friend Point operator+               ( const Point &a ) { Point res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = + a[ i ]; return res; }
    friend Point operator-               ( const Point &a ) { Point res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = - a[ i ]; return res; }
    friend Point floor                   ( const Point &a ) { using namespace std; Point res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = floor( a[ i ] ); return res; }
    friend Point ceil                    ( const Point &a ) { using namespace std; Point res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = ceil ( a[ i ] ); return res; }

    friend Point operator+               ( const Point &a, const Point &b ) { Point res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] + b[ i ]; return res; }
    friend Point operator-               ( const Point &a, const Point &b ) { Point res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] - b[ i ]; return res; }
    friend Point operator*               ( const T &a, const Point &b ) { Point res( b.size() ); for( PI i = 0; i < b.size(); ++i ) res[ i ] = a * b[ i ]; return res; }
    friend Point operator/               ( const Point &a, const T &b ) { Point res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] / b; return res; }
    friend Point max                     ( const Point &a, const Point &b ) { using namespace std; Point res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = max( a[ i ], b[ i ] ); return res; }
    friend Point min                     ( const Point &a, const Point &b ) { using namespace std; Point res( a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = min( a[ i ], b[ i ] ); return res; }
    friend T     dot                     ( const Point &a, const Point &b ) { T res = 0; for( PI i = 0; i < a.size(); ++i ) res += a[ i ] * b[ i ]; return res; }

    friend void  operator+=              ( Point &a, const Point &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] += b[ i ]; }
    friend void  operator-=              ( Point &a, const Point &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] -= b[ i ]; }
    friend void  operator/=              ( Point &a, const auto &b )         { for( PI i = 0; i < a.size(); ++i ) a[ i ] /= b; }

    friend void  _for_each_in_range       ( const Point &beg, const Point &end, Point &cur, int i, const auto &func ) { if ( i == beg.size() ) { func( cur ); return; } for( T v = beg[ i ]; v < end[ i ]; ++v ) { cur[ i ] = v; _for_each_in_range( beg, end, cur, i + 1, func ); } }
    friend void  for_each_in_range        ( const Point &beg, const Point &end, auto &&func ) { Point cur = beg; _for_each_in_range( beg, end, cur, 0, func ); }

    Content      content;
};


} // namespace sdot

T_TdA std::ostream &operator<<( std::ostream &os, const sdot::Point<T,d,A> &p );

#include "Point.cxx" // IWYU pragma: export
