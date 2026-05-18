#pragma once

#include "common_types.h"
#include "Cpu.h" // IWYU pragma: export
#include <span>

#ifdef USE_ZPP
#include <zpp_bits.h>
#endif

namespace sdot {

/// dynamic or static vector, with the same interfaces in all the cases (that's why there's no resize for instance, and size is required in ctors even if ct known)
///
template<class T,class Arch,int ct_size=-1>
class Vector {
public:
    struct       CtContent                   { std::byte values[ sizeof( T ) * ct_size ]; };
    struct       RtContent                   { T *values; PI size; };

    using        value_type                  = T;
    using        Content                     = std::conditional_t<(ct_size>=0),CtContent,RtContent>;

    /**/         Vector                      ( const auto &values ) requires( requires { values.size(); } );
    /**/         Vector                      ( Reserved, PI size ); ///< `new ( item ) T` has to be done
    /**/         Vector                      ( Values, auto &&...values );
    /**/         Vector                      ( Size, PI size, auto &&...ctor_args );
    /**/         Vector                      ();

    /**/         Vector                      ( const Vector &that );
    /**/         Vector                      ( Vector &&that ) noexcept;

    /**/         ~Vector                     ();

    Vector&      operator=                   ( const Vector &that );
    Vector&      operator=                   ( Vector &&that ) noexcept;

    T_Up         operator std::array<U,p>    () const;
    T_U          operator std::vector<U>     () const;
    /**/         operator std::span<T>       () const;

    const T&     operator[]                  ( PI index ) const;
    T&           operator[]                  ( PI index );

    bool         operator<                   ( const std::span<T> &that ) const;
    bool         operator<                   ( const Vector &that ) const;

    static Vector with_func                  ( PI size, auto &&func );
    static Vector value_at                   ( PI size, PI index, T value ); ///< 0 ... 0 value 0 ... 0. `value` is positionned at `index`
    static Vector zeros                      ( PI size );
    static Vector ones                       ( PI size );

    auto          with_pushed_value          ( T value ) const;
    auto          without_index              ( PI ind_to_remove ) const;
    T_d auto      from                       () const { return Vector<T,Arch,ct_size-d>( std::span( begin() + d, end() ) ); }

    auto          size                       () const;

    const T*      data                       () const;
    T*            data                       ();

    auto          begin                      () const;
    auto          begin                      ();
    auto          end                        () const;
    auto          end                        ();

    PI            arg_max                    () const;
    T             max                        () const;


    friend Vector normalized                 ( const Vector &a ) { return a / norm_2( a ); }
    friend T      norm_2_p2                  ( const Vector &a ) { return dot( a, a ); }
    friend T      norm_2                     ( const Vector &a ) { using namespace std; return sqrt( norm_2_p2( a ) ); }

    friend Vector operator+                  ( const Vector &a ) { Vector res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = + a[ i ]; return res; }
    friend Vector operator-                  ( const Vector &a ) { Vector res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = - a[ i ]; return res; }
    friend Vector floor                      ( const Vector &a ) { using namespace std; Vector res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = floor( a[ i ] ); return res; }
    friend Vector ceil                       ( const Vector &a ) { using namespace std; Vector res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = ceil ( a[ i ] ); return res; }

    friend Vector operator+                  ( const Vector &a, const Vector &b ) { Vector res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] + b[ i ]; return res; }
    friend Vector operator-                  ( const Vector &a, const Vector &b ) { Vector res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] - b[ i ]; return res; }
    friend Vector operator*                  ( const T &a, const Vector &b ) { Vector res( Size(), b.size() ); for( PI i = 0; i < b.size(); ++i ) res[ i ] = a * b[ i ]; return res; }
    friend Vector operator/                  ( const Vector &a, const T &b ) { Vector res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] / b; return res; }
    friend Vector max                        ( const Vector &a, const Vector &b ) { using namespace std; Vector res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = std::max( a[ i ], b[ i ] ); return res; }
    friend Vector min                        ( const Vector &a, const Vector &b ) { using namespace std; Vector res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = std::min( a[ i ], b[ i ] ); return res; }
    friend T      dot                        ( const Vector &a, const Vector &b ) { T res = 0; for( PI i = 0; i < a.size(); ++i ) res += a[ i ] * b[ i ]; return res; }

    friend void   operator+=                 ( Vector &a, const Vector &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] += b[ i ]; }
    friend void   operator-=                 ( Vector &a, const Vector &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] -= b[ i ]; }
    friend void   operator/=                 ( Vector &a, const auto &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] /= b; }

    friend void   _for_each_in_range         ( const Vector &beg, const Vector &end, Vector &cur, int i, const auto &func ) { if ( i == beg.size() ) { func( cur ); return; } for( T v = beg[ i ]; v < end[ i ]; ++v ) { cur[ i ] = v; _for_each_in_range( beg, end, cur, i + 1, func ); } }
    friend void   for_each_in_range          ( const Vector &beg, const Vector &end, auto &&func ) { Vector cur = beg; _for_each_in_range( beg, end, cur, 0, func ); }
    friend void   for_each_in_range          ( const Vector &end, auto &&func ) { Vector beg( Size(), end.size(), 0 ); for_each_in_range( beg, end, func ); }

    #ifdef        USE_ZPP
    using         serialize                  = zpp::bits::members<1>;
    #endif

    Content       content;
};


} // namespace sdot

#include "Vector.cxx" // IWYU pragma: export
