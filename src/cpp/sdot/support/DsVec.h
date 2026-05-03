#pragma once

#include "common_types.h"
#include "Arch.h"
#include <cmath>
#include <span>

#ifdef USE_ZPP
#include <zpp_bits.h>
#endif

namespace sdot {

// dynamic or static vector, with the same interfaces in all the cases (that's why there's no resize for instance)
template<class T,int ct_size,class Arch=Cpu>
class DsVec {
public:
    struct       CtContent               { std::byte values[ sizeof( T ) * ct_size ]; };
    struct       RtContent               { T *values; PI size; };

    using        value_type              = T;
    using        Content                 = std::conditional_t<(ct_size>=0),CtContent,RtContent>;

    /**/         DsVec                   ( const auto &values ) requires( requires { values.size(); } );
    /**/         DsVec                   ( Reserved, PI size ); ///< `new ( item ) T` has to be done
    /**/         DsVec                   ( Values, auto &&...values );
    /**/         DsVec                   ( Size, PI size, auto &&...ctor_args );

    /**/         DsVec                   ( const DsVec &that );
    /**/         DsVec                   ( DsVec &&that ) noexcept;

    /**/         ~DsVec                  ();

    DsVec&       operator=               ( const DsVec &that );
    DsVec&       operator=               ( DsVec &&that ) noexcept;

    T_Up         operator std::array<U,p>() const;
    T_U          operator std::vector<U> () const;
    /**/         operator std::span<T>   () const;

    const T&     operator[]              ( PI index ) const;
    T&           operator[]              ( PI index );

    bool         operator<               ( const std::span<T> &that ) const;
    bool         operator<               ( const DsVec &that ) const;

    static DsVec with_func               ( PI size, auto &&func );
    static DsVec value_at                ( PI size, PI index, T value ); ///< 0 ... 0 value 0 ... 0. `value` is positionned at `index`
    static DsVec zeros                   ( PI size );
    static DsVec ones                    ( PI size );

    auto         with_pushed_value       ( T value ) const;
    auto         without_index           ( PI ind_to_remove ) const;
    T_d auto     from                    () const { return DsVec<T,ct_size-d,Arch>( std::span( begin() + d, end() ) ); }
    PI           size                    () const;

    const T*     data                    () const;
    T*           data                    ();

    auto         begin                   () const;
    auto         begin                   ();
    auto         end                     () const;
    auto         end                     ();

    PI           arg_max                 () const;
    T            max                     () const;


    friend DsVec normalized              ( const DsVec &a ) { return a / norm_2( a ); }
    friend T     norm_2_p2               ( const DsVec &a ) { return dot( a, a ); }
    friend T     norm_2                  ( const DsVec &a ) { using namespace std; return sqrt( norm_2_p2( a ) ); }

    friend DsVec operator+               ( const DsVec &a ) { DsVec res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = + a[ i ]; return res; }
    friend DsVec operator-               ( const DsVec &a ) { DsVec res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = - a[ i ]; return res; }
    friend DsVec floor                   ( const DsVec &a ) { using namespace std; DsVec res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = floor( a[ i ] ); return res; }
    friend DsVec ceil                    ( const DsVec &a ) { using namespace std; DsVec res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = ceil ( a[ i ] ); return res; }

    friend DsVec operator+               ( const DsVec &a, const DsVec &b ) { DsVec res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] + b[ i ]; return res; }
    friend DsVec operator-               ( const DsVec &a, const DsVec &b ) { DsVec res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] - b[ i ]; return res; }
    friend DsVec operator*               ( const T &a, const DsVec &b ) { DsVec res( Size(), b.size() ); for( PI i = 0; i < b.size(); ++i ) res[ i ] = a * b[ i ]; return res; }
    friend DsVec operator/               ( const DsVec &a, const T &b ) { DsVec res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = a[ i ] / b; return res; }
    friend DsVec max                     ( const DsVec &a, const DsVec &b ) { using namespace std; DsVec res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = std::max( a[ i ], b[ i ] ); return res; }
    friend DsVec min                     ( const DsVec &a, const DsVec &b ) { using namespace std; DsVec res( Size(), a.size() ); for( PI i = 0; i < a.size(); ++i ) res[ i ] = std::min( a[ i ], b[ i ] ); return res; }
    friend T     dot                     ( const DsVec &a, const DsVec &b ) { T res = 0; for( PI i = 0; i < a.size(); ++i ) res += a[ i ] * b[ i ]; return res; }

    friend void  operator+=              ( DsVec &a, const DsVec &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] += b[ i ]; }
    friend void  operator-=              ( DsVec &a, const DsVec &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] -= b[ i ]; }
    friend void  operator/=              ( DsVec &a, const auto &b ) { for( PI i = 0; i < a.size(); ++i ) a[ i ] /= b; }

    friend void  _for_each_in_range      ( const DsVec &beg, const DsVec &end, DsVec &cur, int i, const auto &func ) { if ( i == beg.size() ) { func( cur ); return; } for( T v = beg[ i ]; v < end[ i ]; ++v ) { cur[ i ] = v; _for_each_in_range( beg, end, cur, i + 1, func ); } }
    friend void  for_each_in_range       ( const DsVec &beg, const DsVec &end, auto &&func ) { DsVec cur = beg; _for_each_in_range( beg, end, cur, 0, func ); }
    friend void  for_each_in_range       ( const DsVec &end, auto &&func ) { DsVec beg( Size(), end.size(), 0 ); for_each_in_range( beg, end, func ); }

    #ifdef       USE_ZPP
    using        serialize               = zpp::bits::members<1>;
    #endif

    friend std::ostream &operator<<      ( std::ostream &os, const DsVec &p ) { for( sdot::PI i = 0; i < p.size(); ++i ) os << ( i ? ", " : "" ) << p[ i ]; return os; }

    Content      content;
};


} // namespace sdot

#include "DsVec.cxx" // IWYU pragma: export
