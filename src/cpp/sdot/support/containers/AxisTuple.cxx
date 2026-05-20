#pragma once

#include "AxisTuple.h"
#include "IndexRange.h"
#include <utility>

namespace sdot {

//
template<class TI,int ct_rank_0,class... attributes_0,int ct_rank_1,class... attributes_1>
auto concat( const AxisTuple<TI,ct_rank_0,attributes_0...> &t0, const AxisTuple<TI,ct_rank_1,attributes_1...> &t1 ) {
    using TR = AxisTuple<TI,ct_rank_0+ct_rank_1,attributes_0...,typename WithOffset<int,ct_rank_0,attributes_1>::type...>;
    return t0.apply_values( [&]( auto ...v0 ) {
        return t1.apply_values( [&]( auto ...v1 ) {
            return TR( Values(), v0..., v1... );
        } );
    } );
}

// ---- runtime front value -----------------------------------------------------------
#define UTP template<class _TI,int _ct_rank,class... ct_tail>
#define DTP AxisTuple<_TI,_ct_rank,ct_tail...>

UTP HD DTP::AxisTuple( Function, auto &&func, auto index ) : front_value( func( index ) ), next_values( Function(), FORWARD( func ), index + Ct<int,1>() ) {
}

UTP HD DTP::AxisTuple( Function, auto &&func ) : AxisTuple( Function(), func, Ct<int,0>() ) {
}

UTP HD DTP::AxisTuple( Values, TI front_value, auto... next_values ) : front_value( front_value ), next_values( Values(), next_values... ) {
}

UTP HD auto DTP::all_indices() const { return IndexRange<DTP>{ *this }; }

UTP HD auto DTP::apply_values( auto &&cb ) const { return next_values.apply_values( [&]( auto ...nxt ) {
    return cb( front_value, nxt... ); } );
}

UTP T_Uu HD auto DTP::operator[]( Ct<U,u> ) const {
    if constexpr ( u == 0 )
        return front_value;
    else
        return next_values[ Ct<U,u-1>() ];
}

UTP HD auto DTP::operator[]( TI u ) const {
    if constexpr( ct_rank == 1 )
        return front_value;
    else
        return u ? SI( next_values[ u - 1 ] ) : SI( front_value );
}

UTP HD auto DTP::has_value( auto &&func ) const {
    if constexpr ( requires { DECAYED_TYPE_OF( func( front_value ) )::value; } ) {
        if constexpr ( DECAYED_TYPE_OF( func( front_value ) )::value )
            return Ct<bool,true>();
        else
            return next_values.has_value( func );
    } else {
        return func( front_value ) || next_values.has_value( func );
    }
}

UTP HD auto DTP::all_value( auto &&func ) const {
    if constexpr ( requires { DECAYED_TYPE_OF( func( front_value ) )::value; } ) {
        if constexpr ( ! DECAYED_TYPE_OF( func( front_value ) )::value )
            return Ct<bool,false>();
        else
            return next_values.all_value( func );
    } else {
        return func( front_value ) && next_values.all_value( func );
    }
}

UTP HD void DTP::display( auto &os, const char *prefix ) const {
    if ( prefix == nullptr )
        prefix = "[ ";
    next_values.display( os << prefix << front_value, ", " );
}

UTP HD auto DTP::size() const {
    return Ct<int,ct_rank>();
}

UTP HD auto DTP::front_shape() const {
    return AxisTuple<TI,1>( Values(), front_value );
}

UTP T_Uu HD auto DTP::without_index( Ct<U,u> ) const {
    if constexpr ( u == 0 )
        return next_values;
    else
        return concat( front_shape(), next_values.without_index( Ct<U,u-1>() ) );
}

UTP auto DTP::without_index( TI u ) const {
    using Res = AxisTuple<TI,ct_rank-1>;
    return apply_values( [&]( auto ...vals ) {
        TI arr[ ct_rank ] = { TI( vals )... };
        return [&]<std::size_t... Js>( std::index_sequence<Js...> ) {
            return Res( Values(), arr[ Js + ( TI( Js ) >= u ? 1 : 0 ) ]... );
        }( std::make_index_sequence<ct_rank-1>{} );
    } );
}

#undef UTP
#undef DTP

// // ---- compile-time front value (KnownAxisSize at offset 0) --------------------------
// #define UTP template<class _TI,class _Arch,int _ct_rank,_TI ct_front_value,class... ct_tail>
// #define DTP AxisTuple<_TI,_Arch,_ct_rank,KnownAxisSize<_TI,0,ct_front_value>,ct_tail...>

// UTP HD DTP::AxisTuple( Function, auto &&func, auto index ) : next_values( Function(), FORWARD( func ), index + Ct<int,1>() ) { ASSERT_EQ( func( index ), ct_front_value ); }
// UTP HD DTP::AxisTuple( Function, auto &&func ) : AxisTuple( Function(), func, Ct<int,0>() ) {}
// UTP HD DTP::AxisTuple( Values, TI front_value, auto... next_values ) : next_values( Values(), next_values... ) { ASSERT_EQ( front_value, ct_front_value ); }

// UTP HD void DTP::for_each_index( auto &&func, auto ...indices_so_far ) const { for( TI i = 0; i < ct_front_value; ++i ) next_values.for_each_index( func, indices_so_far..., i ); }
// UTP HD auto DTP::apply_values( auto &&cb ) const { return next_values.apply_values( [&]( auto ...nxt ) { return cb( front_value, nxt... ); } ); }
// UTP T_Uu HD auto DTP::operator[]( Ct<U,u> ) const { if constexpr ( u == 0 ) return front_value; else return next_values[ Ct<U,u-1>() ]; }
// UTP HD auto DTP::operator[]( TI u ) const { if ( u == 0 ) return TI( ct_front_value ); return next_values[ u - 1 ]; }
// UTP HD auto DTP::has_value( auto &&func ) const { return _axis_tuple_has_value( *this, func ); }
// UTP HD auto DTP::all_value( auto &&func ) const { return _axis_tuple_all_value( *this, func ); }
// UTP HD auto DTP::nb_items() const { return front_value * next_values.nb_items(); }
// UTP HD void DTP::display( auto &os, const char *prefix ) const { if ( prefix == nullptr ) prefix = "[ "; next_values.display( os << prefix << ct_front_value, ", " ); }
// UTP HD auto DTP::size() const { return Ct<int,ct_rank>(); }
// UTP HD auto DTP::front_shape() const { return AxisTuple<TI,Arch,1,KnownAxisSize<TI,0,ct_front_value>>( Values(), ct_front_value ); }
// UTP T_Uu HD auto DTP::without_index( Ct<U,u> ) const { if constexpr ( u == 0 ) return next_values; else return concat( front_shape(), next_values.without_index( Ct<U,u-1>() ) ); }
// UTP auto DTP::without_index( TI u ) const { return _axis_tuple_without_index( *this, u ); }

// #undef UTP
// #undef DTP

// ---- rank 0 ------------------------------------------------------------------------
#define UTP template<class _TI>
#define DTP AxisTuple<_TI,0>

UTP HD auto DTP::all_indices() const { return IndexRange<DTP>{ *this }; }
UTP HD auto DTP::apply_values( auto &&cb ) const { return cb(); }
UTP HD auto DTP::operator[]( TI ) const -> TI { ASSERT( false ); return 0; }
UTP HD auto DTP::all_value( auto &&/*func*/ ) const { return Ct<bool,true>(); }
UTP HD auto DTP::has_value( auto &&/*func*/ ) const { return Ct<bool,false>(); }
UTP HD void DTP::display( auto &os, const char *prefix ) const { if ( prefix == nullptr ) os << "[]"; else os << " ]"; }
UTP HD auto DTP::size() const { return Ct<int,0>(); }

#undef UTP
#undef DTP

} // namespace sdot
