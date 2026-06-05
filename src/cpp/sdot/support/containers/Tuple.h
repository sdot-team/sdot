#pragma once

#include "../common_types.h"
#include "../Ct.h"  // IWYU pragma: export

namespace sdot {

template<class... Types>
class Tuple;


/// known ct_rank > 0, runtime front value
///   ct_tail may be empty, or contain higher-index StaticAxisValue entries
template<class Head,class... Tail>
class Tuple<Head,Tail...> {
public:
    SCInt          ct_size            = 1 + sizeof...( Tail );
    using          Next               = Tuple<Tail...>;

    HD T_TA        Tuple              ( Function, T &&func, A index ); ///< allows for starting with index != 0
    HD T_T         Tuple              ( Function, T &&func );

    HD T_Tv        Tuple              ( Values, T head, V... tail );
    T_VT HD        Tuple              ( const Tuple<T...> &that );
    HD             Tuple              ( const Tuple &that ) = default;
    HD             Tuple              () = default;

    HD T_T void    for_each_item      ( T &&cb_func ) const;
    HD T_T void    for_each_item      ( T &&cb_func );
    HD T_T auto    apply_values       ( T &&cb_func ) const;
    HD T_T auto    operator[]         ( T &&index ) const;
    HD T_T auto    operator==         ( const T &that ) const;
    static HD auto size               ();
    HD T_TA void   set                ( T &&index, A &&value );

    HD T_T auto    with_appended_value( T &&new_value ) const;
    HD T_T auto    without_index      ( T index ) const;

    Head           head;
    Next           tail;
};

/// known ct_rank == 0
template<>
class Tuple<> {
public:
    SCInt          ct_size            = 0;

    HD T_TA        Tuple              ( Function, T &&/*func*/, A /*index*/ );
    HD T_T         Tuple              ( Function, T &&/*func*/ );
    HD             Tuple              ( const Tuple &that ) = default;
    HD             Tuple              ( Values );
    HD             Tuple              () = default;

    HD T_T void    for_each_item      ( T &&/* cb */ ) const;
    HD T_T auto    apply_values       ( T &&cb ) const;
    HD T_T Void    operator[]         ( T ) const;
    HD T_T auto    operator==         ( const T &that ) const;
    static HD auto size               ();
    HD T_TA void   set                ( T &&index, A &&value );

    HD T_T auto    with_appended_value( T &&new_value ) const;
};

T_VT constexpr auto tuple( T &&...a ) {
    return Tuple<DECAYED_TYPE_OF( a )...>( Values(), a... );
}

template<class... A,class... B>
constexpr auto concat( const Tuple<A...> &a, const Tuple<B...> &b ) {
    return a.apply_values( [&]( auto... va ) {
        return b.apply_values( [&]( auto... vb ) {
            return tuple( va..., vb... );
        } );
    } );
}

T_TA constexpr auto map( T &&list, A &&func ) { // requires requires { list.apply_values( []( auto... ) {} ); } {
    return list.apply_values( [&]( auto... values ) {
        return tuple( func( values )... );
    } );
}

template<class... A>
constexpr auto product( const Tuple<A...> &list ) {
    return list.apply_values( [&]( auto... values ) {
        return ( values * ... * Ct<int,1>() );
    } );
}

template<class... A>
constexpr auto sum( const Tuple<A...> &list ) {
    return list.apply_values( [&]( auto... values ) {
        return ( values + ... + Ct<int,0>() );
    } );
}

} // namespace sdot

#include "Tuple.cxx" // IWYU pragma: export
