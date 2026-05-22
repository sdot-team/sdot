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

    HD             Tuple              ( Function, auto &&func, auto index ); ///< allows for starting with index != 0
    HD             Tuple              ( Function, auto &&func );

    HD             Tuple              ( Values, auto head, auto... tail );
    T_VT HD        Tuple              ( const Tuple<T...> &that );
    HD             Tuple              ( const Tuple &that ) = default;
    HD             Tuple              () = default;

    HD void        for_each_item      ( auto &&cb_func ) const;
    HD void        for_each_item      ( auto &&cb_func );
    HD auto        apply_values       ( auto &&cb_func ) const;
    HD auto        operator[]         ( auto &&index ) const;
    HD auto        operator==         ( const auto &that ) const;
    static HD auto size               ();

    HD auto        with_appended_value( auto &&new_value ) const;
    HD auto        without_index      ( auto index ) const;

    Head           head;
    Next           tail;
};

/// known ct_rank == 0
template<>
class Tuple<> {
public:
    SCInt          ct_size            = 0;

    HD             Tuple              ( Function, auto &&/*func*/, auto /*index*/ );
    HD             Tuple              ( Function, auto &&/*func*/ );
    HD             Tuple              ( const Tuple &that ) = default;
    HD             Tuple              ( Values );
    HD             Tuple              () = default;

    HD void        for_each_item      ( auto &&/* cb */ ) const;
    HD auto        apply_values       ( auto &&cb ) const;
    HD Void        operator[]         ( auto ) const;
    HD auto        operator==         ( const auto &that ) const;
    static HD auto size               ();

    HD auto        with_appended_value( auto &&new_value ) const;
};

constexpr auto tuple( auto &&...a ) {
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

constexpr auto map( auto &&list, auto &&func ) { // requires requires { list.apply_values( []( auto... ) {} ); } {
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
