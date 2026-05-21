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
    using        Next            = Tuple<Tail...>;

    HD           Tuple           ( Function, auto &&func, auto index ); ///< allows for starting with index != 0
    HD           Tuple           ( Function, auto &&func );

    HD           Tuple           ( Values, auto head, auto... tail );
    HD           Tuple           ( const Tuple &that );
    HD           Tuple           ();

    HD void      for_each_item   ( auto &&cb ) const { cb( head ); tail.for_each_item( FORWARD( cb ) ); }
    HD auto      apply_values    ( auto &&cb ) const;
    HD auto      operator[]      ( auto &&index ) const;
    HD auto      size            () const;

    HD auto      without_index   ( auto index ) const;
    HD auto      all_indices     () const; ///< the list of all multi-indices, for run_*()

    Head         head;
    Next         tail;
};

/// known ct_rank == 0
template<>
class Tuple<> {
public:
    HD           Tuple           ( Function, auto &&/*func*/, auto /*index*/ );
    HD           Tuple           ( Function, auto &&/*func*/ );
    HD           Tuple           ( const Tuple &that );
    HD           Tuple           ( Values );
    HD           Tuple           ();

    HD void      for_each_item   ( auto &&/* cb */ ) const;
    HD auto      apply_values    ( auto &&cb ) const;
    HD Void      operator[]      ( auto ) const;
    HD auto      size            () const;
};

auto tuple( auto &&...a ) {
    return Tuple<DECAYED_TYPE_OF( a )...>( Values(), a... );
}

template<class... A,class... B>
auto concat( const Tuple<A...> &a, const Tuple<B...> &b ) {
    return a.apply_values( [&]( auto... va ) {
        return b.apply_values( [&]( auto... vb ) {
            return tuple( va..., vb... );
        } );
    } );
}

} // namespace sdot

#include "Tuple.cxx" // IWYU pragma: export
