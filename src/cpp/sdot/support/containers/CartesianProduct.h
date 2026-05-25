#pragma once

#include "Tuple.h"
#include "Range.h"

namespace sdot {

template<class... Lists>
struct CartesianProducts;

template<class Head,class... Tail>
struct CartesianProducts<Head,Tail...> {
    using Next = CartesianProducts<Tail...>;

    CartesianProducts( auto &&head, auto &&...tail ) : head( FORWARD( head ) ), next( FORWARD( tail )... ) {
    }

    constexpr void for_each_item( auto &&func, auto &&...values_so_far ) const {
        head.for_each_item( [&]( auto &&value ) {
            next.for_each_item( func, values_so_far..., FORWARD( value ) );
        } );
    }

    constexpr auto nb_items() const {
        return head.nb_items() * next.nb_items();
    }

    Head head;
    Next next;
};

template<>
struct CartesianProducts<> {
    void for_each_item( auto &&func, auto &&...values_so_far ) const {
        func( tuple( FORWARD( values_so_far )... ) );
    }

    auto nb_items() const {
        return 1;
    }
};

auto cartesian_product_args( auto &&...lists ) {
    return CartesianProducts<DECAYED_TYPE_OF( lists )...>{ FORWARD( lists )... };
}

auto cartesian_product( auto &&tuple_of_lists ) {
    return tuple_of_lists.apply_values( [&]( auto &&...lists ) {
        return cartesian_product_args( FORWARD( lists )... );
    } );
}

auto cartesian_product_ranges( auto &&tuple_of_lists ) {
    return tuple_of_lists.apply_values( [&]( auto &&...lists ) {
        return cartesian_product_args( range( FORWARD( lists ) )... );
    } );
}

} // namespace sdot
