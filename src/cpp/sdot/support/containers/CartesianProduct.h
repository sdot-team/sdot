#pragma once

#include "for_each_item_split.h"
#include "Tuple.h"
#include "Range.h"

#include <type_traits>

namespace sdot {

template<class... Lists>
struct CartesianProducts;

namespace CartesianProductDetail {
    // for_each_item recursion step (functor, not a lambda: nvcc forbids generic / by-ref extended
    // __host__ __device__ lambdas). Appends the current `value` to the accumulated multi-index
    // `acc` (a Tuple) and recurses into the remaining axes.
    template<class Next,class Func,class Acc>
    struct ForEachStep {
        const Next &next;
        Func       &func;
        Acc         acc;
        HD void operator()( auto &&value ) const {
            next.for_each_item( func, acc.with_appended_value( FORWARD( value ) ) );
        }
    };

    // builders used by cartesian_product / cartesian_product_ranges (functors over apply_values)
    struct MakeProduct       { HD auto operator()( auto &&...lists ) const; };
    struct MakeProductRanges { HD auto operator()( auto &&...lists ) const; };
}

template<class Head,class... Tail>
struct CartesianProducts<Head,Tail...> {
    using Next = CartesianProducts<Tail...>;

    HD CartesianProducts( const CartesianProducts & ) = default;
    HD CartesianProducts( CartesianProducts && ) = default;

    // forwarding ctor — constrained so it does not hijack copy/move construction
    template<class H,class... T> requires ( ! std::is_same_v<std::decay_t<H>,CartesianProducts> )
    HD CartesianProducts( H &&head, T &&...tail ) : head( FORWARD( head ) ), next( FORWARD( tail )... ) {
    }

    // call func( multi_index ) for each combination; `acc` accumulates the indices chosen so far
    // HD void for_each_item_split( auto &&func, PI rel, PI mod, auto acc ) const {
    //     sdot::for_each_item( head, CartesianProductDetail::ForEachStep<Next,DECAYED_TYPE_OF( func ),DECAYED_TYPE_OF( acc )>{ next, func, acc } );
    // }

    // HD void for_each_item_split( auto &&func, PI rel, PI mod ) const {
    //     for_each_item_split( FORWARD( func ), tuple() );
    // }

    // call func( multi_index ) for each combination; `acc` accumulates the indices chosen so far
    HD void for_each_item( auto &&func, auto acc ) const {
        sdot::for_each_item( head, CartesianProductDetail::ForEachStep<Next,DECAYED_TYPE_OF( func ),DECAYED_TYPE_OF( acc )>{ next, func, acc } );
    }

    HD void for_each_item( auto &&func ) const {
        for_each_item( FORWARD( func ), tuple() );
    }

    constexpr auto nb_items() const {
        return head.nb_items() * next.nb_items();
    }

    Head head;
    Next next;
};

template<>
struct CartesianProducts<> {
    HD void for_each_item( auto &&func, auto acc ) const {
        func( acc );
    }

    HD void for_each_item( auto &&func ) const {
        for_each_item( FORWARD( func ), tuple() );
    }

    constexpr auto nb_items() const {
        return 1;
    }
};

HD auto cartesian_product_args( auto &&...lists ) {
    return CartesianProducts<DECAYED_TYPE_OF( lists )...>{ FORWARD( lists )... };
}

HD auto cartesian_product( auto &&tuple_of_lists ) {
    return tuple_of_lists.apply_values( CartesianProductDetail::MakeProduct{} );
}

HD auto cartesian_product_ranges( auto &&tuple_of_lists ) {
    return tuple_of_lists.apply_values( CartesianProductDetail::MakeProductRanges{} );
}

namespace CartesianProductDetail {
    HD auto MakeProduct::operator()( auto &&...lists ) const { return cartesian_product_args( FORWARD( lists )... ); }
    HD auto MakeProductRanges::operator()( auto &&...lists ) const { return cartesian_product_args( range( FORWARD( lists ) )... ); }
}

} // namespace sdot
