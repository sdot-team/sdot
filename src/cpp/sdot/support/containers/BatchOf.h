#pragma once

#include "AxisNames.h" // IWYU pragma: export
#include "Tuple.h"

namespace sdot {

/// One resolved batch axis: the name tag `Ax` and its runtime index. A `batch_index` handed to a
/// BatchOf (or any named accessor) is a `Tuple` of these — keyed by name (type), not by position.
template<class Ax> struct AxisIndex { PI index; };

template<class Ax> HD AxisIndex<Ax> axis_index( PI i ) { return { i }; }

namespace detail {
    /// locate `AxisIndex<Ax>` inside a `Tuple<...>` by type: presence + runtime value.
    template<class Ax,class Tup> struct find_axis_index;
    template<class Ax> struct find_axis_index<Ax,Tuple<>> {
        static constexpr bool present = false;
        static HD PI get( const Tuple<> & ) { return 0; }
    };
    template<class Ax,class Head,class... Tail>
    struct find_axis_index<Ax,Tuple<Head,Tail...>> {
        using Rest = find_axis_index<Ax,Tuple<Tail...>>;
        static constexpr bool here    = std::is_same_v<Head,AxisIndex<Ax>>;
        static constexpr bool present = here || Rest::present;
        static HD PI get( const Tuple<Head,Tail...> &t ) {
            if constexpr ( here ) return t.head.index;
            else                  return Rest::get( t.tail );
        }
    };
}

/// does `batch_index` (a Tuple of AxisIndex) carry the axis named `Ax` ?
template<class Ax,class Tup> static constexpr bool batch_index_has = detail::find_axis_index<Ax,Tup>::present;
/// runtime index of axis `Ax` in `batch_index` ( 0 if absent — callers gate with `batch_index_has` )
template<class Ax,class Tup> HD PI batch_index_get( const Tup &t ) { return detail::find_axis_index<Ax,Tup>::get( t ); }


// --- build a named batch_index from a positional one -----------------------------------------
//
// `cartesian_product_ranges` yields a plain positional `Tuple<PI,...>`; the generated
// `Parallel_*` functor turns it into a `Tuple<AxisIndex<Ax>...>` keyed by the axis names it was
// instantiated with (in iteration order). `named_batch_index<Ax...>( positional )` does that zip.
namespace detail {
    template<class Names,class BI> struct NamedIndex;
    template<class BI> struct NamedIndex<container_tags::AxisNames<>,BI> {
        static HD Tuple<> get( const BI & ) { return {}; }
    };
    template<class A0,class... Ar,class Head,class... Tail>
    struct NamedIndex<container_tags::AxisNames<A0,Ar...>,Tuple<Head,Tail...>> {
        static HD auto get( const Tuple<Head,Tail...> &bi ) {
            return concat( tuple( axis_index<A0>( bi.head ) ),
                           NamedIndex<container_tags::AxisNames<Ar...>,Tuple<Tail...>>::get( bi.tail ) );
        }
    };
}

template<class... Ax,class BI>
HD auto named_batch_index( const BI &positional ) {
    return detail::NamedIndex<container_tags::AxisNames<Ax...>,BI>::get( positional );
}


/// Resolve the named axes `Names...` present in `batch_index` by squeezing `value` *by name*, one
/// axis at a time. An axis absent from `batch_index` is left untouched (broadcast) — it stays a
/// named axis of the result, addressable later (e.g. by a hand-written deeper loop). `value` only
/// needs `squeeze( name, index )`: a `TensorView` / `DynamicAxis` has it natively; a batched
/// aggregate gets a generated one that squeezes the name across all its fields.
template<class U,class BI>
HD auto peel_named_axes( U value, const BI &, container_tags::AxisNames<> ) { return value; }

template<class U,class BI,class A0,class... Ar>
HD auto peel_named_axes( U value, const BI &bi, container_tags::AxisNames<A0,Ar...> ) {
    if constexpr ( batch_index_has<A0,BI> )
        return peel_named_axes( value.squeeze( A0{}, batch_index_get<A0>( bi ) ), bi, container_tags::AxisNames<Ar...>{} );
    else
        return peel_named_axes( value, bi, container_tags::AxisNames<Ar...>{} ); // broadcast: leave A0 on the result
}


/// A generic "batch of `Underlying`" carrying named batch axes `Ax...` (e.g. `BatchOf<TensorView…,
/// ax_n>`). Used to wrap a *raw* batched field ( a tensor / dynamic axis whose own `operator()`
/// would otherwise consume the index positionally ), giving it a uniform named `operator()`. A
/// batched aggregate is not wrapped — it carries its own named `operator()` directly.
template<class Underlying,class... Ax>
struct BatchOf {
    Underlying underlying;

    template<class BI>
    HD auto operator()( const BI &batch_index ) const {
        return peel_named_axes( underlying, batch_index, container_tags::AxisNames<Ax...>{} );
    }

    /// transparency for the apply_values-based infrastructure (transfer_cost, data movement, …):
    /// a BatchOf is seen through to its single underlying value.
    template<class F> HD auto apply_values( F &&func ) const { return func( underlying ); }
    template<class F> HD auto apply_values( F &&func )       { return func( underlying ); }
};

} // namespace sdot
