#pragma once

#include "../common_types.h" // Values, HD, FORWARD

namespace sdot {

// AxisValues is only named in make_item (a template, instantiated where AxisValues is
// complete); a forward declaration is enough here and avoids an include cycle with
// AxisValues.h, whose all_indices() returns an IndexRange.
template<class TI,int ct_rank,class... attributes> class AxisValues;

/// Iterable set of all the multi-indices of a shape — the "list" consumed by run_*().
///
/// Obtained from `shape.all_indices()`. The shape (an AxisValues) only knows the axis
/// sizes; walking the index space lives here. Each item is a multi-index, materialized
/// as an AxisValues whose values are the coordinates (so it plugs directly into
/// `TensorView::operator()`, which accepts any size()/[]/without_index multi-index).
template<class Shape>
struct IndexRange {
    using TI = typename Shape::TI;

    Shape shape;

    /// total number of multi-indices (product of the axis sizes; 1 for a rank-0 shape)
    HD auto nb_items() const {
        return shape.apply_values( [&]( auto ...sizes ) { return ( TI( 1 ) * ... * TI( sizes ) ); } );
    }

    /// Visit every multi-index in row-major order, calling `func( multi_index )` for each.
    HD void for_each_item( auto &&func ) const { _rec( shape, func ); }

    /// Visit the multi-indices handled by thread `index` out of `size` threads, calling
    /// `func( multi_index )` for each. Items are spread round-robin (item k -> thread k % size).
    HD void for_each_item_split( int index, int size, auto &&func ) const {
        if constexpr ( Shape::ct_rank == 0 ) {
            // single item (the empty multi-index) -> only thread 0 runs it
            if ( index == 0 )
                func( make_item() );
        } else if constexpr ( Shape::ct_rank == 1 ) {
            // strided walk: no per-item modulo, each thread starts at its offset
            for ( TI i = index; i < shape.front_value; i += size )
                func( make_item( i ) );
        } else {
            // general case: walk the whole space, keep the items owned by this thread
            TI cpt = 0;
            _rec( shape, [&]( auto &&item ) {
                if ( cpt++ % TI( size ) == TI( index ) )
                    func( FORWARD( item ) );
            } );
        }
    }

    /// build a multi-index AxisValues from a pack of coordinates (rank = nb of coordinates)
    HD static auto make_item( auto ...coords ) {
        return AxisValues<TI,int( sizeof...( coords ) )>( Values(), coords... );
    }

private:
    /// recurse over the axes of `sub`, accumulating coordinates, then yield each full item
    template<class Sub>
    HD static void _rec( const Sub &sub, auto &&func, auto ...coords ) {
        if constexpr ( Sub::ct_rank == 0 )
            func( make_item( coords... ) );
        else
            for ( TI i = 0; i < sub.front_value; ++i )
                _rec( sub.next_values, func, coords..., i );
    }
};

} // namespace sdot
