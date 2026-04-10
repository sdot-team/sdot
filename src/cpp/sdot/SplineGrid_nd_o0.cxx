#pragma once

#include "SplineGrid_nd_o0.h"

namespace sdot {

#define UTP template<class TF,int ct_dim,class Arch>
#define DTP SplineGrid<TF,ct_dim,0,Arch>

UTP DTP::SplineGrid( Values values, Bounds bounds, const std::vector<Knots> & ) : values( values ), bounds( bounds ), pif( values.rank() ), pf( values.rank() ) {
    TF mean = 0, c = 0;
    for_each_in_range( pif.zeros(), Pi( values.shape() ), [&]( const auto &index ) {
        mean += values[ index ];
        c += 1;
    } );
    mean /= c;

    coeff_values = 1 / mean;
}

UTP void DTP::_for_each_piece( const auto &cell, const auto &func, auto beg, auto end, auto &cur, PI d ) const {
    if ( d == dim() ) {
        func( cell, coeff_values * values[ cur ] );
        return;
    }

    for( PI v = beg[ d ]; v < end[ d ]; ++v ) {
        auto new_cell = cell;

        new_cell.cut( pf.value_at( d, - 1 ), - knot( d, v + 0 ), {} );
        new_cell.cut( pf.value_at( d, + 1 ), + knot( d, v + 1 ), {} );

        cur[ d ] = v;
        _for_each_piece( new_cell, func, beg, end, cur, d + 1 );
    }
}

UTP TF DTP::knot( PI d, PI index ) const {
    return TF( index ) / values.size( d );
}

UTP void DTP::for_each_piece( const auto &cell, auto &&func ) const {
    using namespace std;
    if ( ! bounds.empty() )
        TODO;

    // FP boundaries
    const Pt mi = cell.min_pos( pf.ones() );
    const Pt ma = cell.max_pos( pf.zeros() );
    for( PI d = 0; d < dim(); ++d )
        if ( ma[ d ] <= mi[ d ] )
            return;

    // I boundaries
    sdot::PointFactory<PI,ct_dim,Arch> pif( dim() );
    auto b = pif.with_func( [&]( PI d ) { return floor( mi[ d ] * values.size( d ) ); } );
    auto e = pif.with_func( [&]( PI d ) { return ceil( ma[ d ] * values.size( d ) ); } );

    // traversal
    auto c = b;
    _for_each_piece( cell, func, b, e, c, 0 );
}

UTP auto DTP::facet_integral( auto facet, auto value ) const {
    return value * facet.measure();
}

UTP auto DTP::integral( auto cell, auto value ) const {
    return value * cell.measure();
}

UTP T_U auto DTP::base_cell( PI dim, typename Cell<U,ct_dim,Arch>::CellInfo cell_info, typename Cell<U,ct_dim,Arch>::CutInfo cut_info ) const {
    return Cell<U,ct_dim,Arch>::axis_aligned_hypercube( pf.zeros(), pf.ones(), cell_info, cut_info );
}

#undef UTP
#undef DTP

} // namespace sdot
