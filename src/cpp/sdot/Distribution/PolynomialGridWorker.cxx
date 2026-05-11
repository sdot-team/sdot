#pragma once

#include "PolynomialGridWorker.h"
#include "../Cell/CellWorker.h"

namespace sdot {

#define UTP template<int nb_coeffs, int dim,typename Arch,typename TF,typename TI,typename Knots>
#define DTP PolynomialGridWorker<nb_coeffs,dim,Arch,TF,TI,Knots>

UTP void DTP::with_preparation_for_cell_traversal( auto &batch_of_cell_workspace, auto &batch_of_cells, auto &&func ) {
    batch_of_cell_workspace.with_same_shape( [&]( auto &batch_of_cell_workspace ) {
        batch_of_cells.with_same_shape( [&]( auto &batch_of_cells ) {
            DTP npw( pgrid, knots, &batch_of_cell_workspace, &batch_of_cells );
            func( npw );
        } );
    } );
}

UTP void DTP::for_each_sub_cell( const auto &cell, TI batch_index, auto &&func ) {
    CellWorkspace new_cell_workspace = batch_of_cell_workspace->row( batch_index );
    Cell new_cell = batch_of_cell->row( batch_index );

    CellWorker<dim,Arch,TF,TI> new_cw( new_cell, new_cell_workspace, dim );

    // ( auto &cell_worker, const auto &local_function )
    for_each_index( [&]( const Vector<PI,dim,Arch> &index ) {
        new_cw.get_data_from( cell );

        for( TI d = 0; d < dim; ++d ) {
            auto cp = cut_plane( d );
            new_cw.cut( cp, cut_offset( d, index + Vector<PI,dim,Arch>::value_at( dim, d, 1 ) ), CellBoundary::BOUNDARY );
            new_cw.cut( - cp, - cut_offset( d, index ), CellBoundary::BOUNDARY );
        }

        func( new_cw, polynomial( index ) );
    } );

}

UTP DTP::Pt DTP::cut_plane( TI d ) const {
    return Pt::value_at( dim, d, TF( 1 ) );
}

UTP TF DTP::cut_offset( TI d, Pi index ) {
    return knots( d, index[ d ] );
}


UTP void DTP::for_each_index( auto &&func ) const {
    VectorFactory<PI,dim,Arch> pf( dim );
    if ( dim == 0 ) {
        func( pf.zeros() );
        return;
    }

    Vector<PI,dim,Arch> index = pf.zeros();
    while ( true ) {
        func( index );

        PI n = dim - 1;
        while ( ++index[ n ] == pgrid.values.size( n ) ) {
            if ( n == 0 )
                return;
            index[ n ] = 0;
            --n;
        }
    }
}

UTP DTP::Polynomial DTP::polynomial( auto position ) const {
    Polynomial res;
    for( TI d = 0; d < nb_coeffs; ++d )
        res.coeffs[ d ] = pgrid.values( position, d );
    return res;
}

UTP TF DTP::piece_integral( auto index, const Polynomial &pol ) const {
    // if ( has_skew_or_rotation() )
    //     TODO;

    // Q_k basis: coefficient c corresponds to multi-index (p0,p1,...,p_{d-1})
    // in lex order, each p_d in 0..order.
    // ∫_{cell} x^p dx = ∏_d ∫_{a_d}^{b_d} x_d^{p_d} dx_d  = ∏_d m_{p_d}[d]
    // where m_k[d] = (b_d^{k+1} - a_d^{k+1}) / (k+1)

    // Per-axis moments m[d][k] = ∫_{a_d}^{b_d} x^k dx,  k = 0..order
    Vector<TF,Arch,dim * ( order + 1 ),Arch> m( Size(), dim * ( order + 1 ) );
    for( TI d = 0; d < dim; ++d ) {
        const TF a = knots( d, index[ d ] + 0 );
        const TF b = knots( d, index[ d ] + 1 );
        TF ak = a, bk = b;  // a^{k+1}, b^{k+1} starting at k=0 -> a^1, b^1
        for ( TI k = 0; k <= PI( order ); ++k ) {
            m[ d * ( order + 1 ) + k ] = ( bk - ak ) / TF( k + 1 );
            ak *= a;
            bk *= b;
        }
    }

    // Iterate over all Q_k multi-indices in lex order
    TF res = 0;
    // use a stack-based counter
    TI powers[ dim >= 0 ? dim : 1 ] = {};
    for ( PI c = 0;; ++c ) {
        // ∫ x^powers = ∏_d m[d][powers[d]]
        TF integ = 1;
        for ( TI d = 0; d < dim; ++d )
            integ *= m[ d * ( order + 1 ) + powers[ d ] ];
        res += pol.coeffs[ c ] * integ;

        // advance multi-index (lex, last axis fastest)
        TI d = dim;
        while ( d-- ) {
            if ( ++powers[ d ] <= TI( order ) )
                break;
            powers[ d ] = 0;

            if ( d == 0 )
                goto done;
        }
    }
    done:

    return res;
}

UTP TF DTP::mass() {
    TF res = 0;
    for_each_in_range( pgrid.values.shape().without_index( dim ), [&]( const auto &index ) {
        res += piece_integral( index, polynomial( index ) );
    } );
    return res;
}

UTP void DTP::init_cell( Cell &cell ) {
    SimpleSquareMatrix<TF,dim+1,Arch> frame( Size(), dim + 1 );
    for( TI c = 0; c < dim; ++c )
        frame( 0, c ) = 0;
    for( TI r = 0; r < dim; ++r )
        for( TI c = 0; c < dim; ++c )
            frame( r + 1, c ) = ( r == c ) * knots( r, pgrid.values.shape( r ) );

    if ( pgrid.frame.is_valid() )
        TODO;

    make_hypercube( cell, frame, CellBoundary::BOUNDARY );
}

#undef UTP
#undef DTP

} // namespace sdot
