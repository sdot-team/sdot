#pragma once

#include <sdot/generated_includes/PolynomialGrid.h>
#include <sdot/generated_includes/Cell.h>
#include <sdot/Cell/make_hypercube.h>
#include <sdot/Cell/CellBoundary.h>

namespace sdot {

template<int nb_coeffs, int dim,typename Arch,typename TF,typename TI,typename Knots>
struct PolynomialGridWorker {
    using Pgrid = PolynomialGrid<nb_coeffs,dim,Arch,TF,TI>;
    using Cell = sdot::Cell<dim,Arch,TF,TI>;

    static constexpr int _order() {
        // nb_coeffs == (order+1)^dim: find the integer dim-th root of nb_coeffs minus 1
        for ( int k = 1; k <= nb_coeffs; ++k ) {
            int p = 1;
            for ( int d = 0; d < dim; ++d )
                p *= k;
            if ( p >= nb_coeffs )
                return k - 1;
        }
        return -1;
    }

    static constexpr int order = _order();

    struct Polynomial{
        Polynomial() : coeffs( Size(), nb_coeffs ) {}
        DsVec<TF,nb_coeffs,Arch> coeffs; /** Q_k basis, lex multi-index (p0,p1,...) each in 0..order. Ex order=1 dim=2: 1 y x xy */
    };

    Polynomial polynomial( auto position ) const {
        Polynomial res;
        for( PI d = 0; d < nb_coeffs; ++d )
            res.coeffs[ d ] = pgrid.values( position, d );
        return res;
    }

    TF piece_integral( auto index, const Polynomial &pol ) const {
        // if ( has_skew_or_rotation() )
        //     TODO;

        // Q_k basis: coefficient c corresponds to multi-index (p0,p1,...,p_{d-1})
        // in lex order, each p_d in 0..order.
        // ∫_{cell} x^p dx = ∏_d ∫_{a_d}^{b_d} x_d^{p_d} dx_d  = ∏_d m_{p_d}[d]
        // where m_k[d] = (b_d^{k+1} - a_d^{k+1}) / (k+1)

        // Per-axis moments m[d][k] = ∫_{a_d}^{b_d} x^k dx,  k = 0..order
        DsVec<TF,dim * ( order + 1 ),Arch> m( Size(), dim * ( order + 1 ) );
        for( PI d = 0; d < dim; ++d ) {
            const TF a = knots( d, index[ d ] + 0 );
            const TF b = knots( d, index[ d ] + 1 );
            TF ak = a, bk = b;  // a^{k+1}, b^{k+1} starting at k=0 -> a^1, b^1
            for ( PI k = 0; k <= PI( order ); ++k ) {
                m[ d * ( order + 1 ) + k ] = ( bk - ak ) / TF( k + 1 );
                ak *= a;
                bk *= b;
            }
        }

        // Iterate over all Q_k multi-indices in lex order
        TF res = 0;
        // use a stack-based counter
        PI powers[ dim >= 0 ? dim : 1 ] = {};
        for ( PI c = 0;; ++c ) {
            // ∫ x^powers = ∏_d m[d][powers[d]]
            TF integ = 1;
            for ( PI d = 0; d < dim; ++d )
                integ *= m[ d * ( order + 1 ) + powers[ d ] ];
            res += pol.coeffs[ c ] * integ;

            // advance multi-index (lex, last axis fastest)
            PI d = dim;
            while ( d-- ) {
                if ( ++powers[ d ] <= PI( order ) )
                    break;
                powers[ d ] = 0;

                if ( d == 0 )
                    goto done;
            }
        }
        done:

        return res;
    }

    // void for_each_piece( auto &&func ) {
    // }

    TF mass() {
        TF res = 0;
        for_each_in_range( pgrid.values.shape().without_index( dim ), [&]( const auto &index ) {
            res += piece_integral( index, polynomial( index ) );
        } );
        return res;
    }

    void init_cell( Cell &cell ) {
        SimpleSquareMatrix<TF,dim+1,Arch> frame( Size(), dim + 1 );
        for( PI c = 0; c < dim; ++c )
            frame( 0, c ) = 0;
        for( PI r = 0; r < dim; ++r )
            for( PI c = 0; c < dim; ++c )
                frame( r + 1, c ) = ( r == c ) * knots( r, pgrid.values.shape( r ) );

        if ( pgrid.frame.is_valid() )
            TODO;

        make_hypercube( cell, frame, CellBoundary::BOUNDARY );
    }

    Pgrid pgrid;
    Knots knots;
};

template<int nb_coeffs,int dim,typename Arch,typename TF,typename TI>
auto with_worker_for( PolynomialGrid<nb_coeffs,dim,Arch,TF,TI> &polynomial_grid, auto &&func ) {
    if ( polynomial_grid.knots.is_valid() ) {
        PolynomialGridWorker pw( polynomial_grid, polynomial_grid.knots );
        return func( pw );
    }

    PolynomialGridWorker pw( polynomial_grid, []( PI, PI i ) { return i; } );
    return func( pw );
}



} // namespace sdot
