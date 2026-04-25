#pragma once

#include "../support/SimpleSquareMatrix.h"
// #include "../support/P.h"
#include <numeric>
#include "Bsp.h"

namespace sdot {

template<class TF,class Arch,int ct_dim_>
struct BspMaker {
    Bsp<TF,Arch>          bsp;
    SI                    max_points_per_cell;
    TensorView<TF,2,Arch> positions;
    TensorView<TF,1,Arch> weights;
    CtInt<ct_dim_>        ct_dim;

    enum { degree_w_approx       = 1 };
    enum { ct_nb_coeffs_w_approx = 1 + ct_dim_ * ( degree_w_approx >= 1 ) + ct_dim_ * ( ct_dim_ + 1 ) / 2 * ( degree_w_approx >= 2 ) };
    using                 Pt = DsVec<TF,ct_dim_>;

    std::pair<DsVec<TF,ct_dim_>,DsVec<TF,ct_dim_>> min_max( SI beg_in_sorted_vertex_indices, SI end_in_sorted_vertex_indices ) {
        using namespace std;

        const PI ind = bsp.sorted_vertex_indices( beg_in_sorted_vertex_indices );
        DsVec<TF,ct_dim_> mi( positions.row( ind ) );
        DsVec<TF,ct_dim_> ma = mi;

        const PI dim = ct_dim_ >= 0 ? ct_dim_ : positions.size( 0 );
        for( SI i = beg_in_sorted_vertex_indices + 1; i < end_in_sorted_vertex_indices; ++i ) {
            const PI ind = bsp.sorted_vertex_indices( i );
            for( PI d = 0; d < dim; ++d ) {
                const TF v = positions( ind, d );
                mi[ d ] = min( mi[ d ], v );
                ma[ d ] = max( ma[ d ], v );
            }
        }

        return { mi, ma };
    }

    void update_rec( SI num_cell_to_update ) {
        using namespace std;

        // update min max
        const SI beg_si = bsp.cell_indices( num_cell_to_update, 2 );
        const SI end_si = bsp.cell_indices( num_cell_to_update, 3 );
        const auto [ min_pos, max_pos ] = min_max( beg_si, end_si );

        const PI dim = ct_dim_ >= 0 ? ct_dim_ : positions.size( 0 );
        for( int d = 0; d < dim; ++d ) {
            bsp.cell_bounds( num_cell_to_update, 0 * dim + d ) = min_pos[ d ];
            bsp.cell_bounds( num_cell_to_update, 1 * dim + d ) = min_pos[ d ];
        }

        // update_weights
        update_weights( num_cell_to_update, beg_si, end_si );

        //
        const SI nb_points = end_si - beg_si;
        if ( nb_points < 30 )
            return;

        //
        const auto dlt_pos = max_pos - min_pos;
        const PI dim_to_cut = dlt_pos.arg_max();

        Pt split_dir = Pt::value_at( dim, dim_to_cut, 1 );
        TF beg_dot = dot( split_dir, min_pos );
        TF end_dot = dot( split_dir, max_pos );

        std::vector<PI> bins( 256 );
        for( PI num_si = beg_si; num_si < end_si; ++num_si ) {
            const PI index = bsp.sorted_vertex_indices[ num_si ];
            TF p = dot( positions.row( index ), split_dir ) - beg_dot;
            p = bins.size() * p / ( end_dot - beg_dot );

            SI i = max( TF( 0 ), min( TF( bins.size() - 1 ), p ) );
            ++bins[ i ];
        }

        //
        TF split_dot = ( beg_dot + end_dot ) / 2;
        for( PI b = 0, acc = 0; b < bins.size(); ++b ) {
            if ( acc >= nb_points / 2 ) {
                split_dot = beg_dot + ( end_dot - beg_dot ) * b / bins.size();
                break;
            }
            acc += bins[ b ];
        }

        //
        PI beg = beg_si;
        PI end = end_si;
        while ( beg < end ) {
            const SI beg_index = bsp.sorted_vertex_indices[ beg ];
            const TF sp = dot( positions.row( beg_index ), split_dir ) - split_dot;
            if ( sp > 0 )
                std::swap( bsp.sorted_vertex_indices[ beg ], bsp.sorted_vertex_indices[ --end ] );
            else
                ++beg;
        }

        // all the points in the same place ?
        if ( beg == beg_si || beg == end_si )
            return;

        //
        const SI c0 = add_cell( beg_si, beg );
        const SI c1 = add_cell( beg, end_si );
        bsp.cell_indices( num_cell_to_update, 0 ) = c0;
        bsp.cell_indices( num_cell_to_update, 1 ) = c1;
        update_rec( c0 );
        update_rec( c1 );
    }

    SI add_cell( SI beg_in_sorted_vertex_indices, SI end_in_sorted_vertex_indices ) {
        SI res = bsp.nb_cells()++;
        if ( res == bsp.cell_indices.size( 0 ) )
            throw std::runtime_error( "reservation of cell_indices" );

        bsp.cell_indices( res, 0 ) = -1;
        bsp.cell_indices( res, 1 ) = -1;
        bsp.cell_indices( res, 2 ) = beg_in_sorted_vertex_indices;
        bsp.cell_indices( res, 3 ) = end_in_sorted_vertex_indices;

        return res;
    }

    void update_weights( SI num_cell_to_update, SI beg_si, SI end_si ) {
        const PI dim = ct_dim_ >= 0 ? ct_dim_ : positions.size( 0 );
        const int nb_coeffs_w_approx = 1 + dim * ( degree_w_approx >= 1 ) + dim * ( dim + 1 ) / 2 * ( degree_w_approx >= 2 );

        // M, V
        SimpleSquareMatrix<TF,ct_nb_coeffs_w_approx,Arch> M( Size(), nb_coeffs_w_approx, 0 );
        DsVec<TF,ct_nb_coeffs_w_approx,Arch> V( Size(), nb_coeffs_w_approx, 0 );
        for( SI num_si = beg_si; num_si < end_si; ++num_si ) {
            const PI index = bsp.sorted_vertex_indices[ num_si ];
            const TF w = weights.is_invalid() ? 0 : weights[ index ];
            const Pt p = positions.row( index );

            DsVec<TF,ct_nb_coeffs_w_approx,Arch> coeffs( Size(), nb_coeffs_w_approx );
            coeffs[ 0 ] = 1;
            if ( degree_w_approx >= 1 )
                for( std::size_t d = 0; d < dim; ++d )
                    coeffs[ 1 + d ] = p[ d ];
            if ( degree_w_approx >= 2 )
                for( std::size_t d = 0; d < dim; ++d )
                    for( std::size_t e = 0; e <= d; ++e )
                        coeffs[ 1 + dim + d * ( d + 1 ) / 2 + e ] = p[ d ] * p[ e ];

            for( int r = 0; r < nb_coeffs_w_approx; ++r ) {
                for( int c = 0; c <= r; ++c )
                    M( r, c ) += coeffs[ r ] * coeffs[ c ];
                V[ r ] += coeffs[ r ] * w;
            }
        }

        // solve dir
        for( int c = 0; c < nb_coeffs_w_approx; ++c )
            for( int r = 0; r < c; ++r )
                M( r, c ) = M( c, r );
        TF ad = 1e-10 * M.diagonal().max();
        for( int c = 0; c < nb_coeffs_w_approx; ++c )
            M( c, c ) += ad;

        const auto A = M.solve_ge( V );

        // update offset
        for( SI num_si = beg_si; num_si < end_si; ++num_si ) {
            const PI index = bsp.sorted_vertex_indices[ num_si ];
            TF w = weights.is_invalid() ? 0 : weights[ index ];
            const Pt p = positions.row( index );

            if ( degree_w_approx >= 1 )
                for( int d = 0; d < dim; ++d )
                    w -= V[ 1 + d ] * p[ d ];
            if ( degree_w_approx >= 2 )
                for( int d = 0; d < dim; ++d )
                    for( int e = 0; e <= d; ++e )
                        w -= V[ 1 + dim + d * ( d + 1 ) / 2 + e ] * p[ d ] * p[ e ];
            V[ 0 ] = std::max( V[ 0 ], w );
        }

        // store the coefficients
        for( int d = 0; d < nb_coeffs_w_approx; ++d )
            bsp.cell_bounds( num_cell_to_update, 2 * dim + d ) = V[ d ];
    }

    void exec() {
        // base init
        std::iota( bsp.sorted_vertex_indices.begin(), bsp.sorted_vertex_indices.end(), PI( 0 ) );
        bsp.nb_cells() = 0;

        // first cell
        add_cell( 0, bsp.sorted_vertex_indices.size() );

        // get all the sub cells
        update_rec( 0 );
    }
};

void make_bsp( auto &&bm ) {
    bm.exec();
}

} // namespace sdot
