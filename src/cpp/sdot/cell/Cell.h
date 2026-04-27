#pragma once

#include "../support/SimpleSquareMatrix.h"
#include "../support/DynamicAxis.h"
#include "../support/TensorView.h"
// #include "../support/P.h"

namespace sdot {

///
template<class TF,class Arch>
struct Cell {
    static constexpr SI   INFINITE          = -2;
    static constexpr SI   BOUNDARY          = -1;

    PI                    dim               () const { return vertex_positions.size( 1 ); }

    TensorView<TF,2,Arch> vertex_positions; ///< ( vertex_capacity, dim )
    TensorView<SI,2,Arch> vertex_indices;   ///< ( vertex_capacity, dim ) : sorted cut indices for each vertex
    TensorView<SI,2,Arch> edge_indices;     ///< ( edge_capacity, dim + 1 ) : edge index -> vertex indices (vertex on each side) + cut_indices
    TensorView<TF,2,Arch> cut_planes;       ///< ( cut_capacity, dim + 1 )
    TensorView<SI,1,Arch> cut_ids;          ///< ( cut_capacity )

    TensorView<SI,0,Arch> is_fully_closed;

    DynamicAxis<0,Arch>   nb_vertices;
    DynamicAxis<0,Arch>   nb_edges;
    DynamicAxis<0,Arch>   nb_cuts;
};

template<class TF,class Arch>
struct BatchOfCell {
    Cell<TF,Arch> row( PI batch_index ) {
        return {
            vertex_positions.row( batch_index ),
            vertex_indices.row( batch_index ),
            edge_indices.row( batch_index ),
            cut_planes.row( batch_index ),
            cut_ids.row( batch_index ),
            is_fully_closed.row( batch_index ),
            nb_vertices.row( batch_index ),
            nb_edges.row( batch_index ),
            nb_cuts.row( batch_index )
        };
    }

    TensorView<TF,3,Arch> vertex_positions; ///< ( batch_index, vertex_capacity, dim )
    TensorView<SI,3,Arch> vertex_indices;   ///< ( batch_index, vertex_capacity, dim ) : sorted cut indices for each vertex
    TensorView<SI,3,Arch> edge_indices;     ///< ( batch_index, edge_capacity, dim + 1 ) : edge index -> vertex indices (vertex on each side) + cut_indices
    TensorView<TF,3,Arch> cut_planes;       ///< ( batch_index, cut_capacity, dim + 1 )
    TensorView<SI,2,Arch> cut_ids;          ///< ( batch_index, cut_capacity )

    TensorView<SI,1,Arch> is_fully_closed;

    DynamicAxis<1,Arch>   nb_vertices;
    DynamicAxis<1,Arch>   nb_edges;
    DynamicAxis<1,Arch>   nb_cuts;
};

// #define UTP2 template<class TF,class Arch>
// #define DTP2 Cell<TF,2,Arch>

// #define UTP1 template<class TF,class Arch>
// #define DTP1 Cell<TF,2,Arch>

#define UTP template<class TF,class Arch,int ct_dim>
#define DTP Cell<TF,Arch>


UTP void make_aligned_simplex( DTP &cell, SI cut_id, CtInt<ct_dim> ) {
    const PI dim = cell.vertex_positions.size( 1 );
    const PI nb_edges = ( dim + 1 ) * dim / 2;
    const PI nb_vertices = dim + 1;
    const PI nb_cuts = dim + 1;

    cell.is_fully_closed() = cut_id != DTP::INFINITE;
    cell.nb_vertices() = nb_vertices;
    cell.nb_edges() = nb_edges;
    cell.nb_cuts() = nb_cuts;

    // vertex_positions
    for( PI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex )
        for( PI d = 0; d < dim; ++d )
            cell.vertex_positions( num_vertex, d ) = ( d + 1 == num_vertex );

    // vertex_inds
    if ( dim != 2 )
        for( PI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex )
            for( PI d = 0; d < dim; ++d )
                cell.vertex_indices( num_vertex, d ) = d + ( d >= num_vertex );

    // edge_indices
    if ( dim != 2 ) {
        for ( PI a = 0, o = 0; a < nb_vertices; ++a ) {
            for ( PI b = a + 1; b < nb_vertices; ++b ) {
                if ( a != b ) {
                    cell.edge_indices( o, 0 ) = a;
                    cell.edge_indices( o, 1 ) = b;
                    for( PI d = 0; d < dim - 1; ++d )
                        cell.edge_indices( o, 2 + d ) = d + ( d >= a ) + ( d >= b - 1 );
                    ++o;
                }
            }
        }
    }

    // cut_planes
    for( PI num_cut = 0; num_cut < dim; ++num_cut ) {
        for( PI d = 0; d < dim; ++d )
            cell.cut_planes( num_cut, d ) = - ( d == num_cut );
        cell.cut_planes( num_cut, dim ) = 0;
    }
    for( PI d = 0; d < dim + 1; ++d )
        cell.cut_planes( dim, d ) = 1;

    // cut_ids
    for( PI num_cut = 0; num_cut < nb_cuts; ++num_cut )
        cell.cut_ids( num_cut ) = cut_id;
}

void make_empty_cell( auto &&p ) {
    make_aligned_simplex( p.cell, p.cell.INFINITE, p.ct_dim );
}

void make_empty_cell_backward( auto && ) {
    TODO; // make_aligned_simplex( p.cell, p.cell.INFINITE );
}

UTP void make_hypercube( DTP &cell, const auto &frame, SI cut_id, CtInt<ct_dim> ) {
    const PI dim = cell.vertex_positions.size( 1 );
    const PI nb_vertices = PI( 1 ) << dim;

    cell.is_fully_closed() = cut_id != DTP::INFINITE;
    cell.nb_vertices() = nb_vertices;
    cell.nb_edges() = dim * ( PI( 1 ) << ( dim - 1 ) );
    cell.nb_cuts() = 2 * dim;

    // shared: F^T[r][c] = axis_c[r], used to compute rows of F^{-1} via solve_ge
    auto FT = SimpleSquareMatrix<TF,ct_dim,Arch>::with_func( dim, [&]( PI r, PI c ) {
        return TF( frame( 1 + c, r ) );
    } );

    // vertex_positions: origin + sum of selected axes; vertex_indices: cut 2b or 2b+1 per axis
    const PI vertex_ordering_2D[] = { 0, 1, 3, 2 };
    for ( PI k = 0; k < nb_vertices; ++k ) {
        const PI l = ( dim != 2 ? k : vertex_ordering_2D[ k ] );
        for ( PI d = 0; d < dim; ++d ) {
            TF pos = frame( 0, d );
            for ( PI b = 0; b < dim; ++b )
                if ( ( k >> b ) & 1 )
                    pos += frame( 1 + b, d );
            cell.vertex_positions( l, d ) = pos;
        }
    }

    // vertex_indices
    if ( dim >= 2 ) {
        for ( PI k = 0; k < nb_vertices; ++k )
            for ( PI b = 0; b < dim; ++b )
                cell.vertex_indices( k, b ) = 2 * b + ( ( k >> b ) & 1 );
    }

    // edge_indices: edges in direction b, from vertex k (bit b=0) to k|(1<<b)
    if ( dim >= 2 ) {
        for ( PI b = 0, e = 0; b < dim; ++b ) {
            for ( PI k = 0; k < nb_vertices; ++k ) {
                if ( ( k >> b ) & 1 )
                    continue;
                cell.edge_indices( e, 0 ) = k;
                cell.edge_indices( e, 1 ) = k | ( PI( 1 ) << b );
                for ( PI d = 0, col = 2; d < dim; ++d ) {
                    if ( d == b )
                        continue;
                    cell.edge_indices( e, col++ ) = 2 * d + ( ( k >> d ) & 1 );
                }
                ++e;
            }
        }
    }

    // cut planes: row d of F^{-1} via shared FT
    const PI cut_ordering_2D[] = { 3, 1, 0, 2 };
    for ( PI d = 0; d < dim; ++d ) {
        auto e_d = DsVec<TF,ct_dim,Arch>::with_func( dim, [d]( PI i ) {
            return i == d ? TF( 1 ) : TF( 0 );
        } );
        const auto row = FT.solve_ge( e_d );

        TF row_dot_origin = 0;
        for ( PI c = 0; c < dim; ++c )
            row_dot_origin += row[ c ] * frame( 0, c );

        const PI r0 = ( dim != 2 ? 2 * d + 0 : cut_ordering_2D[ 2 * d + 0 ] );
        for ( PI c = 0; c < dim; ++c )
            cell.cut_planes( r0, c ) = -row[ c ];
        cell.cut_planes( r0, dim ) = -row_dot_origin;
        cell.cut_ids( r0 ) = cut_id;

        const PI r1 = ( dim != 2 ? 2 * d + 1 : cut_ordering_2D[ 2 * d + 1 ] );
        for ( PI c = 0; c < dim; ++c )
            cell.cut_planes( r1, c ) = row[ c ];
        cell.cut_planes( r1, dim ) = row_dot_origin + 1;
        cell.cut_ids( r1 ) = cut_id;
    }
}

UTP void make_hypercube_backward( const DTP &cell, const auto &frame, SI /*cut_id*/, CtInt<ct_dim>, auto &grad_inp_frame, const auto &grad_out_vertex_positions, const auto &grad_out_cut_planes ) {
    const PI dim = cell.dim();
    const PI nb_vertices = PI(1) << dim;

    // R = FT^{-1} (same FT as in forward)
    const auto FT = SimpleSquareMatrix<TF,ct_dim,Arch>::with_func( dim, [&]( PI r, PI c ) {
        return TF( frame( 1 + c, r ) );
    } );
    const auto R = FT.inverse();

    // --- vertex positions backward ---
    const PI vertex_ordering_2D[] = { 0, 1, 3, 2 };
    for ( PI k = 0; k < nb_vertices; ++k ) {
        const PI l = ( dim != 2 ? k : vertex_ordering_2D[ k ] );
        for ( PI d_coord = 0; d_coord < dim; ++d_coord ) {
            const TF g = grad_out_vertex_positions( l, d_coord );
            grad_inp_frame( 0, d_coord ) += g;
            for ( PI b = 0; b < dim; ++b )
                if ( ( k >> b ) & 1 )
                    grad_inp_frame( 1 + b, d_coord ) += g;
        }
    }

    // --- cut planes backward ---
    // tG_R[d,j] = total gradient w.r.t. R[d,j] = row d of F^{-1}
    const PI cut_ordering_2D[] = { 3, 1, 0, 2 };
    SimpleSquareMatrix<TF,ct_dim,Arch> tG_R( dim );
    for ( PI d = 0; d < dim; ++d ) {
        const PI r0 = ( dim != 2 ? 2 * d     : cut_ordering_2D[ 2 * d     ] );
        const PI r1 = ( dim != 2 ? 2 * d + 1 : cut_ordering_2D[ 2 * d + 1 ] );

        const TF G_d = grad_out_cut_planes( r1, dim ) - grad_out_cut_planes( r0, dim );

        // gradient to origin via d_d = R[d,:] · origin
        for ( PI c = 0; c < dim; ++c )
            grad_inp_frame( 0, c ) += G_d * R( d, c );

        // tG_R[d,j] = direct gradient from ±R[d,j] in cut_planes + G_d * origin[j]
        for ( PI j = 0; j < dim; ++j )
            tG_R( d, j ) = ( grad_out_cut_planes( r1, j ) - grad_out_cut_planes( r0, j ) )
                         + G_d * frame( 0, j );
    }

    // grad_FT = -R^T · tG_R · R^T,  i.e. grad_frame(1+c, r) = grad_FT[r,c] = -Σ_d R[d,r] · M[d,c]
    // where M[d,c] = tG_R[d,:] · R[c,:]
    SimpleSquareMatrix<TF,ct_dim,Arch> M( dim );
    for ( PI d = 0; d < dim; ++d )
        for ( PI c = 0; c < dim; ++c ) {
            TF m = 0;
            for ( PI j = 0; j < dim; ++j )
                m += tG_R( d, j ) * R( c, j );
            M( d, c ) = m;
        }
    for ( PI c = 0; c < dim; ++c )
        for ( PI r = 0; r < dim; ++r ) {
            TF g = 0;
            for ( PI d = 0; d < dim; ++d )
                g -= R( d, r ) * M( d, c );
            grad_inp_frame( 1 + c, r ) += g;
        }
}


#undef UTP
#undef DTP

} // namespace sdot
