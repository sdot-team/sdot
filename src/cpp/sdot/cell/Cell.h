#pragma once

#include "../support/SimpleSquareMatrix.h"
#include "../support/TensorView.h"
#include "../support/P.h"

// #define SDOT_KEEP_FULL_CELL_INFO_FOR_2D_CASE

namespace sdot {

///
template<class TF,int ct_dim,class Arch>
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
    TensorView<SI,0,Arch> nb_vertices;
    TensorView<SI,0,Arch> nb_edges;
    TensorView<SI,0,Arch> nb_cuts;
};

#define UTP2 template<class TF,class Arch>
#define DTP2 Cell<TF,2,Arch>

#define UTP1 template<class TF,class Arch>
#define DTP1 Cell<TF,2,Arch>

#define UTP template<class TF,int ct_dim,class Arch>
#define DTP Cell<TF,ct_dim,Arch>

UTP void make_aligned_simplex( DTP &cell, SI cut_id ) {
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

UTP void make_empty_cell( DTP &cell ) {
    make_aligned_simplex( cell, DTP::INFINITE );
}

UTP void make_hypercube( DTP &cell, const auto &frame, SI cut_id ) {
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

    if ( dim == 2 ) {
        // vertices CCW: bitmasks {0,1,3,2} → (0,0),(1,0),(1,1),(0,1)
        const PI bitmasks[ 4 ] = { 0, 1, 3, 2 };
        for ( PI k = 0; k < 4; ++k ) {
            for ( PI d = 0; d < 2; ++d ) {
                TF pos = frame( 0, 0 );
                if ( ( bitmasks[k] >> 0 ) & 1 )
                    pos += frame( 1, d );
                if ( ( bitmasks[k] >> 1 ) & 1 )
                    pos += frame( 2, d );
                cell.vertex_positions( k, d ) = pos;
            }
        }

        // rows of F^{-1} and dot products with origin
        const auto r0 = FT.solve_ge( DsVec<TF,ct_dim,Arch>::with_func( 2, []( PI i ) { return TF( i == 0 ? 1 : 0 ); } ) );
        const auto r1 = FT.solve_ge( DsVec<TF,ct_dim,Arch>::with_func( 2, []( PI i ) { return TF( i == 1 ? 1 : 0 ); } ) );
        const TF d0 = r0[ 0 ] * frame( 0, 0 ) + r0[ 1 ] * frame( 0, 1 );
        const TF d1 = r1[ 0 ] * frame( 0, 0 ) + r1[ 1 ] * frame( 0, 1 );

        // cut k = edge k→(k+1)%4: lower_y, upper_x, upper_y, lower_x
        cell.cut_planes( 0, 0 ) = - r1[ 0 ]; cell.cut_planes( 0, 1 ) = - r1[ 1 ]; cell.cut_planes( 0, 2 ) = - d1;
        cell.cut_planes( 1, 0 ) = + r0[ 0 ]; cell.cut_planes( 1, 1 ) = + r0[ 1 ]; cell.cut_planes( 1, 2 ) = + d0 + 1;
        cell.cut_planes( 2, 0 ) = + r1[ 0 ]; cell.cut_planes( 2, 1 ) = + r1[ 1 ]; cell.cut_planes( 2, 2 ) = + d1 + 1;
        cell.cut_planes( 3, 0 ) = - r0[ 0 ]; cell.cut_planes( 3, 1 ) = - r0[ 1 ]; cell.cut_planes( 3, 2 ) = - d0;
        for ( PI k = 0; k < 4; ++k )
            cell.cut_ids( k ) = cut_id;

        #ifdef SDOT_KEEP_FULL_CELL_INFO_FOR_2D_CASE
        for ( PI k = 0; k < 4; ++k ) {
            const PI k1 = ( k + 1 ) % 4, kp = ( k + 3 ) % 4;
            cell.edge_indices( k, 0 ) = k;
            cell.edge_indices( k, 1 ) = k1;
            cell.edge_indices( k, 2 ) = k;
            cell.vertex_indices( k, 0 ) = std::min( kp, k );
            cell.vertex_indices( k, 1 ) = std::max( kp, k );
        }
        #endif

        return;
    }

    // vertex_positions: origin + sum of selected axes; vertex_indices: cut 2b or 2b+1 per axis
    for ( PI k = 0; k < nb_vertices; ++k ) {
        for ( PI d = 0; d < dim; ++d ) {
            TF pos = frame( 0, d );
            for ( PI b = 0; b < dim; ++b )
                if ( ( k >> b ) & 1 )
                    pos += frame( 1 + b, d );
            cell.vertex_positions( k, d ) = pos;
        }
        for ( PI b = 0; b < dim; ++b )
            cell.vertex_indices( k, b ) = 2 * b + ( ( k >> b ) & 1 );
    }

    // edge_indices: edges in direction b, from vertex k (bit b=0) to k|(1<<b)
    for ( PI b = 0, e = 0; b < dim; ++b ) {
        for ( PI k = 0; k < nb_vertices; ++k ) {
            if ( ( k >> b ) & 1 ) continue;
            cell.edge_indices( e, 0 ) = k;
            cell.edge_indices( e, 1 ) = k | ( PI( 1 ) << b );
            for ( PI d = 0, col = 2; d < dim; ++d ) {
                if ( d == b ) continue;
                cell.edge_indices( e, col++ ) = 2 * d + ( ( k >> d ) & 1 );
            }
            ++e;
        }
    }

    // cut planes: row d of F^{-1} via shared FT
    for ( PI d = 0; d < dim; ++d ) {
        auto e_d = DsVec<TF,ct_dim,Arch>::with_func( dim, [d]( PI i ) {
            return i == d ? TF( 1 ) : TF( 0 );
        } );
        const auto row = FT.solve_ge( e_d );

        TF row_dot_origin = 0;
        for ( PI c = 0; c < dim; ++c )
            row_dot_origin += row[ c ] * frame( 0, c );

        for ( PI c = 0; c < dim; ++c )
            cell.cut_planes( 2 * d, c ) = -row[ c ];
        cell.cut_planes( 2 * d, dim ) = -row_dot_origin;
        cell.cut_ids( 2 * d ) = cut_id;

        for ( PI c = 0; c < dim; ++c )
            cell.cut_planes( 2 * d + 1, c ) = row[ c ];
        cell.cut_planes( 2 * d + 1, dim ) = row_dot_origin + 1;
        cell.cut_ids( 2 * d + 1 ) = cut_id;
    }
}

// grad_inputs, *grad_outputs
UTP void make_hypercube_backward( DTP &cell, const auto &frame, SI cut_id, const auto &grad_inp_frame, const auto &grad_out_cut_planes, const auto &grad_out_vertex_positions ) {
    TODO;
}


#undef UTP2
#undef DTP2
#undef UTP1
#undef DTP1
#undef UTP
#undef DTP

} // namespace sdot
