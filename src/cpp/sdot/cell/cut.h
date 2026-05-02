#pragma once

#include <sdot/generated_includes/CutWorkspace.h>
#include "../support/MapOfUniqueSortedIndices.h"
#include <sdot/generated_includes/Cell.h>
// #include "CutWorkspace.h"
#include "../support/P.h"
#include <numeric>

namespace sdot::_cut_detail {

template<int ct_dim,class Arch,class TF,class TI>
PI scalar_products( const Cell<ct_dim,Arch,TF,TI> &cell, const auto &cut_dir, auto cut_dot, CutWorkspace<Arch,TF,TI> &ws ) {
    const PI nb_vertices = cell.nb_vertices();
    const PI dim = cell.dim();
    ws.reservation = nb_vertices;
    PI nb_out = 0;
    for ( PI v = 0; v < nb_vertices; ++v ) {
        TF sp = cell.vertex_positions( v, 0 ) * cut_dir[ 0 ];
        for ( PI d = 1; d < dim; ++d )
            sp += cell.vertex_positions( v, d ) * cut_dir[ d ];
        sp -= cut_dot;
        ws.sps[ v ] = sp;
        nb_out += ( sp > 0 );
    }
    return nb_out;
}

// generic swap-and-pop (indices_to_remove sorted ascending), fills ws.corr with old->new map
template<class Arch,class TF,class TI>
void swap_and_pop( CutWorkspace<Arch,TF,TI> &ws, auto &nb, auto &&move_row ) {
    const PI nb_initial = PI( nb );
    ws.reservation = nb_initial;
    std::iota( ws.corr.data(), ws.corr.data() + nb_initial, 0 );
    while ( ws.nb_indices_to_remove() ) {
        const PI dst = ws.indices_to_remove[ --ws.nb_indices_to_remove ];
        const PI src = --nb;
        ws.corr[ src ] = dst;
        if ( dst != src )
            move_row( dst, src );
    }
    // When a src was itself placed there by a prior step, corr[src]=dst only records
    // the intermediate hop. Follow each chain to its fixed point (corr[j]==j).
    // Chains are strictly decreasing (dst<src in every non-trivial step) so no cycles.
    for ( PI i = 0; i < nb_initial; ++i ) {
        PI j = ws.corr[ i ];
        while ( ws.corr[ j ] != j )
            j = ws.corr[ j ];
        ws.corr[ i ] = j;
    }
}

template<int ct_dim,class Arch,class TF,class TI>
void process_edges( const Cell<ct_dim,Arch,TF,TI> &cell, PI nc, CutWorkspace<Arch,TF,TI> &ws ) {
    const PI nb_edges = cell.nb_edges();
    const PI dim = cell.dim();
    MapOfUniqueSortedIndices<ct_dim-2,TI,Arch> face_map( ws.map_items, ws.nb_map_items, dim - 2, nc );
    face_map.reserve( cell.nb_vertices() + nb_edges );

    ws.nb_indices_to_remove = 0;
    for ( PI num_edge = 0; num_edge < nb_edges; ++num_edge ) {
        const PI n0 = cell.edge_indices( num_edge, 0 );
        const PI n1 = cell.edge_indices( num_edge, 1 );
        const TF s0 = ws.sps[ n0 ];
        const TF s1 = ws.sps[ n1 ];
        const bool e0 = s0 > 0;
        const bool e1 = s1 > 0;
        if ( e0 == e1 ) {
            if ( e0 )
                ws.indices_to_remove[ ws.nb_indices_to_remove++ ] = num_edge;
            continue;
        }

        // add the new vertex
        const PI nn = cell.nb_vertices()++;
        for ( PI d = 0; d < dim; ++d ) {
            const TF p0 = cell.vertex_positions( n0, d );
            const TF p1 = cell.vertex_positions( n1, d );
            cell.vertex_positions( nn, d ) = p0 - s0 / ( s1 - s0 ) * ( p1 - p0 );
        }
        for ( PI d = 0; d < dim - 1; ++d )
            cell.vertex_indices( nn, d ) = cell.edge_indices( num_edge, 2 + d );
        cell.vertex_indices( nn, dim - 1 ) = nc;

        // update the edge
        cell.edge_indices( num_edge, ! e0 ) = nn;

        // register the node / add new edge for each connected face
        for ( PI ind_to_remove = 0; ind_to_remove < dim - 1; ++ind_to_remove ) {
            constexpr PI ct_face_dim = ct_dim > 0 ? ct_dim - 2 : -1;
            auto face_inds = DsVec<PI,ct_face_dim,Arch>::with_func( dim - 2, [&]( PI i ) {
                return cell.edge_indices( num_edge, 2 + i + ( i >= ind_to_remove ) );
            } );

            auto face_corr = face_map[ face_inds ];
            if ( ! face_corr ) {
                face_corr = nn;
                continue;
            }

            const PI ne = cell.nb_edges()++;
            cell.edge_indices( ne, 0 ) = face_corr;
            cell.edge_indices( ne, 1 ) = nn;
            for ( PI d = 0; d < dim - 2; ++d )
                cell.edge_indices( ne, 2 + d ) = face_inds[ d ];
            cell.edge_indices( ne, 2 + dim - 2 ) = nc;
        }
    }
}

template<int ct_dim,class Arch,class TF,class TI>
void remove_unused_edges( Cell<ct_dim,Arch,TF,TI> &cell, CutWorkspace<Arch,TF,TI> &ws ) {
    _cut_detail::swap_and_pop( ws, cell.nb_edges, [&]( PI dst, PI src ) {
        for ( PI d = 0; d < cell.edge_indices.size( 1 ); ++d )
            cell.edge_indices( dst, d ) = std::move( cell.edge_indices( src, d ) );
    } );
}

template<int ct_dim,class Arch,class TF,class TI>
void remove_unused_vertices( Cell<ct_dim,Arch,TF,TI> &cell, PI nb_vertices_orig, CutWorkspace<Arch,TF,TI> &ws ) {
    const PI dim = cell.dim();
    ws.nb_indices_to_remove = 0;
    for ( PI n = 0; n < nb_vertices_orig; ++n )
        if ( ws.sps[ n ] > 0 )
            ws.indices_to_remove[ ws.nb_indices_to_remove++ ] = n;

    // -> update ws.corr
    swap_and_pop( ws, cell.nb_vertices, [&]( PI dst, PI src ) {
        for ( PI d = 0; d < dim; ++d )
            cell.vertex_positions( dst, d ) = std::move( cell.vertex_positions( src, d ) );
        for ( PI d = 0; d < dim; ++d )
            cell.vertex_indices( dst, d ) = std::move( cell.vertex_indices( src, d ) );
    } );

}

template<int ct_dim,class Arch,class TF,class TI>
void apply_vertex_corr( Cell<ct_dim,Arch,TF,TI> &cell, CutWorkspace<Arch,TF,TI> &ws ) {
    // update edge vertex references
    for ( PI e = 0; e < cell.nb_edges; ++e ) {
        cell.edge_indices( e, 0 ) = ws.corr[ cell.edge_indices( e, 0 ) ];
        cell.edge_indices( e, 1 ) = ws.corr[ cell.edge_indices( e, 1 ) ];
    }
}

template<int ct_dim,class Arch,class TF,class TI>
void remove_unused_cuts( Cell<ct_dim,Arch,TF,TI> &cell, CutWorkspace<Arch,TF,TI> &ws ) {
    const PI dim = cell.dim();

    for( PI i = 0; i < cell.nb_cuts(); ++i )
       ws.used_flags[ i ] = false;

    for ( PI v = 0; v < cell.nb_vertices(); ++v )
        for ( PI d = 0; d < dim; ++d )
            ws.used_flags[ cell.vertex_indices( v, d ) ] = true;

    ws.nb_indices_to_remove = 0;
    for ( PI c = 0; c < cell.nb_cuts(); ++c )
        if ( ! ws.used_flags[ c ] )
            ws.indices_to_remove[ ws.nb_indices_to_remove++ ] = c;

    swap_and_pop( ws, cell.nb_cuts, [&]( PI dst, PI src ) {
        for ( PI d = 0; d <= dim; ++d )
            cell.cut_planes( dst, d ) = std::move( cell.cut_planes( src, d ) );
        cell.cut_ids( dst ) = std::move( cell.cut_ids( src ) );
    } );
}

template<int ct_dim,class Arch,class TF,class TI>
void apply_cut_corr( Cell<ct_dim,Arch,TF,TI> &cell, CutWorkspace<Arch,TF,TI> &ws ) {
    const PI dim = cell.dim();

    for ( PI v = 0; v < cell.nb_vertices(); ++v )
        for ( PI d = 0; d < dim; ++d )
            cell.vertex_indices( v, d ) = ws.corr[ cell.vertex_indices( v, d ) ];

    for ( PI e = 0; e < cell.nb_edges(); ++e )
        for ( PI d = 0; d < dim - 1; ++d )
            cell.edge_indices( e, 2 + d ) = ws.corr[ cell.edge_indices( e, 2 + d ) ];
}

// 2D shortcut: vertices ordered CCW, cut_planes(k,:) = edge k→(k+1)%nb invariant
template<int ct_dim,class Arch,class TF,class TI>
void cut_2d( Cell<ct_dim,Arch,TF,TI> &cell, const auto &cut_dir, auto cut_dot, SI cut_id, CutWorkspace<Arch,TF,TI> &ws ) {
    const PI nb = cell.nb_vertices();

    // find int→out (n0) and out→int (n2) transitions
    PI n0 = 0, n2 = 0;
    for ( PI k = 0; k < nb; ++k ) {
        if ( ws.sps[ k ] <= 0 && ws.sps[ ( k + 1 ) % nb ] > 0 ) n0 = k;
        if ( ws.sps[ k ] >  0 && ws.sps[ ( k + 1 ) % nb ] <= 0 ) n2 = k;
    }
    const PI n1 = ( n0 + 1 ) % nb;
    const PI n3 = ( n2 + 1 ) % nb;

    // compute new vertex positions
    const TF s0 = ws.sps[ n0 ], s1 = ws.sps[ n1 ], s2 = ws.sps[ n2 ], s3 = ws.sps[ n3 ];
    TF nv1_pos[ 2 ], nv2_pos[ 2 ];
    for ( PI d = 0; d < 2; ++d ) {
        nv1_pos[ d ] = cell.vertex_positions( n0, d ) - s0 / ( s1 - s0 ) * ( cell.vertex_positions( n1, d ) - cell.vertex_positions( n0, d ) );
        nv2_pos[ d ] = cell.vertex_positions( n2, d ) - s2 / ( s3 - s2 ) * ( cell.vertex_positions( n3, d ) - cell.vertex_positions( n2, d ) );
    }

    // save NC and cut_n2 plane data before any writes (n2 may alias a write target)
    TF n2_plane[ 3 ];
    for ( PI d = 0; d < 3; ++d ) {
        n2_plane[ d ] = cell.cut_planes( n2, d );
    }
    const SI nc_id_val = cut_id;
    const SI n2_id_val = cell.cut_ids( n2 );

    // copy vertex_positions and cut_planes/cut_ids together
    auto copy_vc = [&]( PI dst, PI src ) {
        for ( PI d = 0; d < 2; ++d )
            cell.vertex_positions( dst, d ) = cell.vertex_positions( src, d );
        for ( PI d = 0; d < 3; ++d )
            cell.cut_planes( dst, d ) = cell.cut_planes( src, d );
        cell.cut_ids( dst ) = cell.cut_ids( src );
    };

    if ( n0 < n3 ) {
        // non-wrapping: new = [0..n0, nv1, nv2, n3..nb-1]
        //               cuts = [0..n0, NC, cut_n2, n3..nb-1]
        const PI nb_after  = nb - n3;
        const PI dst_after = n0 + 3;
        if ( n3 > dst_after )
            for ( PI k = 0; k < nb_after; ++k )
                copy_vc( dst_after + k, n3 + k );
        else if ( n3 < dst_after )
            for ( PI k = nb_after; k-- > 0; )
                copy_vc( dst_after + k, n3 + k );

        for ( PI d = 0; d < 2; ++d )
            cell.vertex_positions( n0 + 1, d ) = nv1_pos[ d ];
        for ( PI d = 0; d < 2; ++d )
            cell.cut_planes( n0 + 1, d ) = cut_dir[ d ];
        cell.cut_planes( n0 + 1, 2 ) = cut_dot;
        cell.cut_ids( n0 + 1 ) = nc_id_val;
        for ( PI d = 0; d < 2; ++d )
            cell.vertex_positions( n0 + 2, d ) = nv2_pos[ d ];
        for ( PI d = 0; d < 3; ++d )
            cell.cut_planes( n0 + 2, d ) = n2_plane[ d ];
        cell.cut_ids( n0 + 2 ) = n2_id_val;

        const PI new_nb = n0 + 3 + nb_after;
        cell.nb_vertices() = new_nb;
        cell.nb_cuts() = new_nb;
    } else {
        // wrapping: new = [nv2, n3..n0, nv1]
        //           cuts = [cut_n2, n3..n0, NC]
        const PI nb_interior = n0 - n3 + 1;
        // n3==0 → right shift → iterate backward to avoid overlap
        if ( n3 > 0 )
            for ( PI k = 0; k < nb_interior; ++k )
                copy_vc( 1 + k, n3 + k );
        else
            for ( PI k = nb_interior; k-- > 0; )
                copy_vc( 1 + k, k );

        for ( PI d = 0; d < 2; ++d )
            cell.vertex_positions( 0, d ) = nv2_pos[ d ];
        for ( PI d = 0; d < 3; ++d )
            cell.cut_planes( 0, d ) = n2_plane[ d ];
        cell.cut_ids( 0 ) = n2_id_val;
        for ( PI d = 0; d < 2; ++d )
            cell.vertex_positions( nb_interior + 1, d ) = nv1_pos[ d ];
        for ( PI d = 0; d < 2; ++d )
            cell.cut_planes( nb_interior + 1, d ) = cut_dir[ d ];
        cell.cut_planes( nb_interior + 1, 2 ) = cut_dot;
        cell.cut_ids( nb_interior + 1 ) = nc_id_val;

        const PI new_nb = nb_interior + 2;
        cell.nb_vertices() = new_nb;
        cell.nb_cuts() = new_nb;
    }

    const PI new_nb = cell.nb_vertices();
    cell.nb_edges() = new_nb;
}

void clear_cell( auto &cell, PI dim ) {
    cell.is_fully_closed() = 1;
    cell.nb_vertices() = 0;
    cell.nb_edges() = 0;
    cell.nb_cuts() = 0;
}

PI register_the_new_cut( auto &cell, PI dim, const auto &cut_dir, auto cut_dot, SI cut_id ) {
    PI res = cell.nb_cuts()++;
    for ( PI d = 0; d < dim; ++d )
        cell.cut_planes( res, d ) = cut_dir[ d ];
    cell.cut_planes( res, dim ) = cut_dot;
    cell.cut_ids( res ) = cut_id;
    return res;
}

} // namespace sdot::_cut_detail

namespace sdot {

template<int ct_dim,class Arch,class TF,class TI>
void cut( Cell<ct_dim,Arch,TF,TI> &cell, CutWorkspace<Arch,TF,TI> &ws, const auto &cut_dir, auto cut_dot, SI cut_id ) {
    const PI nb_vertices = cell.nb_vertices();
    const PI dim = cell.dim();

    const PI nb_out = _cut_detail::scalar_products( cell, cut_dir, cut_dot, ws );

    // no change -> do nothing
    if ( nb_out == 0 )
        return;

    // everything is outside
    if ( nb_out == nb_vertices )
        return _cut_detail::clear_cell( cell, dim );

    // 2D shortcut: vertices ordered CCW, cut_planes(k,:) = edge k→(k+1)%nb invariant maintained by cut_2d
    if ( dim == 2 )
        return _cut_detail::cut_2d( cell, cut_dir, cut_dot, cut_id, ws );

    // else, store the new cut
    const PI new_cut_index = _cut_detail::register_the_new_cut( cell, dim, cut_dir, cut_dot, cut_id );

    // process edges: add new vertices on cut, collect exterior edges into ws.indices_to_remove
    _cut_detail::process_edges( cell, new_cut_index, ws );

    // remove exterior edges (ws.corr unused here)
    _cut_detail::remove_unused_edges( cell, ws );

    // remove exterior vertices, ws.corr = vertex old->new (also updates edge vertex refs)
    _cut_detail::remove_unused_vertices( cell, nb_vertices, ws );
    _cut_detail::apply_vertex_corr( cell, ws );

    // remove unused cuts, ws.corr = cut old->new; apply to vertex_indices and edge cut indices
    _cut_detail::remove_unused_cuts( cell, ws );
    _cut_detail::apply_cut_corr( cell, ws );
}

} // namespace sdot
