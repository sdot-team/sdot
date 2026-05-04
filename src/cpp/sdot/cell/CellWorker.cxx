#pragma once

#include "../support/RecursiveMapOfUniqueSortedIndices.h"
#include "../support/SimpleSquareMatrix.h"
#include "../support/index.h"
#include "../support/P.h"

#include "CellBoundary.h"
#include "CellWorker.h"

#include <numeric>

namespace sdot {

#define UTP template<int ct_dim,class Arch,class TF,class TI>
#define DTP CellWorker<ct_dim,Arch,TF,TI>

UTP typename DTP::Pt DTP::vertex_position( PI num_vertex ) const {
    return cell.vertex_positions.row( num_vertex );
}

UTP typename DTP::Ci DTP::vertex_indices( PI num_vertex ) const {
    if constexpr ( ct_dim == 2 )
        return { Values(), ( num_vertex + cell.nb_vertices - 1 ) % cell.nb_vertices, num_vertex };
    return cell.vertex_indices.row( num_vertex );
}

UTP bool DTP::vertex_inf( PI num_vertex ) const {
    Ci ci = vertex_indices( num_vertex );
    for ( PI d = 0; d < dim; ++d )
        if ( cell.cut_ids( ci[ d ] ) == CellBoundary::INFINITE )
            return true;
    return false;
}

UTP typename DTP::Pt DTP::cut_dir( PI num_cut ) const {
    return std::span( &cell.cut_planes( num_cut, 0 ), dim );
}

UTP TF DTP::cut_dot( PI num_cut ) const {
    return cell.cut_planes( num_cut, dim );
}

UTP bool DTP::already_in_simplex( auto &simplex, PI simplex_size, PI next_num_vertex ) {
    for( PI simplex_ind = 0; simplex_ind < simplex_size; ++simplex_ind )
        if ( next_num_vertex == simplex[ simplex_ind ] )
            return true;
    return false;
}

/// Fan triangulation — recursive core.
UTP void DTP::for_each_simplex_rec( const auto &cut_indices, auto &simplex, PI simplex_size, PI num_vertex, auto &item_map, auto &&func ) {
    // register the new vertex
    simplex[ simplex_size++ ] = num_vertex;

    const PI nb_cut_indices = cut_indices.size();
    if ( nb_cut_indices == 0 ) {
        func( simplex );
        return;
    }

    for( PI ind_to_remove = 0; ind_to_remove < nb_cut_indices; ++ind_to_remove ) {
        // first time we see this item -> use vertex `num_vertex`as reference for this item
        auto new_cut_indices = cut_indices.without_index( ind_to_remove );
        auto ic = item_map[ new_cut_indices ];
        if ( ! ic ) {
            ic = num_vertex;
            continue;
        }

        // else, try to make a new simplex
        const PI next_num_vertex = ic;
        if( already_in_simplex( simplex, simplex_size, next_num_vertex ) )
            continue;

        // and continue the recursion
        for_each_simplex_rec( new_cut_indices, simplex, simplex_size, next_num_vertex, item_map, func );
    }
}

UTP void DTP::for_each_simplex( auto &&func ) {
    constexpr int ct_simplex = ct_dim >= 0 ? ct_dim + 1 : -1;
    const PI nb_vertices = cell.nb_vertices();
    const PI dim = cell.dim();

    if ( cell.nb_vertices() == 0 )
        return;

    // make a list
    RecursiveMapOfUniqueSortedIndices<ct_dim,TI,Arch> item_map( ws.map_items, ws.nb_map_items, dim, cell.nb_cuts );
    item_map.reserve( cell.nb_vertices );

    DsVec<PI,ct_simplex,Arch> simplex( Size(), dim + 1 );
    for( PI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex ) {
        DsVec<PI32,ct_dim,Arch> cut_indices( cell.vertex_indices.row( num_vertex ) );
        for_each_simplex_rec( cut_indices, simplex, 0, num_vertex, item_map, func );
    }
}

UTP TF DTP::measure() {
    const PI nb_vertices = cell.nb_vertices();
    const PI dim = cell.dim();

    // infinite cell
    if ( ! cell.is_fully_closed() )
        return std::numeric_limits<TF>::infinity();

    // 2D: shoelace formula
    if ( dim == 2 ) {
        TF sum = 0;
        for ( PI i = 0; i < nb_vertices; ++i ) {
            const PI j = ( i + 1 ) % nb_vertices;
            sum += cell.vertex_positions( i, 0 ) * cell.vertex_positions( j, 1 )
                 - cell.vertex_positions( j, 0 ) * cell.vertex_positions( i, 1 );
        }
        return sum / 2;
    }

    // nD: fan triangulation
    SimpleSquareMatrix<TF,ct_dim,Arch> M( Size(), dim );
    TF sum = 0;

    for_each_simplex( [&]( const auto &simplex ) {
        const PI v0 = simplex[ 0 ];
        auto M = SimpleSquareMatrix<TF,ct_dim,Arch>::with_func( dim, [&]( PI row, PI col ) {
            return cell.vertex_positions( simplex[ col + 1 ], row ) - cell.vertex_positions( v0, row );
        } );
        sum += std::abs( M.determinant() );
    } );

    return sum / factorial( dim );
}

UTP void DTP::check_if_fully_closed() {
    for( PI num_cut = 0; num_cut < cell.nb_cuts; ++num_cut )
        if ( cell.cut_ids[ num_cut ] == CellBoundary::INFINITE )
            return;
    cell.is_fully_closed() = true;
}

UTP void DTP::cut( const auto &cut_dir, auto cut_dot, SI cut_id ) {
    const PI nb_vertices = cell.nb_vertices();

    // if nothing or everything is cut and the cell has infinite boundaries,
    // grow them so the cut has its correct effect, then recompute
    if ( ! cell.is_fully_closed() )
        grow_infinite_cuts( cut_dir, cut_dot );

    //
    PI nb_out = scalar_products( cut_dir, cut_dot );

    if ( nb_out == 0 )
        return;

    if ( nb_out == nb_vertices )
        return clear_cell();

    // 2D shortcut: vertices ordered CCW, cut_planes(k,:) = edge k→(k+1)%nb invariant maintained by cut_2d
    if ( dim == 2 ) {
        cut_2d( cut_dir, cut_dot, cut_id, nb_out );
    } else {
        // store the new cut
        const PI new_cut_index = register_the_new_cut( cut_dir, cut_dot, cut_id );

        // process edges: add new vertices on cut, collect exterior edges into ws.indices_to_remove
        process_edges( new_cut_index );

        // remove exterior edges (ws.corr unused here)
        remove_unused_edges();

        // remove exterior vertices, ws.corr = vertex old->new (also updates edge vertex refs)
        remove_unused_vertices( nb_vertices );
        apply_vertex_corr();

        // remove unused cuts, ws.corr = cut old->new; apply to vertex_indices and edge cut indices
        remove_unused_cuts();
        apply_cut_corr();
    }

    // check if closed
    if ( ! cell.is_fully_closed() )
        check_if_fully_closed();
}

UTP PI DTP::scalar_products( const auto &cut_dir, auto cut_dot ) {
    const PI nb_vertices = cell.nb_vertices();
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
UTP void DTP::swap_and_pop( auto &nb, auto &&move_row ) {
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

UTP void DTP::process_edges( PI nc ) {
    const PI nb_edges = cell.nb_edges();

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

UTP void DTP::remove_unused_edges() {
    swap_and_pop( cell.nb_edges, [&]( PI dst, PI src ) {
        for ( PI d = 0; d < cell.edge_indices.size( 1 ); ++d )
            cell.edge_indices( dst, d ) = std::move( cell.edge_indices( src, d ) );
    } );
}

UTP void DTP::remove_unused_vertices( PI nb_vertices_orig ) {
    const PI dim = cell.dim();
    ws.nb_indices_to_remove = 0;
    for ( PI n = 0; n < nb_vertices_orig; ++n )
        if ( ws.sps[ n ] > 0 )
            ws.indices_to_remove[ ws.nb_indices_to_remove++ ] = n;

    // -> update ws.corr
    swap_and_pop( cell.nb_vertices, [&]( PI dst, PI src ) {
        for ( PI d = 0; d < dim; ++d )
            cell.vertex_positions( dst, d ) = std::move( cell.vertex_positions( src, d ) );
        for ( PI d = 0; d < dim; ++d )
            cell.vertex_indices( dst, d ) = std::move( cell.vertex_indices( src, d ) );
    } );

}

UTP void DTP::apply_vertex_corr() {
    // update edge vertex references
    for ( PI e = 0; e < cell.nb_edges; ++e ) {
        cell.edge_indices( e, 0 ) = ws.corr[ cell.edge_indices( e, 0 ) ];
        cell.edge_indices( e, 1 ) = ws.corr[ cell.edge_indices( e, 1 ) ];
    }
}

UTP void DTP::remove_unused_cuts() {
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

    swap_and_pop( cell.nb_cuts, [&]( PI dst, PI src ) {
        for ( PI d = 0; d <= dim; ++d )
            cell.cut_planes( dst, d ) = std::move( cell.cut_planes( src, d ) );
        cell.cut_ids( dst ) = std::move( cell.cut_ids( src ) );
    } );
}

UTP void DTP::apply_cut_corr() {
    const PI dim = cell.dim();

    for ( PI v = 0; v < cell.nb_vertices(); ++v )
        for ( PI d = 0; d < dim; ++d )
            cell.vertex_indices( v, d ) = ws.corr[ cell.vertex_indices( v, d ) ];

    for ( PI e = 0; e < cell.nb_edges(); ++e )
        for ( PI d = 0; d < dim - 1; ++d )
            cell.edge_indices( e, 2 + d ) = ws.corr[ cell.edge_indices( e, 2 + d ) ];
}

UTP void DTP::cut_2d( const auto &cut_dir, auto cut_dot, SI cut_id, PI nb_out ) {
    const SI old_nb_vertices = cell.nb_vertices;

    // helper to copy vertex_positions and cut_planes/cut_ids together
    auto copy_vertex_data = [&]( PI dst, PI src ) {
        for ( PI d = 0; d < 2; ++d )
            cell.vertex_positions( dst, d ) = cell.vertex_positions( src, d );
    };
    auto copy_cut_data = [&]( PI dst, PI src ) {
        for ( PI d = 0; d < 3; ++d )
            cell.cut_planes( dst, d ) = cell.cut_planes( src, d );
        cell.cut_ids( dst ) = cell.cut_ids( src );
    };
    auto copy_vertex_and_cut_data = [&]( PI dst, PI src ) {
        copy_vertex_data( dst, src );
        copy_cut_data( dst, src );
    };

    auto ext = [&]( PI num_vertex ) {
        return ws.sps[ num_vertex ] > 0;
    };

    // need to add a vertex
    if ( nb_out == 1 ) {
        const SI n1 = index( ws.sps, []( TF sp ) { return sp > 0; } );
        const SI n0 = ( n1 + old_nb_vertices - 1 ) % old_nb_vertices;
        const SI n2 = ( n1 + 1 ) % old_nb_vertices;

        // compute new vertex positions
        const TF sp0 = ws.sps[ n0 ], sp1 = ws.sps[ n1 ], sp2 = ws.sps[ n2 ];
        const TF d01 = sp0 / ( sp1 - sp0 );
        const TF d12 = sp1 / ( sp2 - sp1 );
        TF p01[ 2 ], p12[ 2 ];
        for ( PI d = 0; d < 2; ++d ) {
            const TF p0 = cell.vertex_positions( n0, d );
            const TF p1 = cell.vertex_positions( n1, d );
            const TF p2 = cell.vertex_positions( n2, d );
            p01[ d ] = p0 - d01 * ( p1 - p0 );
            p12[ d ] = p1 - d12 * ( p2 - p1 );
        }

        // get room for the new point and the new cut
        cell.nb_vertices = old_nb_vertices + 1;
        cell.nb_edges = old_nb_vertices + 1;
        cell.nb_cuts = old_nb_vertices + 1;
        for( SI nn = old_nb_vertices; --nn > n1; )
            copy_vertex_and_cut_data( nn + 1, nn );
        copy_cut_data( n1 + 1, n1 );

        // set data for the new point and the new cut
        for ( PI d = 0; d < 2; ++d ) {
            cell.vertex_positions( n1 + 0, d ) = p01[ d ];
            cell.vertex_positions( n1 + 1, d ) = p12[ d ];
            cell.cut_planes( n1, d ) = cut_dir[ d ];
        }
        cell.cut_planes( n1, 2 ) = cut_dot;
        cell.cut_ids[ n1 ] = cut_id;
        return;
    }

    // we stay on the same number of vertices
    if ( nb_out == 2 ) {
        const SI n0 = index( [&]( SI n0 ) { return ! ext( n0 ) && ext( ( n0 + 1 ) % old_nb_vertices ); } );
        const SI n1 = ( n0 + 1 ) % old_nb_vertices;
        const SI n2 = ( n0 + 2 ) % old_nb_vertices;
        const SI n3 = ( n0 + 3 ) % old_nb_vertices;

        // compute new vertex positions
        const TF sp0 = ws.sps[ n0 ], sp1 = ws.sps[ n1 ], sp2 = ws.sps[ n2 ], sp3 = ws.sps[ n3 ];
        const TF d01 = sp0 / ( sp1 - sp0 );
        const TF d23 = sp2 / ( sp3 - sp2 );
        TF p01[ 2 ], p23[ 2 ];
        for ( PI d = 0; d < 2; ++d ) {
            const TF p0 = cell.vertex_positions( n0, d );
            const TF p1 = cell.vertex_positions( n1, d );
            const TF p2 = cell.vertex_positions( n2, d );
            const TF p3 = cell.vertex_positions( n3, d );
            p01[ d ] = p0 - d01 * ( p1 - p0 );
            p23[ d ] = p2 - d23 * ( p3 - p2 );
        }

        // set data for the new point and the new cut
        for ( PI d = 0; d < 2; ++d ) {
            cell.vertex_positions( n1, d ) = p01[ d ];
            cell.vertex_positions( n2, d ) = p23[ d ];
            cell.cut_planes( n1, d ) = cut_dir[ d ];
        }
        cell.cut_planes( n1, 2 ) = cut_dot;
        cell.cut_ids[ n1 ] = cut_id;
        return;
    }

    SI n0 = index( [&]( SI n0 ) { return ! ext( n0 ) && ext( ( n0 + 1 ) % old_nb_vertices ); } );
    SI n1 = ( n0 + 1 ) % old_nb_vertices;
    SI n2 = index( [&]( SI n2 ) { return ext( n2 ) && ! ext( ( n2 + 1 ) % old_nb_vertices ); } );
    SI n3 = ( n2 + 1 ) % old_nb_vertices;

    // compute new vertex positions
    const TF sp0 = ws.sps[ n0 ], sp1 = ws.sps[ n1 ], sp2 = ws.sps[ n2 ], sp3 = ws.sps[ n3 ];
    const TF d01 = sp0 / ( sp1 - sp0 );
    const TF d23 = sp2 / ( sp3 - sp2 );
    TF p01[ 2 ], p23[ 2 ];
    for ( PI d = 0; d < 2; ++d ) {
        const TF p0 = cell.vertex_positions( n0, d );
        const TF p1 = cell.vertex_positions( n1, d );
        const TF p2 = cell.vertex_positions( n2, d );
        const TF p3 = cell.vertex_positions( n3, d );
        p01[ d ] = p0 - d01 * ( p1 - p0 );
        p23[ d ] = p2 - d23 * ( p3 - p2 );
    }

    // remove intermediate points
    if ( n1 < n2 ) {
        const PI nb_to_remove = n2 - ( n1 + 1 ), nb_new_vertices = old_nb_vertices - nb_to_remove;
        cell.nb_vertices = nb_new_vertices;
        cell.nb_edges = nb_new_vertices;
        cell.nb_cuts = nb_new_vertices;

        copy_cut_data( n2 - nb_to_remove, n2 );
        for( SI nn = n2 + 1; nn < old_nb_vertices; ++nn )
            copy_vertex_and_cut_data( nn - nb_to_remove, nn );

        // set data for the new point and the new cut
        n2 = n1 + 1;
        for ( PI d = 0; d < 2; ++d ) {
            cell.vertex_positions( n1, d ) = p01[ d ];
            cell.vertex_positions( n2, d ) = p23[ d ];
            cell.cut_planes( n1, d ) = cut_dir[ d ];
        }
        cell.cut_planes( n1, 2 ) = cut_dot;
        cell.cut_ids[ n1 ] = cut_id;
    } else {
        const PI nb_to_remove = n2 + ( old_nb_vertices - n1 - 1 ), nb_new_vertices = old_nb_vertices - nb_to_remove;
        cell.nb_vertices = nb_new_vertices;
        cell.nb_edges = nb_new_vertices;
        cell.nb_cuts = nb_new_vertices;

        if ( n2 ) {
            for( SI nn = n2; nn <= n1; ++nn )
                copy_vertex_and_cut_data( nn - n2, nn );
            n0 -= n2;
            n1 -= n2;
            n2 = 0;
            n3 = 1;
        }

        // set data for the new point and the new cut
        for ( PI d = 0; d < 2; ++d ) {
            cell.vertex_positions( n1, d ) = p01[ d ];
            cell.vertex_positions( n2, d ) = p23[ d ];
            cell.cut_planes( n1, d ) = cut_dir[ d ];
        }
        cell.cut_planes( n1, 2 ) = cut_dot;
        cell.cut_ids[ n1 ] = cut_id;
    }
}

UTP void DTP::clear_cell() {
    cell.is_fully_closed() = 1;
    cell.nb_vertices() = 0;
    cell.nb_edges() = 0;
    cell.nb_cuts() = 0;
}

UTP PI DTP::register_the_new_cut( const auto &cut_dir, auto cut_dot, SI cut_id ) {
    PI res = cell.nb_cuts()++;
    for ( PI d = 0; d < dim; ++d )
        cell.cut_planes( res, d ) = cut_dir[ d ];
    cell.cut_planes( res, dim ) = cut_dot;
    cell.cut_ids( res ) = cut_id;
    return res;
}

UTP DTP::Pt DTP::solve_position( PI num_vertex, auto &&add_func ) const {
    Ci ci = vertex_indices( num_vertex );

    auto M  = SimpleSquareMatrix<TF,ct_dim,Arch>::with_func( dim, [&]( PI r, PI c ) {
        return cell.cut_planes( ci[ r ], c );
    } );

    auto V = DsVec<TF,ct_dim,Arch>::with_func( dim, [&]( PI i ) {
        return cell.cut_planes( ci[ i ], dim ) + add_func( ci[ i ] );
    } );

    return M.solve_ge( V );
}

UTP DTP::Pt DTP::solve_position( PI num_vertex ) const {
    return solve_position( num_vertex, []( auto ) { return 0; } );
}

// grow_infinite_cuts: parametrise by a scalar s such that INFINITE cut i gains s * norm_2( n_i )
// added to its offset (uniform spatial growth). sp(s) = cut_dir · v(s) - cut_dot is linear in s;
// evaluate at s=0 and s=1 via SimpleSquareMatrix::solve_ge. For each vertex with sp(0)<0 and
// sp(1)>0 the crossing is at s* = -sp(0)/(sp(1)-sp(0)). Apply s_grow = max(s*) to all INFINITE cuts.
UTP void DTP::grow_infinite_cuts( const auto &new_cut_dir, auto new_cut_dot ) {
    const PI nb_vertices = cell.nb_vertices();

    // check to grow enough so that all the vertices stay on the same side
    TF s_grow = 0; // std::numeric_limits<TF>::max();
    bool need_to_grow = false;
    for ( PI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex ) {
        if ( ! vertex_inf( num_vertex ) )
            continue;

        const Pt p1 = solve_position( num_vertex, [&]( PI num_cut ) { return cell.cut_ids( num_cut ) == CellBoundary::INFINITE ? norm_2( cut_dir( num_cut ) ) : 0; } );
        const TF s0 = dot( vertex_position( num_vertex ), new_cut_dir ) - new_cut_dot;
        const TF s1 = dot( p1, new_cut_dir ) - new_cut_dot;

        // if it can cross 0
        const TF ds = s1 - s0;
        if ( ds && ( s0 > 0 ) != ( ds > 0 )  ) {
            const TF s_trial = - s0 / ( s1 - s0 );
            if ( s_grow < s_trial )
                s_grow = s_trial;
            need_to_grow = true;
            break;
        }
    }

    //
    if ( need_to_grow ) {
        s_grow += 1;

        // add s_grow * norm_2( n_c ) to each INFINITE cut's offset
        for ( PI num_cut = 0; num_cut < cell.nb_cuts(); ++num_cut )
            if ( cell.cut_ids( num_cut ) == CellBoundary::INFINITE )
                cell.cut_planes( num_cut, dim ) += s_grow * norm_2( cut_dir( num_cut ) );

        // recompute vertex positions for all vertices with any INFINITE adjacent cut
        for ( PI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex )
            if ( vertex_inf( num_vertex ) )
                cell.vertex_positions.row( num_vertex ).get_data_from( solve_position( num_vertex ) );
    }
}

UTP void DTP::disp_cell() {
    P( cell.nb_vertices() );
    for( PI i = 0; i < cell.nb_vertices(); ++i ) {
        auto pos = DsVec<TF,ct_dim,Arch>::with_func( 2, [&]( PI d ) { return cell.vertex_positions( i, d ); } );
        auto cut = DsVec<TF,ct_dim+1,Arch>::with_func( 3, [&]( PI d ) { return cell.cut_planes( i, d ); } );
        P( pos, cut );
    }
}

UTP void DTP::check_consistency() {
    const PI nb_vertices = cell.nb_vertices();

    auto get_cut_inds = [&]( PI v, PI *out ) {
        if ( dim == 2 ) {
            out[ 0 ] = ( v + nb_vertices - 1 ) % nb_vertices;
            out[ 1 ] = v;
        } else {
            for ( PI d = 0; d < dim; ++d )
                out[ d ] = cell.vertex_indices( v, d );
        }
    };

    for( PI v = 0; v < cell.nb_vertices(); ++v ) {
        PI ci[ ct_dim ];
        get_cut_inds( v, ci );

        auto M = SimpleSquareMatrix<TF,ct_dim,Arch>::with_func( dim, [&]( PI r, PI c ) { return cell.cut_planes( ci[ r ], c ); } );
        auto V = DsVec<TF,ct_dim,Arch>::with_func( dim, [&]( PI i ) { return cell.cut_planes( ci[ i ], dim ); } );
        const auto pos = M.solve_ge( V );

        for ( PI d = 0; d < dim; ++d )
            P( cell.vertex_positions( v, d ), pos[ d ] );
    }
}

// disp_cell( cell );
// check_consistency( cell );

// _cut( cell, ws, DsVec<TF,2,Arch>( Values(), 1, 0 ), 0.5, 1 );
// disp_cell( cell );
// check_consistency( cell );

// _cut( cell, ws, DsVec<TF,2,Arch>( Values(), 1, 0 ), 0.3, 2 );
// disp_cell( cell );
// check_consistency( cell );

// // _cut( cell, ws, DsVec<TF,2,Arch>( Values(), 1, 1 ), 0.1, 3 );
// _cut( cell, ws, DsVec<TF,2,Arch>( Values(), -1, +1 ), -0.25, 3 );
// disp_cell( cell );
// check_consistency( cell );

#undef UTP
#undef DTP

} // namespace sdot
