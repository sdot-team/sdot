#pragma once

#include <sdot/support/MapOfUniqueSortedIndices.h>
#include <sdot/generated_includes/Cell.h>
#include <sdot/support/P.h>
#include <algorithm>
#include <numeric>

namespace sdot {

/// Call ``func( face )`` for every non-empty face of ``cell``.
template<int ct_dim,typename Arch,typename TF,typename TI>
void for_each_face( const Cell<ct_dim,Arch,TF,TI> &cell, auto workspace, auto &&func ) {
    const PI dim = cell.dim();
    if ( dim == 2 ) {
        std::vector<PI> indices( cell.nb_vertices() );
        std::iota( indices.begin(), indices.end(), 0 );
        func( indices );
        return;
    }

    workspace.nb_links = 0;

    // on remplit une liste chainée d'edges
    // pour chaque face, on récupère l'index dans la liste chainée
    MapOfUniqueSortedIndices<( ct_dim > 0 ? ct_dim - 2 : -1 ),TI,Arch> face_map( workspace.map_items, workspace.nb_map_items, dim - 2, cell.nb_cuts );
    face_map.reserve_full_capacity();
    for( PI edge = 0; edge < cell.nb_edges; ++edge ) {
        DsVec<TI,( ct_dim >= 0 ? ct_dim - 1 : -1 ),Arch> edge_cut_indices = std::span( &cell.edge_indices( edge, 2 ), dim - 1 );
        for( PI ind_to_remove = 0; ind_to_remove < dim - 1; ++ind_to_remove ) {
            auto face_cut_indices = edge_cut_indices.without_index( ind_to_remove );
            auto index_in_links = face_map[ face_cut_indices ];

            const TI prev = index_in_links.has_a_value() ? TI( index_in_links ) : -1;
            const TI indl = workspace.nb_links.post_increment( 2 );
            workspace.links[ indl + 0 ] = prev;
            workspace.links[ indl + 1 ] = edge;
            index_in_links = indl;
        }
    }

    face_map.for_each_item( [&]( auto /*face_cut_indices*/, TI index_in_links ) {
        // reserve some room to store the vertices
        PI nb_vertices = 0;
        for ( TI idx = index_in_links; idx != TI( -1 ); idx = workspace.links[ idx ] )
            ++nb_vertices;
        const PI scratch = workspace.nb_links.post_increment( nb_vertices );

        const TI e0 = TI( workspace.links[ index_in_links + 1 ] );
        const TI vs = cell.edge_indices( e0, 0 ); // start vertex
        TI prev_edge = e0;
        TI vc = vs; // current vertex

        // first vertex
        nb_vertices = 0;
        workspace.links[ scratch + nb_vertices++ ] = vc;

        // extend the vertex chain until it closes; use a raw loop + break to stop early
        do {
            for ( TI idx = index_in_links; idx != TI( -1 ); idx = workspace.links[ idx ] ) {
                const TI e1 = TI( workspace.links[ idx + 1 ] );
                if ( e1 == prev_edge )
                    continue;

                const TI v0 = cell.edge_indices( e1, 0 );
                const TI v1 = cell.edge_indices( e1, 1 );
                if ( v0 == vc ) {
                    prev_edge = e1;
                    vc = v1;
                    break;
                }
                if ( v1 == vc ) {
                    prev_edge = e1;
                    vc = v0;
                    break;
                }
            }

            workspace.links[ scratch + nb_vertices++ ] = vc;
        } while ( vc != vs );

        func( std::span( &workspace.links[ scratch ], nb_vertices - 1 ) );
    } );
}

} // namespace sdot
