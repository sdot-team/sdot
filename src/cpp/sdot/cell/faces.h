#pragma once

#include "../support/RecursiveMapOfUniqueSortedIndices.h"
#include <numeric>
#include "Cell.h"

namespace sdot {

/// list of vertices for each face
template<class TF, int ct_dim, class Arch>
std::vector<std::vector<PI>> faces( const Cell<TF,ct_dim,Arch> &cell ) {
    constexpr int ct_nb_face_inds = ct_dim >= 2 ? ct_dim - 2 : -1;
    const PI nb_vertices = cell.nb_vertices();
    const PI nb_edges = cell.nb_edges();
    const PI nb_cuts = cell.nb_cuts();
    const PI dim = cell.dim();

    //
    if ( dim == 2 ) {
        std::vector<PI> res( cell.nb_vertices() );
        std::iota( res.begin(), res.end(), 0 );
        return { res };
    }

    //
    RecursiveMapOfUniqueSortedIndices<PI32,ct_nb_face_inds,Arch> item_map( dim );
    item_map.prepare_for( nb_cuts, nb_vertices ); // TODO: a posteriori update (nb faces may be > nb_vertices ?)

    // for each edge, find the connected faces
    std::vector<std::vector<PI>> faces_to_edges;
    for( PI num_edge = 0; num_edge < nb_edges; ++num_edge ) {
        auto face_inds = Vector<PI32,ct_nb_face_inds,Arch>::with_func( dim - 2, [&]( PI i ) {
            return cell.edge_indices( num_edge, i + ( i >= num_edge ) );
        } );

        // first time encounter -> add an entry in res
        auto ic = item_map[ face_inds ];
        if ( ! ic ) {
            ic = faces_to_edges.size();
            faces_to_edges.emplace_back();
        }

        //
        faces_to_edges[ ic ].push_back( num_edge );
    }

    // for each face, chain edges into an ordered vertex list
    std::vector<std::vector<PI>> res( faces_to_edges.size() );
    for ( PI f = 0; f < faces_to_edges.size(); ++f ) {
        const auto &edges = faces_to_edges[ f ];
        auto &verts = res[ f ];
        if ( edges.empty() )
            continue;

        const PI n = edges.size();
        verts.reserve( n );
        std::vector<bool> used( n, false );

        used[ 0 ] = true;
        PI head = cell.edge_indices( edges[ 0 ], 0 );
        PI tail = cell.edge_indices( edges[ 0 ], 1 );
        verts.push_back( head );
        verts.push_back( tail );

        for ( PI step = 1; step < n - 1; ++step ) {
            for ( PI j = 0; j < n; ++j ) {
                if ( used[ j ] )
                    continue;
                const PI va = cell.edge_indices( edges[ j ], 0 );
                const PI vb = cell.edge_indices( edges[ j ], 1 );
                if ( va == tail ) { tail = vb; verts.push_back( tail ); used[ j ] = true; break; }
                if ( vb == tail ) { tail = va; verts.push_back( tail ); used[ j ] = true; break; }
            }
        }
    }

    // orientation: CCW for dim==2 (signed area), outward normal for dim==3 (Newell + centroid)
    if ( dim == 2 ) {
        for ( auto &verts : res ) {
            TF area2 = 0;
            for ( PI i = 0; i < verts.size(); ++i ) {
                const PI j = ( i + 1 ) % verts.size();
                area2 += cell.vertex_positions( verts[ i ], 0 ) * TF( cell.vertex_positions( verts[ j ], 1 ) )
                       - cell.vertex_positions( verts[ j ], 0 ) * TF( cell.vertex_positions( verts[ i ], 1 ) );
            }
            if ( area2 < 0 )
                std::reverse( verts.begin(), verts.end() );
        }
    } else if ( dim == 3 ) {
        // cell centroid
        TF cx = 0, cy = 0, cz = 0;
        for ( PI v = 0; v < nb_vertices; ++v ) {
            cx += cell.vertex_positions( v, 0 );
            cy += cell.vertex_positions( v, 1 );
            cz += cell.vertex_positions( v, 2 );
        }
        const TF inv_nv = TF( 1 ) / nb_vertices;
        cx *= inv_nv; cy *= inv_nv; cz *= inv_nv;

        for ( auto &verts : res ) {
            if ( verts.size() < 3 ) continue;

            // Newell's method for face normal
            TF nx = 0, ny = 0, nz = 0;
            for ( PI i = 0; i < verts.size(); ++i ) {
                const PI j = ( i + 1 ) % verts.size();
                const TF xi = cell.vertex_positions( verts[ i ], 0 ), yi = cell.vertex_positions( verts[ i ], 1 ), zi = cell.vertex_positions( verts[ i ], 2 );
                const TF xj = cell.vertex_positions( verts[ j ], 0 ), yj = cell.vertex_positions( verts[ j ], 1 ), zj = cell.vertex_positions( verts[ j ], 2 );
                nx += ( yi - yj ) * ( zi + zj );
                ny += ( zi - zj ) * ( xi + xj );
                nz += ( xi - xj ) * ( yi + yj );
            }

            // face centroid
            TF fx = 0, fy = 0, fz = 0;
            for ( PI v : verts ) {
                fx += cell.vertex_positions( v, 0 );
                fy += cell.vertex_positions( v, 1 );
                fz += cell.vertex_positions( v, 2 );
            }
            const TF inv_nf = TF( 1 ) / verts.size();
            fx *= inv_nf; fy *= inv_nf; fz *= inv_nf;

            // if normal points inward, reverse
            if ( ( fx - cx ) * nx + ( fy - cy ) * ny + ( fz - cz ) * nz < 0 )
                std::reverse( verts.begin(), verts.end() );
        }
    }

    return res;
}

} // namespace sdot
