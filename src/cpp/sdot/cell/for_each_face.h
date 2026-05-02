#pragma once

#include <sdot/generated_includes/Cell.h>
#include <algorithm>
#include <numeric>

namespace sdot {

/// Call ``func( face )`` for every non-empty face of ``cell``.
template<int ct_dim,typename Arch,typename TF,typename TI>
void for_each_face( const Cell<ct_dim,Arch,TF,TI> &cell, auto &&func ) {
    // const PI dim        = cell.dim();
    // const PI nb_verts   = cell.nb_vertices();
    // const PI nb_cuts    = cell.nb_cuts();

    // Facet facet;
    // for ( SI c = 0; c < SI( nb_cuts ); ++c ) {
    //     facet.defining_cut = c;
    //     facet.vertex_indices.clear();

    //     for ( PI v = 0; v < nb_verts; ++v ) {
    //         for ( PI d = 0; d < dim; ++d ) {
    //             if ( cell.vertex_inds( v, d ) == c ) {
    //                 facet.vertex_indices.push_back( v );
    //                 break;
    //             }
    //         }
    //     }

    //     if ( !facet.vertex_indices.empty() )
    //         func( facet );
    // }
    std::vector<PI> indices( cell.nb_vertices() );
    std::iota( indices.begin(), indices.end(), 0 );
    func( indices );
}

} // namespace sdot
