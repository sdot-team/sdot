#pragma once

#include "Facet.h"
#include "Cell.h"
#include <algorithm>

namespace sdot {

/// Call ``func( facet )`` for every non-empty facet of ``cell``.
///
/// A facet is the (dim-1)-face defined by a single cut.  The vertices of a
/// facet are exactly those cell vertices whose active-cut set contains that
/// cut's index.
///
/// Complexity: O( nb_cuts × nb_vertices × dim ).
template<class TF, int ct_dim, class Arch>
void for_each_facet( const Cell<TF,ct_dim,Arch> &cell, auto &&func ) {
    const PI dim        = cell.dim();
    const PI nb_verts   = cell.nb_vertices();
    const PI nb_cuts    = cell.nb_cuts();

    Facet facet;
    for ( SI c = 0; c < SI( nb_cuts ); ++c ) {
        facet.defining_cut = c;
        facet.vertex_indices.clear();

        for ( PI v = 0; v < nb_verts; ++v ) {
            for ( PI d = 0; d < dim; ++d ) {
                if ( cell.vertex_inds( v, d ) == c ) {
                    facet.vertex_indices.push_back( v );
                    break;
                }
            }
        }

        if ( !facet.vertex_indices.empty() )
            func( facet );
    }
}

} // namespace sdot
