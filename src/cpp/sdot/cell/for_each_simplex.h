#pragma once

#include "Cell.h"

namespace sdot {

namespace _detail {

/// True if vertex ``num_vertex`` has cut ``num_cut`` in its active set.
/// vertex_inds rows are sorted by construction — linear scan is still fast (dim ≤ ~8).
template<class TF,int ct_dim,class Arch>
HD bool vertex_has_cut( const Cell<TF,ct_dim,Arch> &cell, PI num_vertex, SI num_cut ) {
    const PI dim = cell.dim();
    for ( PI d = 0; d < dim; ++d )
        if ( cell.vertex_inds( num_vertex, d ) == num_cut )
            return true;
    return false;
}

/// True if vertex ``num_vertex`` belongs to the sub-polytope defined by ``fixed_cuts``
/// (i.e. all fixed cuts appear in num_vertex's active set).
template<class TF,int ct_dim,class Arch,int ct_fixed>
HD bool vertex_in_subpoly( const Cell<TF,ct_dim,Arch> &cell, PI num_vertex, const DsVec<SI,ct_fixed,Arch> &fixed_cuts ) {
    for ( PI k = 0; k < fixed_cuts.size(); ++k )
        if ( ! vertex_has_cut( cell, num_vertex, fixed_cuts[ k ] ) )
            return false;
    return true;
}

/// Fan triangulation — recursive core.
///
/// Template parameter ``ct_fixed`` tracks the number of cuts already committed
/// (= recursion depth = current simplex position - 1).  For ct_dim ≥ 0 this
/// gives stack-allocated DsVecs at every level; for ct_dim == -1 all ct values
/// are -1 (dynamic, heap-allocated as usual).
///
/// ``simplex`` collects vertex indices; ``pos`` is the next write position.
/// ``func`` is called with ``(simplex, pos)`` when a complete simplex is ready.
template<int ct_fixed, class TF, int ct_dim, class Arch, int ct_simplex, class Func>
HD void for_each_simplex_rec( const Cell<TF,ct_dim,Arch> &cell, const DsVec<SI,ct_fixed,Arch> &fixed_cuts, DsVec<PI,ct_simplex,Arch> &simplex, PI pos, Func &func ) {
    const PI nb_vertices = cell.nb_vertices();
    const PI dim = cell.dim();

    // ── find v0: first vertex in the current sub-polytope ───────────────────
    PI v0 = PI( -1 );
    for ( PI v = 0; v < nb_vertices; ++v ) {
        if ( vertex_in_subpoly( cell, v, fixed_cuts ) ) {
            v0 = v;
            break;
        }
    }
    if ( v0 == PI( -1 ) )
        return;

    simplex[ pos ] = v0;

    // ── base case: 0-simplex ─────────────────────────────────────────────────
    if constexpr ( ct_fixed >= 0 && ct_dim >= 0 && ct_fixed == ct_dim ) {
        func( simplex );
        return;
    } else {
        if ( fixed_cuts.size() == dim ) {
            func( simplex );
            return;
        }

        // ── enumerate facet cuts: present in some sub-poly vertex ≠ v0
        //    but absent from v0's own cut set.
        //    We visit each qualifying cut exactly once by taking the first
        //    sub-poly vertex (other than v0) that introduces it.
        constexpr int new_ct_fixed = ct_fixed >= 0 ? ct_fixed + 1 : -1;

        for ( PI v = 0; v < nb_vertices; ++v ) {
            if ( v == v0 )
                continue;
            if ( ! vertex_in_subpoly( cell, v, fixed_cuts ) )
                continue;

            for ( PI d = 0; d < dim; ++d ) {
                const SI c = cell.vertex_inds( v, d );

                // skip if already a fixed cut
                bool already_fixed = false;
                for ( PI k = 0; k < fixed_cuts.size(); ++k ) {
                    if ( fixed_cuts[ k ] == c ) {
                        already_fixed = true;
                        break;
                    }
                }
                if ( already_fixed )
                    continue;

                // skip if c is in v0's cut set (these define facets that contain v0)
                if ( vertex_has_cut( cell, v0, c ) )
                    continue;

                // deduplicate: process c only for the first sub-poly vertex ≠ v0 that has it
                bool is_first = true;
                for ( PI vv = 0; vv < v && is_first; ++vv ) {
                    if ( vv == v0 )
                        continue;
                    if ( ! vertex_in_subpoly( cell, vv, fixed_cuts ) )
                        continue;
                    if ( vertex_has_cut( cell, vv, c ) )
                        is_first = false;
                }
                if ( ! is_first )
                    continue;

                // recurse into the facet defined by cut c
                auto new_fixed = fixed_cuts.with_pushed_value( c );
                for_each_simplex_rec<new_ct_fixed>( cell, new_fixed, simplex, pos + 1, func );
            }
        }
    }
}

} // namespace _detail


/// Call ``func( simplex )`` for every simplex in a fan triangulation of ``cell``.
///
/// Uses DsVec throughout: stack-allocated when ``ct_dim ≥ 0``, heap otherwise.
/// No dynamic allocation in the cut-set vectors when ct_dim is known.
///
/// Precondition: ``cell.is_fully_closed()``.
template<class TF, int ct_dim, class Arch>
void for_each_simplex( const Cell<TF,ct_dim,Arch> &cell, auto &&func ) {
    constexpr int ct_simplex = ct_dim >= 0 ? ct_dim + 1 : -1;
    const PI dim = cell.dim();

    DsVec<PI,ct_simplex,Arch> simplex( Size(), dim + 1 );
    DsVec<SI,0,Arch> fixed_cuts( Size(), 0 );

    _detail::for_each_simplex_rec<0>( cell, fixed_cuts, simplex, 0, func );
}

} // namespace sdot
