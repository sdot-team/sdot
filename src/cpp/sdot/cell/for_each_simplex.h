#pragma once

#include "../support/RecursiveMapOfUniqueSortedIndices.h"
#include <sdot/generated_includes/Cell.h>

namespace sdot {

namespace _detail {

bool already_in_simplex( auto &simplex, PI simplex_size, PI next_num_vertex ) {
    for( PI simplex_ind = 0; simplex_ind < simplex_size; ++simplex_ind )
        if ( next_num_vertex == simplex[ simplex_ind ] )
            return true;
    return false;
}

/// Fan triangulation — recursive core.
void for_each_simplex_rec( const auto &cut_indices, auto &simplex, PI simplex_size, PI num_vertex, auto &item_map, auto &&func ) {
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

} // namespace _detail


/// Call ``func( simplex )`` for every simplex in a fan triangulation of ``cell``.
///
template<int ct_dim,class Arch,class TF,class TI>
void for_each_simplex( const Cell<ct_dim,Arch,TF,TI> &cell, auto &&cell_workspace, auto &&func ) {
    constexpr int ct_simplex = ct_dim >= 0 ? ct_dim + 1 : -1;
    const PI nb_vertices = cell.nb_vertices();
    const PI dim = cell.dim();

    if ( cell.nb_vertices() == 0 )
        return;

    // make a list
    RecursiveMapOfUniqueSortedIndices<ct_dim,TI,Arch> item_map( cell_workspace.map_items, cell_workspace.nb_map_items, dim, cell.nb_cuts );
    item_map.reserve( cell.nb_vertices );

    DsVec<PI,ct_simplex,Arch> simplex( Size(), dim + 1 );
    for( PI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex ) {
        DsVec<PI32,ct_dim,Arch> cut_indices( cell.vertex_indices.row( num_vertex ) );
        _detail::for_each_simplex_rec( cut_indices, simplex, 0, num_vertex, item_map, func );
    }
}

} // namespace sdot
