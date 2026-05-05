#pragma once

#include <sdot/generated_includes/CellWorkspace.h>
#include <sdot/generated_includes/Cell.h>

namespace sdot {

template<int ct_dim,class Arch,class TF,class TI>
struct CellWorker {
    using Pt                    = DsVec<TF,ct_dim,Arch>; ///< point
    using Ci                    = DsVec<TI,ct_dim,Arch>; ///< cut indices

    // retrieve info in tensors --------------------------------------------------------------
    Pt    vertex_position       ( PI num_vertex ) const;
    Ci    vertex_indices        ( PI num_vertex ) const;
    bool  vertex_inf            ( PI num_vertex ) const; ///< true if at least one cut is inf
    Pt    cut_dir               ( PI num_cut ) const;
    TF    cut_dot               ( PI num_cut ) const;
    TI    cut_id                ( PI num_cut ) const;

    Pt    solve_position        ( PI num_vertex, auto &&add_func ) const;
    Pt    solve_position        ( PI num_vertex ) const;

    // info, computations --------------------------------------------------------------------
    void  for_each_simplex      ( auto &&func );
    void  for_each_facet        ( auto &&func ); ///< func( facet_repr, cut_id )
    void  for_each_face         ( auto &&func ); ///< func( num_vertices, cut_indices_for_this_face )

    void  check_consistency     ();
    void  disp_cell             ();
    TF    measure               ();

    // modifications -------------------------------------------------------------------------
    void  get_data_from         ( const auto &cell );
    void  clear_cell            ();
    void  cut                   ( const auto &cut_dir, auto cut_dot, SI cut_id );

    // internal functions --------------------------------------------------------------------
    void  remove_unused_vertices( PI nb_vertices_orig );
    void  remove_unused_edges   ();
    void  remove_unused_cuts    ();

    void  apply_vertex_corr     ();
    void  apply_cut_corr        ();

    void  check_if_fully_closed ();
    PI    register_the_new_cut  ( const auto &cut_dir, auto cut_dot, SI cut_id );
    void  grow_infinite_cuts    ( const auto &cut_dir, auto cut_dot );
    PI    scalar_products       ( const auto &cut_dir, auto cut_dot );
    void  process_edges         ( PI nc );
    void  swap_and_pop          ( auto &nb, auto &&move_row ); ///< generic swap-and-pop (indices_to_remove sorted ascending), fills ws.corr with old->new map
    void  cut_2d                ( const auto &cut_dir, auto cut_dot, SI cut_id, PI nb_out );

    void  for_each_simplex_rec  ( const auto &cut_indices, auto &simplex, PI simplex_size, PI num_vertex, auto &item_map, auto &&func );
    bool  already_in_simplex    ( auto &simplex, PI simplex_size, PI next_num_vertex );

    // attributes ---------------------------------------------------------------------------
    Cell<ct_dim,Arch,TF,TI>&    cell;
    CellWorkspace<Arch,TF,TI>&  ws;
    const PI                    dim;
};

// template<int ct_dim,class Arch,class TF,class TI>
// TF integral( CellWorker<ct_dim,Arch,TF,TI> &cell_worker, auto &&local_function );

} // namespace sdot

#include "CellWorker.cxx" // IWYU pragma: export
