#pragma once

#include "support/RecursiveMapOfUniqueSortedIndices.h"
#include <sdot/generated_includes/Cell.h>

namespace sdot {

PARAMETERS_DECLARATION_OF_Cell // template<int ct_dim,class Arch,class TF,class TI>
struct Cell {
    // static constexpr int ct_dim = ct_dim_value;
    ATTRIBUTES_OF_Cell

    using    Pt                     = Vector<TF,Arch,ct_dim>; ///< point
    using    Ci                     = Vector<TI,Arch,ct_dim>; ///< cut indices

    //
    void     init_as_aligned_simplex( TI cut_id );
    void     init_as_unbounded      ();

    // retrieve info in tensors --------------------------------------------------------------
    Pt       vertex_position        ( PI num_vertex ) const;
    Ci       vertex_cuts            ( PI num_vertex ) const;
    bool     vertex_inf             ( PI num_vertex ) const; ///< true if at least one cut is inf
    Pt       cut_dir                ( PI num_cut ) const;
    TF       cut_dot                ( PI num_cut ) const;
    TI       cut_id                 ( PI num_cut ) const;

    Pt       solve_position         ( PI num_vertex, auto &&add_func ) const;
    Pt       solve_position         ( PI num_vertex ) const;

    // info, computations --------------------------------------------------------------------
    void     for_each_simplex       ( RecursiveMapOfUniqueSortedIndices<ct_dim-1,TI,Arch> &item_map, auto &&func ); ///<
    void     for_each_facet         ( auto &&func ); ///< func( facet_repr, cut_id )
    void     for_each_face          ( auto &&func ); ///< func( num_vertices, cut_indices_for_this_face )

    void     check_consistency      ();
    void     disp_cell              ();
    bool     contains               ( const Pt &p ) const;
    Pt       centroid               ();
    HD TF    measure                ( RecursiveMapOfUniqueSortedIndices<ct_dim-1,TI,Arch> &item_map );

    T_d auto simplex_from_indices   ( const Vector<TI,Arch,d> &indices ) const;

    // modifications -------------------------------------------------------------------------
    void     get_data_from          ( const auto &cell );
    void     clear_cell             ();
    void     cut                    ( const auto &cut_dir, auto cut_dot, SI cut_id );

    // internal functions --------------------------------------------------------------------
    void     remove_unused_vertices ( PI nb_vertices_orig );
    void     remove_unused_cuts     ();

    void     apply_vertex_corr      ();
    void     apply_cut_corr         ();

    void     check_if_fully_closed  ();
    PI       register_the_new_cut   ( const auto &cut_dir, auto cut_dot, SI cut_id );
    void     grow_infinite_cuts     ( const auto &cut_dir, auto cut_dot );
    PI       scalar_products        ( auto &sps, const auto &cut_dir, auto cut_dot );
    void     process_edges          ( PI nc );
    void     swap_and_pop           ( auto &nb, auto &&move_row ); ///< generic swap-and-pop (indices_to_remove sorted ascending), fills ws.corr with old->new map
    void     cut_2d                 ( const auto &cut_dir, auto cut_dot, SI cut_id, PI nb_out );

    void     for_each_simplex_rec   ( const auto &cut_indices, auto &simplex, PI simplex_size, PI num_vertex, auto &item_map, auto &&func );
    bool     already_in_simplex     ( auto &simplex, PI simplex_size, PI next_num_vertex );
};

// template<int ct_dim,class Arch,class TF,class TI>
// TF integral( CellWorker<ct_dim,Arch,TF,TI> &cell_worker, auto &&local_function );

} // namespace sdot

#include "Cell.cxx" // IWYU pragma: export
