#pragma once

#include <sdot/generated_includes/Cell.h>
#include "Cell/Simplex.h"

namespace sdot {

TEMPLATE_PARAMETERS_OF_Cell
struct Cell {
    ATTRIBUTES_OF_Cell

    using       TF                     = typename T_vertex_positions::TF; ///< float type, recovered from attribute
    using       Pt                     = Vector<TF,ct_dim>; ///< point
    using       Ci                     = Vector<TI,ct_dim>; ///< cut indices

    //
    HD void     init_as_aligned_simplex( TI cut_id );
    HD void     init_as_unbounded      ();

    HD void     init_as_hypercube_bwd  ( const auto &frame, auto &p, const auto &batch_index );
    HD void     init_as_hypercube      ( const auto &frame, const auto &cut_id );


    // retrieve info in tensors --------------------------------------------------------------
    HD Pt       vertex_position        ( PI num_vertex ) const;
    HD Ci       vertex_cuts            ( PI num_vertex ) const;
    HD bool     vertex_inf             ( PI num_vertex ) const; ///< true if at least one cut is inf
    HD Pt       cut_dir                ( PI num_cut ) const;
    HD TF       cut_dot                ( PI num_cut ) const;
    HD TI       cut_id                 ( PI num_cut ) const;

    HD Pt       solve_position         ( PI num_vertex, auto &&add_func ) const;
    HD Pt       solve_position         ( PI num_vertex ) const;

    // info, computations --------------------------------------------------------------------
    HD void     for_each_simplex       ( auto &item_map, auto &&func ); ///< RecursiveMapOfUniqueSortedIndices<ct_dim-1,...>
    HD void     for_each_facet         ( auto &&func ); ///< func( facet_repr, cut_id )
    HD void     for_each_face          ( auto &&func ); ///< func( num_vertices, cut_indices_for_this_face )

    HD void     check_consistency      ();
    HD void     disp_cell              ();
    HD bool     contains               ( const Pt &p ) const;
    HD Pt       centroid               ();

    HD void     measure_bwd            ( auto &item_map, auto &&p );
    HD TF       measure                ( auto &item_map );

    T_d HD auto simplex_from_indices   ( const Vector<TI,d> &indices ) const;

    // modifications -------------------------------------------------------------------------
    HD void     get_data_from          ( const auto &cell );
    HD void     clear_cell             ();
    HD void     cut                    ( const auto &cut_dir, auto cut_dot, SI cut_id );

    // internal functions --------------------------------------------------------------------
    HD void     remove_unused_vertices ( PI nb_vertices_orig );
    HD void     remove_unused_cuts     ();

    HD void     apply_vertex_corr      ();
    HD void     apply_cut_corr         ();

    HD void     check_if_fully_closed  ();
    HD PI       register_the_new_cut   ( const auto &cut_dir, auto cut_dot, SI cut_id );
    HD void     grow_infinite_cuts     ( const auto &cut_dir, auto cut_dot );
    HD PI       scalar_products        ( auto &sps, const auto &cut_dir, auto cut_dot );
    HD void     process_edges          ( PI nc );
    HD void     swap_and_pop           ( auto &nb, auto &&move_row ); ///< generic swap-and-pop (indices_to_remove sorted ascending), fills ws.corr with old->new map
    HD void     cut_2d                 ( const auto &cut_dir, auto cut_dot, SI cut_id, PI nb_out );

    T_d HD void for_each_simplex_rec   ( const Vector<TI,d> &cut_indices, auto &simplex, PI num_vertex, auto &item_map, auto &&func );
    HD bool     already_in_simplex     ( auto &simplex, PI simplex_size, PI next_num_vertex );
};

// template<int ct_dim,class Arch,class TF,class TI>
// TF integral( CellWorker<ct_dim,Arch,TF,TI> &cell_worker, auto &&local_function );

} // namespace sdot

#include "Cell.cxx" // IWYU pragma: export
