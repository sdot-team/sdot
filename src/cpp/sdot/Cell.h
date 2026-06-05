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

    T_TAB HD void init_as_hypercube_bwd( const T &frame, A &p, const B &batch_index );
    T_TA  HD void init_as_hypercube    ( const T &frame, const A &cut_id );


    // retrieve info in tensors --------------------------------------------------------------
    HD Pt       vertex_position        ( PI num_vertex ) const;
    HD Ci       vertex_cuts            ( PI num_vertex ) const;
    HD bool     vertex_inf             ( PI num_vertex ) const; ///< true if at least one cut is inf
    HD Pt       cut_dir                ( PI num_cut ) const;
    HD TF       cut_dot                ( PI num_cut ) const;
    HD TI       cut_id                 ( PI num_cut ) const;

    T_T HD Pt   solve_position         ( PI num_vertex, T &&add_func ) const;
    HD Pt       solve_position         ( PI num_vertex ) const;

    // info, computations --------------------------------------------------------------------
    T_TA HD void for_each_facet_simplex( T &item_map, A &&func ); ///< func( simplex_vertex_indices, cut_index ); cut_index into cut_planes/cut_ids; reuses item_map.next via generation trick
    T_TA HD void for_each_simplex      ( T &item_map, A &&func ); ///< RecursiveMapOfUniqueSortedIndices<ct_dim-1,...>
    T_T  HD void for_each_face         ( T &&func ); ///< func( num_vertices, cut_indices_for_this_face )

    HD void     check_consistency      ();
    HD void     disp_cell              ();
    HD bool     contains               ( const Pt &p ) const;
    HD Pt       centroid               ();

    T_TAB HD void measure_bwd          ( T &item_map, A &&p, B &&batch_index );
    T_T HD TF   measure                ( T &item_map );

    T_d HD auto simplex_from_indices   ( const Vector<TI,d> &indices ) const;

    // modifications -------------------------------------------------------------------------
    T_T HD void get_data_from          ( const T &cell );
    HD void     clear_cell             ();
    T_TA HD void cut                   ( const T &cut_dir, A cut_dot, SI cut_id );

    // internal functions --------------------------------------------------------------------
    HD void     remove_unused_vertices ( PI nb_vertices_orig );
    HD void     remove_unused_cuts     ();

    HD void     apply_vertex_corr      ();
    HD void     apply_cut_corr         ();

    HD void     check_if_fully_closed  ();
    T_TA HD PI  register_the_new_cut   ( const T &cut_dir, A cut_dot, SI cut_id );
    T_TA HD void grow_infinite_cuts    ( const T &cut_dir, A cut_dot );
    T_TAB HD PI scalar_products        ( T &sps, const A &cut_dir, B cut_dot );
    HD void     process_edges          ( PI nc );
    T_TA HD void swap_and_pop          ( T &nb, A &&move_row ); ///< generic swap-and-pop (indices_to_remove sorted ascending), fills ws.corr with old->new map
    T_TA HD void cut_2d                ( const T &cut_dir, A cut_dot, SI cut_id, PI nb_out );

    template<int d,class S,class IM,class F> HD void for_each_simplex_rec( const Vector<TI,d> &cut_indices, S &simplex, PI num_vertex, IM &item_map, F &&func );
    T_T HD bool already_in_simplex     ( T &simplex, PI simplex_size, PI next_num_vertex );
};

// template<int ct_dim,class Arch,class TF,class TI>
// TF integral( CellWorker<ct_dim,Arch,TF,TI> &cell_worker, auto &&local_function );

} // namespace sdot

#include "Cell.cxx" // IWYU pragma: export
